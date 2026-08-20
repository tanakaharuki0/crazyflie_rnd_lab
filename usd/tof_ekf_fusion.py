#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToF(法線ベクトル)由来のyaw情報を、CrazyflieのEKF(kalman_core.py に移植した
kalman_core.c)へ「疎結合」で融合するプロトタイプ。

疎結合の設計方針 (tof_odometry.py での議論を踏まえた設計):
  - tof_odometry.py の run_odometry() はそのまま独立に走らせ、その出力
    (kabsch_dR_ypr, rot_src) を「完成した答え」として消費するだけで、
    tof_odometry.py 自体には一切手を入れない。
  - EKFへ渡すのは「累積yaw」そのものではなく、rank_R>=2 (rot_src==1, ジャイロの
    pitch/rollに依存しない真に独立な幾何フィット) のフレームだけで得られる
    "前回のToF成功フレームからのΔyaw" というローカルな増分。これを
    kalman_core.c の mm_yaw_error.c (kalmanCoreUpdateWithYawError) と同じ
    形の「yaw誤差」スカラー観測として EKF の D2 (yaw方向の姿勢誤差状態) を
    更新する。
  - rank_R<2 (rot_src==4, solve_yaw_onlyがジャイロのpitch/rollを真値として
    使うフォールバック) のフレームは、EKF自身の出力に依存した情報が混ざる
    (循環参照になる) ため更新に使わない。

比較のため、同じジャイロ/加速度列から
  (a) ToF融合なしのEKF (gyroのみ)
  (b) ToF融合ありのEKF
の2つを同時に走らせ、tof_odometry.py の真値プロファイル (find_rotation_klmn_s /
true_yaw_profile) との誤差を比較する。

flow deckのデータ(motion.deltaX/deltaY等)が使えるかどうかは --use-flow-deck
で切り替える (既定はOFF。log50.csv等flow deck列が無いログでは自動的に無効)。

使い方:
    python3 tof_ekf_fusion.py log50.csv
    python3 tof_ekf_fusion.py log50.csv --use-flow-deck
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tofodo
from kalman_core import KalmanCore, KalmanCoreParams, DEG_TO_RAD, GRAVITY_MAGNITUDE

OUT_DIR = tofodo.OUT_DIR

# ToF由来yaw誤差観測の標準偏差モデル: 有効ペア数が多いほど信頼度が高いとみなす
YAW_MEAS_STD_BASE_DEG = 3.0
YAW_MEAS_STD_MIN_DEG = 1.0
# rank_R<2 (solve_yaw_only, ジャイロのpitch/rollを既知として使う) のフレームを
# 観測に使う場合、その追加の前提のぶん標準偏差を割り増すペナルティ倍率
RANK_DEFICIENT_STD_MULT = 2.0

FLOW_DPIXEL_COLS = ("motion.deltaX", "motion.deltaY")
FLOW_STD_DEG = 0.1  # [pixels] mm_flow.cのデフォルト的な値の目安 (要調整)


def run_ekf_pair(df_raw, t_raw_s, res, idx, use_flow_deck=False, quad_is_flying=False,
                  include_rank_deficient=False):
    """gyroのみのEKFと、ToF yaw融合ありのEKFを同じ入力で走らせ、両方のyaw時系列を返す。

    df_raw/t_raw_s: 間引き前の生ログ (acc.*, gyro.*, 利用可能ならflow列)。
    res, idx: tof_odometry.run_odometry() の出力と、そのフレームに対応する
              df_raw の行インデックス (tof_frames() が返す idx)。
    include_rank_deficient: Trueにすると rot_src==4 (rank_R<2, solve_yaw_only
              フォールバック。法線が1方向しか無いため、ジャイロの生積分由来の
              pitch/rollを既知として使いyawだけを幾何学的に解いたもの) のフレーム
              も観測として使う。この場合、gyro由来のpitch/rollを信じるという
              追加の前提が乗るため、標準偏差に RANK_DEFICIENT_STD_MULT 倍のペナルティを
              かけて信頼度を下げる (既定False: rot_src==1 だけを使う従来の挙動)。

    Returns: dict(t=..., yaw_gyro_only=..., yaw_tof_fused=..., n_yaw_updates=int)
    """
    have_flow = use_flow_deck and all(c in df_raw.columns for c in FLOW_DPIXEL_COLS)
    if use_flow_deck and not have_flow:
        print("警告: flow deckの列が見つからないため --use-flow-deck は無効化されます",
              file=sys.stderr)

    n = len(df_raw)
    acc = df_raw[["acc.x", "acc.y", "acc.z"]].to_numpy(dtype=float)
    gyro = df_raw[["gyro.x", "gyro.y", "gyro.z"]].to_numpy(dtype=float)

    # ToFフレーム(間引き後)の raw行index -> そのフレームの情報を辞書化。
    # kabsch_dR_ypr は「1つ前のToFフレーム(祭 prev1/prev2 のどちらか)」からの
    # フレーム間Δyawであり、rot_src==1(rank_R>=2, ジャイロに依存しない独立な
    # 幾何フィット)のときだけ信頼できる。そのため「直前のToFフレーム時点での
    # 融合後yaw」をアンカーとして全フレームで更新し続け、rot_src==1のフレーム
    # だけそのアンカー+Δyawを観測として使う。
    tof_frame_info = {}
    for f, row in enumerate(idx):
        tof_frame_info[int(row)] = dict(
            rot_src=int(res["rot_src"][f]),
            dyaw_deg=float(res["kabsch_dR_ypr"][f, 0]),
            n_pairs=int(res["n_pairs"][f]),
        )

    ekf_gyro = KalmanCore(KalmanCoreParams(), now_s=t_raw_s[0])
    ekf_tof = KalmanCore(KalmanCoreParams(), now_s=t_raw_s[0])

    yaw_gyro_only = np.zeros(n)
    yaw_tof_fused = np.zeros(n)
    yaw_at_prev_tof_frame_deg = 0.0
    n_yaw_updates = 0

    for i in range(n):
        t = float(t_raw_s[i])
        a = tuple(acc[i] * GRAVITY_MAGNITUDE)
        g = tuple(gyro[i] * DEG_TO_RAD)

        ekf_gyro.predict(a, g, t, quad_is_flying=quad_is_flying)
        ekf_tof.predict(a, g, t, quad_is_flying=quad_is_flying)

        ekf_gyro.add_process_noise(t)
        ekf_tof.add_process_noise(t)

        if have_flow and i > 0:
            dt_flow = t - float(t_raw_s[i - 1])
            dpx = float(df_raw["motion.deltaX"].iloc[i])
            dpy = float(df_raw["motion.deltaY"].iloc[i])
            if dt_flow > 0 and (dpx != 0 or dpy != 0):
                ekf_gyro.update_with_flow(dt_flow, dpx, dpy, FLOW_STD_DEG, FLOW_STD_DEG, g)
                ekf_tof.update_with_flow(dt_flow, dpx, dpy, FLOW_STD_DEG, FLOW_STD_DEG, g)

        if i in tof_frame_info:
            info = tof_frame_info[i]
            ekf_tof.finalize()  # ここまでの更新をqへ畳み込んでから現在のyawを読む
            current_yaw = ekf_tof.externalize_ypr_deg()[0]
            usable_src = (1, 4) if include_rank_deficient else (1,)
            if info["rot_src"] in usable_src and not np.isnan(info["dyaw_deg"]):
                predicted_yaw = yaw_at_prev_tof_frame_deg + info["dyaw_deg"]
                yaw_error_deg = tofodo.angle_diff_deg(predicted_yaw, current_yaw)
                std_dev_deg = max(YAW_MEAS_STD_BASE_DEG / np.sqrt(max(info["n_pairs"], 1)),
                                   YAW_MEAS_STD_MIN_DEG)
                if info["rot_src"] == 4:
                    # rank_R<2: ジャイロ由来のpitch/rollを既知として使う追加の前提が
                    # 乗るぶん、信頼度を下げる (標準偏差を大きくする)
                    std_dev_deg *= RANK_DEFICIENT_STD_MULT
                ekf_tof.update_with_yaw_error(np.radians(yaw_error_deg), np.radians(std_dev_deg))
                ekf_tof.finalize()
                current_yaw = ekf_tof.externalize_ypr_deg()[0]
                n_yaw_updates += 1
            yaw_at_prev_tof_frame_deg = current_yaw

        ekf_gyro.finalize()
        ekf_tof.finalize()

        yaw_gyro_only[i] = ekf_gyro.externalize_ypr_deg()[0]
        yaw_tof_fused[i] = ekf_tof.externalize_ypr_deg()[0]

    return dict(t=t_raw_s - t_raw_s[0], yaw_gyro_only=yaw_gyro_only,
                yaw_tof_fused=yaw_tof_fused, n_yaw_updates=n_yaw_updates)


def plot_comparison(ekf_out, true_klmn, out_path, t_range=tofodo.TRUE_ERR_T_RANGE,
                     fused_label="EKF (ToF yaw融合)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "IPAGothic", "TakaoGothic", "VL Gothic",
                 "Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False

    t = ekf_out["t"]
    true_yaw = tofodo.true_yaw_profile(t, *true_klmn)
    lo, hi = t_range
    mask = (t >= lo) & (t <= hi)
    t_m = t[mask]

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))

    ax[0].plot(t, true_yaw, lw=1.3, color="k", ls=":", label="真値 yaw")
    ax[0].plot(t, ekf_out["yaw_gyro_only"], lw=1.1, color="tab:orange", label="EKF (gyroのみ)")
    ax[0].plot(t, ekf_out["yaw_tof_fused"], lw=1.1, color="tab:blue", label=fused_label)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("yaw [deg]")
    ax[0].set_title(f"累積yaw: EKF (gyroのみ) vs {fused_label} vs 真値")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    specs = [
        ("EKF (gyroのみ)", ekf_out["yaw_gyro_only"], "tab:orange"),
        (fused_label, ekf_out["yaw_tof_fused"], "tab:blue"),
    ]
    for name, yaw, color in specs:
        err = tofodo.angle_diff_deg(yaw, true_yaw)
        err_m = np.abs(err[mask])
        integral = float(np.trapz(err_m, t_m))
        mean_err = float(np.mean(err_m))
        label = f"{name}  (∫|err|dt={integral:.1f} deg·s, mean={mean_err:.2f} deg)"
        ax[1].plot(t_m, err_m, lw=1.2, color=color, label=label)

    for a in ax:
        for name, tv in zip("klmn", true_klmn):
            a.axvline(tv, color="0.4", ls=":", lw=0.8)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("|yaw error| vs 真値 [deg]")
    ax[1].set_title(f"絶対誤差の時間変化 (t={lo:g}〜{hi:g}s)")
    ax[1].set_xlim(lo, hi)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"EKF比較グラフを保存: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="ToF yawのEKFへの疎結合フュージョン (プロトタイプ)")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log50.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+", default=[0, 1, 2, 5, 6, 7])
    ap.add_argument("--gate-deg", type=float, default=tofodo.GATE_DEG)
    ap.add_argument("--use-flow-deck", action="store_true",
                     help="flow deckのデータ(motion.deltaX/Y)が使えるなら融合に使う (既定OFF)")
    ap.add_argument("--quad-is-flying", action="store_true",
                     help="EKFのpredictを飛行中モードで動かす (既定は静止/卓上実験モード)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    path = args.csv
    if not os.path.isfile(path):
        cand = os.path.join(tofodo.CSV_DIR, os.path.basename(path))
        if not os.path.isfile(cand):
            sys.exit(f"ファイルが見つかりません: {args.csv}")
        path = cand
    df_raw = pd.read_csv(path)
    t_raw_s = df_raw["timestamp"].to_numpy(float) / 1000.0

    rng = np.random.default_rng(args.seed)
    arrs = tofodo.sensor_arrays(df_raw, args.sensors)
    idx, fresh = tofodo.tof_frames(arrs, args.sensors, t_raw_s)
    planes, plane_stats, _, _ = tofodo.extract_planes(df_raw, args.sensors, rng, arrs, idx, fresh)

    true_klmn = tofodo.find_rotation_klmn_s(df_raw, t_raw_s)
    if None in true_klmn:
        sys.exit("真値プロファイルの折れ点(k,l,m,n)を検出できませんでした。")

    df_tof = df_raw.iloc[idx].reset_index(drop=True)
    t_tof_s = df_tof["timestamp"].to_numpy(float) / 1000.0
    gyro_ref = tofodo.gyro_reference(df_tof, t_tof_s)
    att = gyro_ref["ypr"] if gyro_ref is not None else None

    res = tofodo.run_odometry(planes, args.sensors, df_tof["timestamp"].to_numpy(),
                               att=att, gate_deg=args.gate_deg)

    n_rank2 = int((res["rot_src"] == 1).sum())
    print(f"入力: {path} (生ログ{len(df_raw)}行, ToFフレーム{len(df_tof)})")
    print(f"  ToF yaw誤差観測として使うフレーム数 (rot_src==1, rank_R>=2): {n_rank2}")
    print(f"  真値の折れ点 k,l,m,n [s]: {[round(v, 2) for v in true_klmn]}")

    ekf_out = run_ekf_pair(df_raw, t_raw_s, res, idx,
                            use_flow_deck=args.use_flow_deck,
                            quad_is_flying=args.quad_is_flying)
    print(f"  ToF yaw観測の適用回数: {ekf_out['n_yaw_updates']}")

    lo, hi = tofodo.TRUE_ERR_T_RANGE
    t = ekf_out["t"]
    mask = (t >= lo) & (t <= hi)
    true_yaw = tofodo.true_yaw_profile(t, *true_klmn)
    for name, yaw in (("EKF(gyroのみ)", ekf_out["yaw_gyro_only"]),
                       ("EKF(ToF融合)", ekf_out["yaw_tof_fused"])):
        err = np.abs(tofodo.angle_diff_deg(yaw, true_yaw))[mask]
        print(f"  {name}: mean|err|={err.mean():.2f} deg, ∫|err|dt={np.trapz(err, t[mask]):.1f} deg·s")

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_ekf_fusion.png")
    plot_comparison(ekf_out, true_klmn, out)


if __name__ == "__main__":
    main()
