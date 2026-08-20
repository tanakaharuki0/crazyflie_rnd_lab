#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tof_odometry.py の「Kabsch自体をロバスト化」版。

tof_odometry.py の run_odometry() は、法線ペア(①)・KabschのΔR(②)それぞれに
ついて「ジャイロ予測との角度ズレが gate_deg 未満か」で二値的に採否を決め、
ダメなら1段だけ前(prev2)に遡って救済する、という手作りのフォールバックを
持っていた(prev1/prev2)。これは一時しのぎで、3フレーム連続で悪いと破綻する。

本モジュールはこの二値的な足切りを撤廃し、代わりに kabsch_normals()/
solve_translation() 自体を IRLS (反復重み付き最小二乗) + Huber損失で
ロバスト化する。仕組みは単純:
  1. 平面性から決まる基礎重み w0 でいつも通り重み付き最小二乗を解く
  2. 解いた結果に対する各ペアの残差 (角度 or mm) を計算する
  3. 残差が閾値(huber_delta)を超えるペアは重み huber_delta/残差 で自動的に
     減衰させる (Huber重み。閾値以内は1倍のまま)
  4. 2-3を数回繰り返す (IRLS)
これにより、外れ値ペアは「二値で切る/切らない」ではなく「残差に応じて
連続的に発言力を失う」ようになり、prev1/prev2のような多段の特別扱いが
不要になる。

ジャイロとの比較 (pair_err/rot_err) は診断・可視化用にのみ残しており、
採否には一切使わない (colored scatterや点線の意味が「ジャイロ棄却」から
「IRLSで大きく減衰させられた」に変わる)。

plot()・plot_true_error() は tof_odometry.py のものをそのまま再利用する
(res の辞書スキーマを合わせてあるため)。

使い方:
    python3 tof_odometry_robust.py log50.csv
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tofodo
from tof_odometry import rot_ypr, rot_to_ypr, solve_yaw_only

OUT_DIR = tofodo.OUT_DIR
CSV_DIR = tofodo.CSV_DIR

# ---- IRLS(Huber)のパラメータ ----
HUBER_ITERS = 5
HUBER_DELTA_DEG = 2.0     # Kabsch(回転)の残差角度の閾値[deg]
HUBER_DELTA_MM = tofodo.RANSAC_THRESH_MM  # Δt(並進)の残差の閾値[mm] (平面フィットと同じ尺度を流用)
GATED_WEIGHT_RATIO = 0.5  # IRLS後の重みが基礎重みのこの比率未満なら「大きく減衰」とみなし可視化上gated扱い

# 元のtof_odometry.kabsch_normals()のrank判定は S>1e-6*S[0] という非常に緩い
# 条件数閾値で、②のジャイロ比較ゲートが「rank>=2と判定されたが実際には
# ほぼ縮退(法線がほぼ平行)で不安定なフィット」を後段で弾く安全網になっていた。
# ここでは②のゲートを撤廃した代わりに、rank判定そのものを厳しくして
# (第2特異値が第1特異値の一定比率以上ないと2方向ぶんの回転を信用しない)、
# 不安定なフィットが直接 R_cum に混入するのを防ぐ。
ROBUST_RANK_COND_RATIO = 0.15

# IRLS(Huber)は「概ね正しい対応点の中に混じったノイズの多い点」を減衰させるのは
# 得意だが、「そもそも別の壁を誤って対応付けてしまった」というデータ対応付け
# 自体の誤りは、残差だけからは(特にpairsが1〜2個で冗長性が無いときは)検出
# できない。そのため、①のgate_deg(≈1.5度)のような厳しい二値棄却ではなく、
# 明らかにおかしい対応(=別の壁を掴んでいる)だけを弾く緩いサニティチェックを
# 残す (通常のノイズはこれより十分小さく、閾値内に収まる想定)。
SANITY_GATE_DEG = 15.0


def kabsch_normals_robust(N_prev, N_curr, w0, huber_delta_deg=HUBER_DELTA_DEG, n_iter=HUBER_ITERS,
                           rank_cond_ratio=ROBUST_RANK_COND_RATIO):
    """ΔR = argmin_R Σ w_i ||n'_i - R n_i||^2 をIRLS(Huber損失)でロバストに解く。

    w0: 平面性などから決まる基礎重み (合計1に正規化済みでなくてよい)。
    残差角度(R適用後のn_prevとn_currのなす角)が huber_delta_deg を超える
    ペアは、重みを huber_delta_deg/残差角度 の比率で自動的に減衰させる。

    rank判定は S[1] >= rank_cond_ratio*S[0] を要求する (条件数ベース)。
    法線がほぼ平行で実質1方向しか無いのに数値誤差でrank=2と誤判定される
    ケースを避けるため、既定のtof_odometry.kabsch_normals()より厳しくしてある。

    Returns: (R, rank, w_final) — w_final は最終イテレーションの重み (可視化用)。
    """
    w = np.asarray(w0, dtype=float).copy()
    R = np.eye(3)
    rank = 0
    for _ in range(n_iter):
        H = (N_prev * w[:, None]).T @ N_curr
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ U.T))])
        R = V @ D @ U.T
        rank = int((S > rank_cond_ratio * max(S[0], 1e-12)).sum())

        pred = (R @ N_prev.T).T
        cos_ang = np.clip(np.sum(pred * N_curr, axis=1), -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cos_ang))
        huber_w = np.where(ang_deg <= huber_delta_deg, 1.0,
                            huber_delta_deg / np.maximum(ang_deg, 1e-6))
        w = np.asarray(w0, dtype=float) * huber_w
    return R, rank, w


def solve_translation_robust(Ns, dd, w0, huber_delta_mm=HUBER_DELTA_MM, n_iter=HUBER_ITERS, rcond=0.05):
    """Σ w_i (Δd_i + n_i^T Δt)^2 をIRLS(Huber損失)でロバストに最小化する。

    tof_odometry.solve_translation() と同じくtruncated SVDで退化方向を0とする。
    Returns: (Δt, rank, 観測可能方向の基底, w_final)
    """
    w = np.asarray(w0, dtype=float).copy()
    dt = np.zeros(3)
    rank = 0
    basis = np.zeros((0, 3))
    for _ in range(n_iter):
        sw = np.sqrt(w)
        A = Ns * sw[:, None]
        b = -dd * sw
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        keep = S > rcond * max(S[0], 1e-12)
        rank = int(keep.sum())
        if rank == 0:
            dt = np.zeros(3)
            basis = np.zeros((0, 3))
            break
        dt = Vt[keep].T @ ((U[:, keep].T @ b) / S[keep])
        basis = Vt[keep]

        resid_mm = np.abs(dd + Ns @ dt)
        huber_w = np.where(resid_mm <= huber_delta_mm, 1.0,
                            huber_delta_mm / np.maximum(resid_mm, 1e-6))
        w = np.asarray(w0, dtype=float) * huber_w
    return dt, rank, basis, w


def run_odometry_robust(planes, sensors, timestamps, att=None):
    """run_odometry() のロバスト版。ジャイロ比較による二値棄却(①②)を撤廃し、
    kabsch_normals_robust()/solve_translation_robust() のIRLS内部で外れ値を
    自動的に減衰させる。ジャイロは rank_R<2 時の solve_yaw_only (pitch/roll
    を既知として使う) と、診断用の pair_err/rot_err 計算にのみ使う。

    Returns: res (tof_odometry.run_odometry() と同じキー構成の辞書。
             plot()/plot_true_error() をそのまま再利用できる)
    """
    F = len(planes)
    out = dict(
        t=np.asarray(timestamps, dtype=float) / 1000.0,
        dR_ypr=np.full((F, 3), np.nan),
        kabsch_dR_ypr=np.full((F, 3), np.nan),
        kabsch_ref_gap=np.full(F, np.nan),
        dt=np.full((F, 3), np.nan),
        rank_R=np.zeros(F, dtype=int),
        rank_t=np.zeros(F, dtype=int),
        n_pairs=np.zeros(F, dtype=int),
        n_gated=np.zeros(F, dtype=int),
        rot_err=np.full(F, np.nan),
        rot_src=np.zeros(F, dtype=int),   # 0:なし 1:ToF(ロバストKabsch) 3:ジャイロ(対応なし) 4:ToF(yawのみ)
        ypr=np.zeros((F, 3)),
        pos=np.zeros((F, 3)),
        plane_d={s: np.full(F, np.nan) for s in sensors},
        plane_n={s: np.full((F, 3), np.nan) for s in sensors},
        planarity={s: np.full(F, np.nan) for s in sensors},
        # IRLSで重みが GATED_WEIGHT_RATIO 未満に大きく減衰させられたペアかどうか (可視化用)
        gated={s: np.zeros(F, dtype=bool) for s in sensors},
        # ジャイロ予測とのズレ角[deg] (診断・可視化専用。採否には使わない)
        pair_err={s: np.full(F, np.nan) for s in sensors},
        pair_gate_err=np.zeros(0),
    )

    for f in range(F):
        for s in sensors:
            pl = planes[f][s]
            if pl is not None:
                out["plane_d"][s][f] = pl["d"]
                out["plane_n"][s][f] = pl["n"]
                out["planarity"][s][f] = pl["planarity"]

    R_att = None
    if att is not None:
        R_att = [rot_ypr(*att[f]) for f in range(F)]

    # 前回の自分の計測だけを保持する (prev1/prev2の2段フォールバックは不要になった)
    prev1 = {s: None for s in sensors}
    R_cum = np.eye(3)
    p_cum = np.zeros(3)
    R_last = R_att[0] if R_att is not None else None
    R_kabsch_prev1 = R_att[0] if R_att is not None else None
    R_cum_at_kabsch_prev1 = np.eye(3)

    for f in range(F):
        pairs = []
        for s in sensors:
            cur = planes[f][s]
            if cur is None:
                continue
            pv1 = prev1[s]
            if pv1 is not None and pv1[0] is not None:
                # ジャイロ予測とのズレ角。SANITY_GATE_DEG(緩い閾値)より大きい
                # 場合だけ「別の壁を誤って対応付けた」とみなして弾く。それ以下の
                # 細かいノイズはKabsch/solve_translationのIRLSに任せる。
                sane = True
                if R_att is not None and pv1[1] is not None:
                    n_pred1 = pv1[1].T @ R_att[f] @ pv1[0]["n"]
                    c1 = float(np.clip(n_pred1 @ cur["n"], -1.0, 1.0))
                    err1 = np.degrees(np.arccos(c1))
                    out["pair_err"][s][f] = err1
                    sane = err1 < SANITY_GATE_DEG
                if sane:
                    pairs.append((s, pv1[0], cur, pv1[1]))
                else:
                    out["gated"][s][f] = True
            prev1[s] = (cur, R_att[f] if R_att is not None else None)

        out["n_pairs"][f] = len(pairs)

        dR_gyro = None
        if R_att is not None and R_last is not None:
            dR_gyro = R_last.T @ R_att[f]

        dRb, src, since_ref = None, 0, False
        if pairs:
            N_prev = np.array([p[1]["n"] for p in pairs])
            N_curr = np.array([p[2]["n"] for p in pairs])
            w0 = np.array([1.0 / max(p[2]["planarity"], 1e-6) for p in pairs])
            w0 /= w0.sum()

            dR, rank_R, w_final = kabsch_normals_robust(N_prev, N_curr, w0)
            out["rank_R"][f] = rank_R
            for (s, *_), wf, wb in zip(pairs, w_final, w0):
                if wf < GATED_WEIGHT_RATIO * wb:
                    out["gated"][s][f] = True

            yaw_only = False
            if rank_R < 2:
                R_curr_hybrid = solve_yaw_only(pairs, att[f]) if att is not None else None
                if R_curr_hybrid is not None:
                    ref = R_kabsch_prev1 if R_kabsch_prev1 is not None else R_att[f]
                    dRb = ref.T @ R_curr_hybrid
                    yaw_only = True
                else:
                    dRb = np.eye(3)
            else:
                dRb = dR.T
            src = 4 if yaw_only else 1
            out["kabsch_dR_ypr"][f] = rot_to_ypr(dRb)

            # ①のサニティチェックと同じ理由(データ対応付け自体の誤りはIRLSの
            # 残差だけでは検出できない)で、②についても厳密な二値棄却(≈1.5度)
            # の代わりに、明らかにおかしい(SANITY_GATE_DEGを超える)ときだけ
            # ジャイロで代替する緩いバックストップを残す。
            if R_att is not None and R_kabsch_prev1 is not None:
                dR_gyro_kabsch1 = R_kabsch_prev1.T @ R_att[f]
                c1 = float(np.clip((np.trace(dRb.T @ dR_gyro_kabsch1) - 1.0) / 2.0, -1.0, 1.0))
                out["rot_err"][f] = np.degrees(np.arccos(c1))
                if out["rot_err"][f] >= SANITY_GATE_DEG and dR_gyro is not None:
                    # kabsch_dR_ypr は棄却前の生の値のまま残す (可視化用)。
                    # 代用そのものは他フレームと同じ1フレーム分の dR_gyro を使う
                    # (R_kabsch_prev1起点の累積値dRbをそのままdR_gyroに差し替える
                    # とR_cumの二重積算になるため、since_refをFalseにして通常の
                    # 1フレーム更新経路に流す)。
                    dRb, src = dR_gyro, 2

            since_ref = src in (1, 4)

            dd = np.array([p[2]["d"] - p[1]["d"] for p in pairs])
            dt, rank_t, _, _ = solve_translation_robust(N_prev, dd, w0)
            out["dt"][f] = dt
            out["rank_t"][f] = rank_t
            p_cum = p_cum + R_cum @ dt

            if R_att is not None:
                R_kabsch_prev1 = R_att[f]
        elif dR_gyro is not None:
            dRb, src = dR_gyro, 3

        if dRb is not None:
            out["dR_ypr"][f] = rot_to_ypr(dRb)
            if since_ref:
                R_cum = R_cum_at_kabsch_prev1 @ dRb
            else:
                R_cum = R_cum @ dRb
        out["rot_src"][f] = src

        if pairs:
            R_cum_at_kabsch_prev1 = R_cum

        if R_att is not None:
            R_last = R_att[f]
        out["ypr"][f] = rot_to_ypr(R_cum)
        out["pos"][f] = p_cum

    return out


def main():
    ap = argparse.ArgumentParser(description="超低密度ToF点群からの移動量・回転量抽出 (Kabschロバスト化版)")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log50.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+", default=[0, 1, 2, 5, 6, 7])
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=None)
    args = ap.parse_args()

    import pandas as pd
    path = args.csv
    if not os.path.isfile(path):
        cand = os.path.join(CSV_DIR, os.path.basename(path))
        if not os.path.isfile(cand):
            sys.exit(f"ファイルが見つかりません: {args.csv}")
        path = cand
    df = pd.read_csv(path)

    if args.eps is not None:
        tofodo.PLANARITY_EPS = args.eps

    rng = np.random.default_rng(args.seed)
    arrs = tofodo.sensor_arrays(df, args.sensors)
    if args.no_dedup:
        idx = np.arange(len(df))
        fresh = {s: np.ones(len(df), dtype=bool) for s in args.sensors}
    else:
        idx, fresh = tofodo.tof_frames(arrs, args.sensors, df["timestamp"].to_numpy(float) / 1000.0)

    planes, plane_stats, _, _ = tofodo.extract_planes(df, args.sensors, rng, arrs, idx, fresh)
    true_klmn = tofodo.find_rotation_klmn_s(df, df["timestamp"].to_numpy(float) / 1000.0)

    df = df.iloc[idx].reset_index(drop=True)
    t_s = df["timestamp"].to_numpy(float) / 1000.0

    gyro = tofodo.gyro_reference(df, t_s)
    att = gyro["ypr"] if gyro is not None else None

    res = run_odometry_robust(planes, args.sensors, df["timestamp"].to_numpy(), att=att)

    print(f"入力: {path}  ({len(df)} フレーム)")
    for s in args.sensors:
        ok = np.isfinite(res["plane_d"][s]).sum()
        print(f"  sensor{s}: 平面抽出成功 {ok}/{len(df)} フレーム ({100.0*ok/len(df):.1f}%)")
    src = res["rot_src"]
    print(f"  ΔR の出所 ToF(ロバストKabsch)/ジャイロ(対応なし)/なし/ToF(yawのみ) : "
          f"{(src==1).sum()}/{(src==3).sum()}/{(src==0).sum()}/{(src==4).sum()}")
    print(f"  累積回転 (yaw,pitch,roll) [deg] : {res['ypr'][-1].round(2)}")
    print(f"  累積移動 (x,y,z) [mm]           : {res['pos'][-1].round(1)}")
    if gyro is not None:
        rmse = np.sqrt(np.nanmean((res["ypr"] - gyro["ypr"]) ** 2, axis=0))
        print(f"  ToF vs ジャイロ RMSE [deg]      : {rmse.round(2)}")

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_tofodo_robust.png")
    # gate_deg はここでは棄却の閾値ではなく、診断用の色分け(pair_err/rot_err)の
    # スケール中心としてのみ使われる。Huberの閾値と揃えておく。
    tofodo.plot(res, args.sensors, df, out, tmax=args.tmax, gate_deg=HUBER_DELTA_DEG, true_klmn=true_klmn)

    if gyro is not None and None not in true_klmn:
        err_out = os.path.join(OUT_DIR, f"{base}_true_error_robust.png")
        tofodo.plot_true_error(res, gyro, true_klmn, err_out)


if __name__ == "__main__":
    main()
