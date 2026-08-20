#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tof_ekf_fusion.py の「棄却なし」版。

tof_odometry.run_odometry() には、gate_deg>0 のとき以下2種類の
「ジャイロ由来のyaw変化量との差分による棄却」処理が入っている:

  ① 法線ペアの棄却: 各センサーの現フレーム法線が、前フレーム法線をジャイロの
     回転で予測した値と gate_deg 以上ずれていたら、そのペアをKabschの入力
     から除外する (prev1がダメならprev2も試す2段フォールバック付き)。
  ② KabschのΔRの棄却: ①を経て求めたΔR (法線ベクトルの方位角どうしの
     幾何フィット) が、ジャイロ由来のΔRと gate_deg 以上ずれていたら、
     ΔRごと捨ててジャイロの回転で代替する (rot_src=2)。

run_odometry() は gate_deg<=0 を渡すとこの①②を丸ごとスキップする作りに
既になっていた (solve_yaw_only 自体—rank_R<2のときにジャイロのpitch/rollを
既知として使うフォールバック—はジャイロとの比較による「棄却」ではなく
アルゴリズムの前提なので、gate_deg<=0でも従来通り働く)。

このスクリプトは tof_ekf_fusion.py の run_ekf_pair()/plot_comparison() を
そのまま再利用し、run_odometry() に gate_deg=0 を渡すことで①②を無効化した
場合の挙動を比較できるようにしたもの。tof_odometry.py 自体には一切手を
入れていない (疎結合のまま)。

期待される効果とトレードオフ:
  - 棄却が無くなる分、rank_R>=2 (rot_src==1, EKFへのyaw誤差観測として使う
    フレーム) が増える見込み (tof_ekf_fusion.py での検証では296フレーム中
    わずか6回しか無く、疎さが精度向上の制約になっていた)。
  - 一方で、本来ジャイロと大きく食い違うノイズ/誤対応のペアもそのまま
    Kabschに混入するため、個々のΔyaw推定自体のノイズは増える可能性がある。

使い方:
    python3 tof_ekf_fusion_without_rejection.py log50.csv
    python3 tof_ekf_fusion_without_rejection.py log50.csv --quad-is-flying
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tofodo
from tof_ekf_fusion import run_ekf_pair, plot_comparison, OUT_DIR

NO_REJECTION_GATE_DEG = 0.0


def main():
    ap = argparse.ArgumentParser(
        description="ToF yawのEKFへの疎結合フュージョン (ジャイロ比較による棄却を無効化した版)")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log50.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+", default=[0, 1, 2, 5, 6, 7])
    ap.add_argument("--gate-deg", type=float, default=NO_REJECTION_GATE_DEG,
                     help="run_odometry()のgate_deg。既定0=棄却なし。"
                          "tof_ekf_fusion.pyとの比較用に敢えて正の値を指定することも可能")
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
    n_rank_lt2 = int((res["rot_src"] == 4).sum())
    print(f"入力: {path} (生ログ{len(df_raw)}行, ToFフレーム{len(df_tof)})")
    print(f"  gate_deg={args.gate_deg:g} ({'棄却なし' if args.gate_deg <= 0 else '棄却あり'})")
    print(f"  ToF yaw誤差観測として使うフレーム数 (rot_src==1, rank_R>=2): {n_rank2}")
    print(f"  参考: rank_R<2でsolve_yaw_onlyのみ成立したフレーム数 (rot_src==4, 観測には未使用): {n_rank_lt2}")
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
    out = args.output or os.path.join(OUT_DIR, f"{base}_ekf_fusion_no_rejection.png")
    plot_comparison(ekf_out, true_klmn, out)


if __name__ == "__main__":
    main()
