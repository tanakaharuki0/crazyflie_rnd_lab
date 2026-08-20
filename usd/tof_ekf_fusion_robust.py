#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tof_ekf_fusion.py の「ロバスト化したKabsch(tof_odometry_robust.py)を使う」版。

tof_odometry.run_odometry() (二値的なジャイロ比較棄却①②あり) の代わりに
tof_odometry_robust.run_odometry_robust() (IRLS+Huber損失でKabsch/Δt自体を
ロバスト化し、①②はSANITY_GATE_DEGという緩いバックストップのみに置き換えた版)
を使ってEKF融合の入力(kabsch_dR_ypr, rot_src, n_pairs)を作る。

run_ekf_pair()/plot_comparison() は tof_ekf_fusion.py のものをそのまま再利用
する (res の辞書スキーマが同じため)。

使い方:
    python3 tof_ekf_fusion_robust.py log50.csv
    python3 tof_ekf_fusion_robust.py log50.csv --quad-is-flying
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tofodo
import tof_odometry_robust as tofodo_robust
from tof_ekf_fusion import run_ekf_pair, plot_comparison, OUT_DIR


def main():
    ap = argparse.ArgumentParser(
        description="ToF yawのEKFへの疎結合フュージョン (ロバストKabsch版)")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log50.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+", default=[0, 1, 2, 5, 6, 7])
    ap.add_argument("--use-flow-deck", action="store_true",
                     help="flow deckのデータ(motion.deltaX/Y)が使えるなら融合に使う (既定OFF)")
    ap.add_argument("--quad-is-flying", action="store_true",
                     help="EKFのpredictを飛行中モードで動かす (既定は静止/卓上実験モード)")
    ap.add_argument("--include-rank-deficient", action="store_true",
                     help="rank_R<2 (solve_yaw_only) のフレームも観測に使う (既定OFF)")
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

    res = tofodo_robust.run_odometry_robust(planes, args.sensors, df_tof["timestamp"].to_numpy(), att=att)

    n_rank2 = int((res["rot_src"] == 1).sum())
    n_rank_lt2 = int((res["rot_src"] == 4).sum())
    n_backstop = int((res["rot_src"] == 2).sum())
    print(f"入力: {path} (生ログ{len(df_raw)}行, ToFフレーム{len(df_tof)})")
    print(f"  rot_src==1 (rank_R>=2, ロバストKabsch): {n_rank2} フレーム")
    print(f"  rot_src==4 (rank_R<2, solve_yaw_only): {n_rank_lt2} フレーム"
          f" (include_rank_deficient={args.include_rank_deficient})")
    print(f"  rot_src==2 (SANITY_GATE_DEGを超えたためジャイロで代替): {n_backstop} フレーム")
    print(f"  真値の折れ点 k,l,m,n [s]: {[round(v, 2) for v in true_klmn]}")

    ekf_out = run_ekf_pair(df_raw, t_raw_s, res, idx,
                            use_flow_deck=args.use_flow_deck,
                            quad_is_flying=args.quad_is_flying,
                            include_rank_deficient=args.include_rank_deficient)
    print(f"  ToF yaw観測の適用回数: {ekf_out['n_yaw_updates']}")

    lo, hi = tofodo.TRUE_ERR_T_RANGE
    t = ekf_out["t"]
    mask = (t >= lo) & (t <= hi)
    true_yaw = tofodo.true_yaw_profile(t, *true_klmn)
    for name, yaw in (("EKF(gyroのみ)", ekf_out["yaw_gyro_only"]),
                       ("EKF(ToF融合, ロバストKabsch)", ekf_out["yaw_tof_fused"])):
        err = np.abs(tofodo.angle_diff_deg(yaw, true_yaw))[mask]
        print(f"  {name}: mean|err|={err.mean():.2f} deg, ∫|err|dt={np.trapz(err, t[mask]):.1f} deg·s")

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_ekf_fusion_robust.png")
    plot_comparison(ekf_out, true_klmn, out)


if __name__ == "__main__":
    main()
