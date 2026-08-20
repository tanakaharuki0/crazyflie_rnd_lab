#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tof_ekf_fusion.py の「rank不足フレームも使う」版。

tof_odometry.run_odometry() は、法線ベクトルの向きが2方向以上そろった
(rank_R>=2, rot_src==1) フレームでは幾何学だけで(ジャイロ非依存の)フルの
Kabsch回転を解く一方、1方向しかそろわない (rank_R<2, rot_src==4) フレームでは
`solve_yaw_only()` にフォールバックする。これは生ジャイロ積分由来の
pitch/rollを既知として使い、yawだけを閉形式で解く方法 (tof_odometry.py の
docstring参照)。

tof_ekf_fusion.py はこれまで rot_src==1 のフレームだけをEKFへのyaw誤差観測に
使っていた (rot_src==4 はpitch/rollをジャイロに依存しているため除外していた)。
本スクリプトは rot_src==4 のフレームも観測に含める (run_ekf_pair() の
include_rank_deficient=True)。ジャイロのpitch/rollを信じるという追加の前提が
乗る分、観測の標準偏差に RANK_DEFICIENT_STD_MULT 倍のペナルティをかけて
信頼度を下げた上で使う。

--quad-is-flying は既定でON (--no-quad-is-flyingで無効化可能)。

使い方:
    python3 tof_ekf_fusion_ignore_rank.py log50.csv
    python3 tof_ekf_fusion_ignore_rank.py log50.csv --no-quad-is-flying
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tofodo
from tof_ekf_fusion import run_ekf_pair, plot_comparison, OUT_DIR


def main():
    ap = argparse.ArgumentParser(
        description="ToF yawのEKFへの疎結合フュージョン (rank_R<2のフレームも観測に使う版)")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log50.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+", default=[0, 1, 2, 5, 6, 7])
    ap.add_argument("--gate-deg", type=float, default=tofodo.GATE_DEG,
                     help="run_odometry()のgate_deg (ジャイロ比較による棄却の閾値)")
    ap.add_argument("--use-flow-deck", action="store_true",
                     help="flow deckのデータ(motion.deltaX/Y)が使えるなら融合に使う (既定OFF)")
    ap.add_argument("--quad-is-flying", action=argparse.BooleanOptionalAction, default=True,
                     help="EKFのpredictを飛行中モードで動かす (既定ON。--no-quad-is-flyingで無効化)")
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
    print(f"  gate_deg={args.gate_deg:g}, quad_is_flying={args.quad_is_flying}")
    print(f"  rot_src==1 (rank_R>=2, フルKabsch): {n_rank2} フレーム")
    print(f"  rot_src==4 (rank_R<2, solve_yaw_only): {n_rank_lt2} フレーム (今回はこれも観測に使用)")
    print(f"  真値の折れ点 k,l,m,n [s]: {[round(v, 2) for v in true_klmn]}")

    ekf_out = run_ekf_pair(df_raw, t_raw_s, res, idx,
                            use_flow_deck=args.use_flow_deck,
                            quad_is_flying=args.quad_is_flying,
                            include_rank_deficient=True)
    print(f"  ToF yaw観測の適用回数 (rank_R>=2 と rank_R<2 の合計): {ekf_out['n_yaw_updates']}")

    lo, hi = tofodo.TRUE_ERR_T_RANGE
    t = ekf_out["t"]
    mask = (t >= lo) & (t <= hi)
    true_yaw = tofodo.true_yaw_profile(t, *true_klmn)
    for name, yaw in (("EKF(gyroのみ)", ekf_out["yaw_gyro_only"]),
                       ("EKF(ToF融合, rank不問)", ekf_out["yaw_tof_fused"])):
        err = np.abs(tofodo.angle_diff_deg(yaw, true_yaw))[mask]
        print(f"  {name}: mean|err|={err.mean():.2f} deg, ∫|err|dt={np.trapz(err, t[mask]):.1f} deg·s")

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_ekf_fusion_ignore_rank.png")
    plot_comparison(ekf_out, true_klmn, out)


if __name__ == "__main__":
    main()
