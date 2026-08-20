#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tof_odometry.py (前回・前々回=prev1/prev2 を見る既存実装、ジャイロ比較による
①②の棄却あり) が計算する回転行列 dRb を、EKFの姿勢(クォータニオン)へ
**直接合成する**版 (オイラー角への分解・再合成を経由しない)。

前バージョン (build_substitute_gyro_z_degs による角速度置換) は、dRbを
オイラー角YPRに分解してyaw成分だけを取り出し、生ジャイロのroll/pitch
レートと合成し直してEKFのクォータニオン積分に入力していた。これは
「本来同時に起きた回転を、後から別々の軸の角速度として合成し直す」ことに
なり数学的に不整合で、tof_odometry.py自身の累積yawとは大きく乖離すること
が確認された (log50で平均37度、最大103度の差)。

本版は tof_odometry.run_odometry() が返す dR_mat (3x3回転行列そのもの) と
since_ref (そのdR_matが「前回ToFが成立した時点からの累積回転」か「単発の
ジャイロ代用」かのフラグ) を使い、tof_odometry.py 自身のR_cum更新
  since_ref: R_cum = R_cum_at_kabsch_prev1 @ dRb
  そうでなければ: R_cum = R_cum @ dRb (生ジャイロの1フレーム分なのでEKF側の
                                        連続的なジャイロ予測に任せて何もしない)
と同じ規則を、EKFのクォータニオンに対して直接 (行列→クォータニオン変換
quat_from_rotmat() はオイラー角を経由しないShepperd法) 適用する。

使い方:
    python3 tof_ekf_fusion_gyro_substitute.py log50.csv
    python3 tof_ekf_fusion_gyro_substitute.py log50.csv --quad-is-flying
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tofodo
from tof_ekf_fusion import plot_comparison, OUT_DIR
from kalman_core import (KalmanCore, KalmanCoreParams, DEG_TO_RAD, GRAVITY_MAGNITUDE,
                          quat_from_rotmat, _quat_mult, _quat_to_R)


def run_ekf_pair_matrix_substitute(df_raw, t_raw_s, res, idx, quad_is_flying=False):
    """生gyroのみのEKFと、dRb行列を姿勢に直接合成するEKFを同じ入力で走らせる。

    ToFフレームで since_ref[f]==True (rot_src in (1,4), 幾何学的に独立な
    ToF由来の情報) のときだけ、「前回ToFが成立した時点のEKF姿勢」に
    dR_mat[f] を直接合成してEKFのクォータニオンを置き換える。
    since_ref==False (単発のジャイロ代用, rot_src in (2,3)) のフレームは
    tof_odometry.py自身もジャイロに頼っているだけなので、EKF側は生ジャイロに
    よる連続的な予測をそのまま使い、何も上書きしない。

    Returns: dict(t=..., yaw_gyro_only=..., yaw_tof_fused=..., n_yaw_updates=int)
             (plot_comparison用のキー名。yaw_tof_fusedが行列合成版の出力)
    """
    n = len(df_raw)
    acc = df_raw[["acc.x", "acc.y", "acc.z"]].to_numpy(dtype=float)
    gyro = df_raw[["gyro.x", "gyro.y", "gyro.z"]].to_numpy(dtype=float)

    tof_frame_info = {}
    for f, row in enumerate(idx):
        tof_frame_info[int(row)] = dict(
            rot_src=int(res["rot_src"][f]),
            since_ref=bool(res["since_ref"][f]),
            dR_mat=res["dR_mat"][f],
        )

    ekf_gyro = KalmanCore(KalmanCoreParams(), now_s=t_raw_s[0])
    ekf_sub = KalmanCore(KalmanCoreParams(), now_s=t_raw_s[0])

    yaw_gyro_only = np.zeros(n)
    yaw_matrix_sub = np.zeros(n)
    q_at_last_fix = ekf_sub.q.copy()
    n_yaw_updates = 0

    for i in range(n):
        t = float(t_raw_s[i])
        a = tuple(acc[i] * GRAVITY_MAGNITUDE)
        g = tuple(gyro[i] * DEG_TO_RAD)

        ekf_gyro.predict(a, g, t, quad_is_flying=quad_is_flying)
        ekf_sub.predict(a, g, t, quad_is_flying=quad_is_flying)
        ekf_gyro.add_process_noise(t)
        ekf_sub.add_process_noise(t)
        ekf_gyro.finalize()
        ekf_sub.finalize()

        if i in tof_frame_info:
            info = tof_frame_info[i]
            if info["since_ref"]:
                # dR_mat をオイラー角に分解せず、クォータニオン積で直接合成する
                dq = quat_from_rotmat(info["dR_mat"])
                new_q = _quat_mult(q_at_last_fix, dq)
                ekf_sub.q = new_q / (np.linalg.norm(new_q) + 1e-9)
                ekf_sub.R = _quat_to_R(ekf_sub.q)
                n_yaw_updates += 1
            if info["rot_src"] in (1, 2, 4):
                # tof_odometry.pyの「R_cum_at_kabsch_prev1 = R_cum (pairsがあれば
                # 常に更新)」と同じく、次回の合成起点として現在の姿勢を保存
                q_at_last_fix = ekf_sub.q.copy()

        yaw_gyro_only[i] = ekf_gyro.externalize_ypr_deg()[0]
        yaw_matrix_sub[i] = ekf_sub.externalize_ypr_deg()[0]

    return dict(t=t_raw_s - t_raw_s[0], yaw_gyro_only=yaw_gyro_only,
                yaw_tof_fused=yaw_matrix_sub, n_yaw_updates=n_yaw_updates)


def main():
    ap = argparse.ArgumentParser(
        description="tof_odometry.pyのdRb行列をEKFの姿勢に直接合成する版 (オイラー角非経由)")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log50.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+", default=[0, 1, 2, 5, 6, 7])
    ap.add_argument("--gate-deg", type=float, default=tofodo.GATE_DEG,
                     help="tof_odometry.run_odometry()のgate_deg (①②の棄却閾値)")
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

    # 既存のtof_odometry.py実装 (prev1/prev2, ①②のジャイロ整合チェックあり) をそのまま使う
    res = tofodo.run_odometry(planes, args.sensors, df_tof["timestamp"].to_numpy(),
                               att=att, gate_deg=args.gate_deg)

    n_since_ref = int(res["since_ref"].sum())
    print(f"入力: {path} (生ログ{len(df_raw)}行, ToFフレーム{len(df_tof)})")
    print(f"  dR_matを直接合成するフレーム数 (since_ref==True, rot_src==1 or 4): "
          f"{n_since_ref}/{len(df_tof)}")
    print(f"  真値の折れ点 k,l,m,n [s]: {[round(v, 2) for v in true_klmn]}")

    ekf_out = run_ekf_pair_matrix_substitute(df_raw, t_raw_s, res, idx,
                                              quad_is_flying=args.quad_is_flying)
    print(f"  実際に合成を適用した回数: {ekf_out['n_yaw_updates']}")

    lo, hi = tofodo.TRUE_ERR_T_RANGE
    t = ekf_out["t"]
    mask = (t >= lo) & (t <= hi)
    true_yaw = tofodo.true_yaw_profile(t, *true_klmn)
    for name, yaw in (("EKF(生gyroのみ)", ekf_out["yaw_gyro_only"]),
                       ("EKF(dRb行列を直接合成)", ekf_out["yaw_tof_fused"])):
        err = np.abs(tofodo.angle_diff_deg(yaw, true_yaw))[mask]
        print(f"  {name}: mean|err|={err.mean():.2f} deg, ∫|err|dt={np.trapz(err, t[mask]):.1f} deg·s")

    # 参考: tof_odometry.py自身の累積yawとの一致度も確認する
    tofodo_yaw_at_tof = res["ypr"][:, 0]
    ekf_yaw_at_tof = ekf_out["yaw_tof_fused"][idx]
    d = tofodo.angle_diff_deg(ekf_yaw_at_tof, tofodo_yaw_at_tof)
    print(f"  参考: tof_odometry.py自身の累積yawとの差 mean|d|={np.mean(np.abs(d)):.3f} deg, "
          f"max|d|={np.max(np.abs(d)):.3f} deg")

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_ekf_gyro_substitute.png")
    plot_comparison(ekf_out, true_klmn, out, fused_label="EKF (dRb行列を直接合成)")


if __name__ == "__main__":
    main()
