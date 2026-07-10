#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VL53L8CX の距離グラフ (現在時刻を示す黒縦線が移動) と、
flow センサーから再構成した x,y,z 軌跡の3D図を横並びにしたGIFを生成する。

左パネル : VL53L8CX 距離グラフ + 現在時刻の細い黒縦線 (コマ送りで横移動)
右パネル : flow 再構成軌跡 (オレンジ) + Kalman ground truth (青)。時間とともに伸びる

コマ送りの速さ (fps / step) は trajectory_gif.py と揃えてある。

== 軌跡の再構成 (trajectory_gif.py と同じ) ==
Crazyflie の Kalman フローモデル (mm_flow.c) の定数を使用:
    dx = deltaX * FLOW_RESOLUTION * thetapix / Npix * z   (dt非依存)
flow と crazyflie の座標系差のため x,y の入替と符号反転を行う。

依存: numpy, pandas, matplotlib (Pillow)。

使い方:
    python3 plot_csv_gif.py log32.csv
    python3 plot_csv_gif.py log32.csv --step 4 --fps 20 -o log32_combo.gif
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

NUM_SENSORS = 11
EXCLUDE_SENSORS = []  # 例: [0, 3, 10]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
GIF_DIR = os.path.join(SCRIPT_DIR, "gif")

# firmware (mm_flow.c) の定数
FLOW_RESOLUTION = 0.10
NPIX = 35.0
THETAPIX = 0.71674
FLOW_K = FLOW_RESOLUTION * THETAPIX / NPIX  # deltaX * FLOW_K * z[m] = 移動量[m]


def resolve_csv(path):
    if not os.path.isfile(path):
        cand = os.path.join(CSV_DIR, os.path.basename(path))
        if os.path.isfile(cand):
            return cand
        sys.exit(f"ファイルが見つかりません: {path} (csv/ も確認しました)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="ToF距離グラフ(現在時刻線)とflow再構成3D軌跡を横並びGIF化")
    parser.add_argument("csvfile", help="入力CSV (log_to_csv.py の出力)")
    parser.add_argument("--fps", type=float, default=None,
                        help="フレームレート (既定: 実データ時間で再生されるよう自動計算)")
    parser.add_argument("--step", type=int, default=4,
                        help="何サンプルおきに1フレーム描くか (既定: 4)")
    parser.add_argument("--zunit", choices=["auto", "mm", "m"], default="auto",
                        help="range.zrange の単位 (既定: auto)")
    parser.add_argument("-o", "--output", default=None,
                        help="出力GIFパス (省略時は gif/<csv名>_combo.gif)")
    args = parser.parse_args()

    csvfile = resolve_csv(args.csvfile)
    df = pd.read_csv(csvfile)

    # 時間軸
    if "timestamp" in df.columns:
        t = ((df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0).to_numpy()
        xlabel = "time [s]"
    else:
        t = np.arange(len(df))
        xlabel = "sample"

    # --- 左パネル用: センサー列 ---
    sensor_cols = [
        f"vl53l8cx.s{i}"
        for i in range(NUM_SENSORS)
        if f"vl53l8cx.s{i}" in df.columns and i not in EXCLUDE_SENSORS
    ]
    if not sensor_cols:
        sys.exit("描画対象のセンサー列がありません (CSV未検出 or 全て除外指定)。")

    # --- 右パネル用: flow再構成軌跡 ---
    traj_ok = all(c in df.columns for c in
                  ["range.zrange", "motion.deltaX", "motion.deltaY"])
    if traj_ok:
        z_raw = df["range.zrange"].to_numpy(dtype=float)
        dX = df["motion.deltaX"].to_numpy(dtype=float)
        dY = df["motion.deltaY"].to_numpy(dtype=float)
        zunit = args.zunit
        if zunit == "auto":
            zunit = "mm" if np.nanmax(z_raw) > 10 else "m"
        z_m = np.nan_to_num(z_raw / 1000.0 if zunit == "mm" else z_raw, nan=0.0)
        dx = np.nan_to_num(dX) * FLOW_K * z_m
        dy = np.nan_to_num(dY) * FLOW_K * z_m
        # flow と crazyflie の座標系差: x,y 入替 + 符号反転
        x = -np.cumsum(dy)
        y = -np.cumsum(dx)
        z = z_m
        gt_keys = ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]
        if all(k in df.columns for k in gt_keys):
            gx = df["stateEstimate.x"].to_numpy(dtype=float)
            gy = df["stateEstimate.y"].to_numpy(dtype=float)
            gz = df["stateEstimate.z"].to_numpy(dtype=float)
        else:
            gx = gy = gz = None
    else:
        print("注意: range.zrange / motion.deltaX / motion.deltaY が無いため"
              "3D軌跡パネルは表示しません。")

    n = len(df)
    frames = list(range(0, n, max(1, args.step)))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    # fps: 未指定なら実データ時間 (timestampの総秒数) で再生されるよう自動計算
    if args.fps is not None:
        fps = args.fps
    elif "timestamp" in df.columns and t[-1] > t[0]:
        total_sec = float(t[-1] - t[0])
        fps = len(frames) / total_sec
        print(f"実時間再生: {total_sec:.1f}秒 / {len(frames)}フレーム -> {fps:.2f} fps")
    else:
        fps = 20.0

    plt.rcParams["figure.facecolor"] = "w"

    # ===== 図の構成 =====
    if traj_ok:
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax3d = None

    # --- 左: ToFグラフ (固定) ---
    for col in sensor_cols:
        ax.plot(t, df[col], "-", lw=1.0, label=col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("distance [mm]")
    ax.set_ylim(-50, 2500)
    ax.set_xlim(t[0], t[-1])
    ax.set_title("VL53L8CX averaged distance")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True)
    vline = ax.axvline(t[0], color="black", lw=0.8)

    # --- 右: 3D軌跡の固定設定 ---
    xlim = (-0.5, 0.5)
    ylim = (-0.5, 0.5)
    zlo, zhi = 0.0, 1.0

    def draw3d(k):
        ax3d.clear()
        kk = max(1, k + 1)  # 1..n サンプルまで描画
        if gx is not None:
            ax3d.plot(gx[:kk], gy[:kk], gz[:kk], color="blue", lw=1.5,
                      label="Kalman (ground truth)")
        ax3d.plot(x[:kk], y[:kk], z[:kk], color="tab:orange", lw=1.5,
                  label="flow reconstructed")
        ax3d.scatter([x[0]], [y[0]], [z[0]], color="green", s=40, label="start")
        ax3d.scatter([x[kk - 1]], [y[kk - 1]], [z[kk - 1]], color="red", s=40,
                     label="current")
        ax3d.set_xlim(*xlim)
        ax3d.set_ylim(*ylim)
        ax3d.set_zlim(zlo, zhi)
        ax3d.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zhi - zlo))
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.set_title(f"flow trajectory  ({x[kk-1]:.2f}, {y[kk-1]:.2f}, "
                       f"{z[kk-1]:.2f}) [m]")
        ax3d.legend(loc="upper left", fontsize=8)

    fig.tight_layout()

    def update(k):
        vline.set_xdata([t[k], t[k]])
        if ax3d is not None:
            draw3d(k)
        return ()

    # 3Dをclearするので blit は使わない
    anim = FuncAnimation(fig, update, frames=frames,
                         interval=1000.0 / fps, blit=False)

    if args.output:
        out = args.output
        if not os.path.dirname(out):
            os.makedirs(GIF_DIR, exist_ok=True)
            out = os.path.join(GIF_DIR, out)
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
    else:
        base = os.path.splitext(os.path.basename(csvfile))[0]
        os.makedirs(GIF_DIR, exist_ok=True)
        out = os.path.join(GIF_DIR, f"{base}_combo.gif")
    if not out.lower().endswith(".gif"):
        out += ".gif"

    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIFを保存しました: {out}")
    print(f"  検出センサー: {sensor_cols}")
    print(f"  3D軌跡パネル: {'あり' if traj_ok else 'なし'}")
    print(f"  サンプル数 {n} / フレーム数 {len(frames)} / {fps:.2f} fps / step {args.step}")


if __name__ == "__main__":
    main()
