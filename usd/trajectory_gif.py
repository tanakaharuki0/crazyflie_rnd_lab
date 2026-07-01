#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow deck の生データ (motion.deltaX/deltaY) と高さ (range.zrange) から
x, y, z の軌跡を再構成し、時間とともに軌跡が伸びていく様子を3D GIFに保存する。

== 位置の再構成方法 ==
Crazyflie の Kalman フローモデル (firmware: mm_flow.c) と同じ定数を使う:
    measured_pixels = deltaX * FLOW_RESOLUTION              (FLOW_RESOLUTION = 0.10)
    measured_pixels = (dt * Npix / thetapix) * (v / z)      (Npix=35, thetapix=0.71674)
これを解くと1サンプルあたりの水平移動量は dt に依存せず:
    dx = deltaX * FLOW_RESOLUTION * thetapix / Npix * z
    dy = deltaY * FLOW_RESOLUTION * thetapix / Npix * z
これを積算して x, y を得る。z は range.zrange をそのまま高さとして使う。

注意: 機体ヨー回転やジャイロ補償は無視した簡易再構成。絶対基準が無いため
ドリフトは残る (stateEstimate.x/y が発散するのと同根)。

依存: numpy, matplotlib (Pillow)。

使い方:
    python3 trajectory_gif.py log23.csv
    python3 trajectory_gif.py log23.csv --step 4 --fps 20 -o traj.gif
"""
import argparse
import csv as csvmod
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面なしでGIF生成
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D投影の登録に必要)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
GIF_DIR = os.path.join(SCRIPT_DIR, "gif")

# firmware (mm_flow.c) の定数
FLOW_RESOLUTION = 0.10
NPIX = 35.0
THETAPIX = 0.71674
FLOW_K = FLOW_RESOLUTION * THETAPIX / NPIX  # deltaX * FLOW_K * z[m] = 移動量[m]


def resolve_csv_path(path):
    if not os.path.isfile(path):
        cand = os.path.join(CSV_DIR, os.path.basename(path))
        if os.path.isfile(cand):
            return cand
        sys.exit(f"ファイルが見つかりません: {path} (csv/ も確認しました)")
    return path


def read_csv_columns(path):
    with open(path, newline="") as f:
        reader = csvmod.DictReader(f)
        cols = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                try:
                    cols[name].append(float(row[name]))
                except (TypeError, ValueError):
                    cols[name].append(np.nan)
    return cols


def main():
    parser = argparse.ArgumentParser(
        description="Flow生データと高さから x,y,z 軌跡の3D GIFを生成")
    parser.add_argument("csv", help="入力CSV (例: log23.csv)")
    parser.add_argument("--fps", type=float, default=20.0, help="GIFのフレームレート (既定: 20)")
    parser.add_argument("--step", type=int, default=4,
                        help="何サンプルおきに1フレーム描くか (既定: 4)")
    parser.add_argument("--zunit", choices=["auto", "mm", "m"], default="auto",
                        help="range.zrange の単位 (既定: auto。最大値>10ならmm判定)")
    parser.add_argument("--tail", type=int, default=0,
                        help="軌跡の表示長(直近Nサンプルのみ描画)。0で全履歴 (既定: 0)")
    parser.add_argument("-o", "--output", default=None,
                        help="出力GIFパス (省略時は gif/<csv名>_traj.gif)")
    args = parser.parse_args()

    path = resolve_csv_path(args.csv)
    cols = read_csv_columns(path)

    need = ["range.zrange", "motion.deltaX", "motion.deltaY"]
    missing = [c for c in need if c not in cols]
    if missing:
        sys.exit(f"必要な列がCSVにありません: {missing}")

    z_raw = np.array(cols["range.zrange"], dtype=float)
    dX = np.array(cols["motion.deltaX"], dtype=float)
    dY = np.array(cols["motion.deltaY"], dtype=float)

    # z の単位をメートルに揃える
    zunit = args.zunit
    if zunit == "auto":
        zunit = "mm" if np.nanmax(z_raw) > 10 else "m"
    z_m = z_raw / 1000.0 if zunit == "mm" else z_raw.copy()
    z_m = np.nan_to_num(z_m, nan=0.0)

    # フロー生データが全ゼロなら警告 (Flow deck未接続など)
    if np.allclose(np.nan_to_num(dX), 0) and np.allclose(np.nan_to_num(dY), 0):
        print("警告: motion.deltaX/deltaY が全てゼロです。"
              "Flow deck が生データを出力していない可能性があります"
              "(x,y は原点から動きません)。")

    # 1サンプルあたりの移動量 → 累積で位置
    # flow sensor と crazyflie で座標系が異なるため x,y を入れ替える
    dx = np.nan_to_num(dX) * FLOW_K * z_m
    dy = np.nan_to_num(dY) * FLOW_K * z_m
    x = -np.cumsum(dy)  # x軸方向の正負が crazyflie と逆なので反転
    y = -np.cumsum(dx)  # y軸方向の正負が crazyflie と逆なので反転
    z = z_m

    # Ground truth: Kalman filter の推定位置 (stateEstimate.x/y/z)。無ければ None。
    gt_keys = ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]
    if all(k in cols for k in gt_keys):
        gx = np.array(cols["stateEstimate.x"], dtype=float)
        gy = np.array(cols["stateEstimate.y"], dtype=float)
        gz = np.array(cols["stateEstimate.z"], dtype=float)
    else:
        gx = gy = gz = None
        print("注意: stateEstimate.x/y/z が無いため ground truth は表示しません。")

    n = len(z)
    frames = list(range(1, n + 1, max(1, args.step)))
    if frames[-1] != n:
        frames.append(n)

    # 軸範囲は固定: x,y は -0.5~0.5m, z は 0~1m
    xlim = (-0.5, 0.5)
    ylim = (-0.5, 0.5)
    zlo, zhi = 0.0, 1.0

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(k):
        ax.clear()
        i0 = 0 if args.tail <= 0 else max(0, k - args.tail)
        # Ground truth: Kalman推定位置を青色の軌跡で表示
        if gx is not None:
            ax.plot(gx[i0:k], gy[i0:k], gz[i0:k], color="blue", lw=1.5,
                    label="Kalman (ground truth)")
        # フロー再構成軌跡
        ax.plot(x[i0:k], y[i0:k], z[i0:k], color="tab:orange", lw=1.5,
                label="flow reconstructed")
        ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=40, label="start")
        ax.scatter([x[k - 1]], [y[k - 1]], [z[k - 1]], color="red", s=40,
                   label="current")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(zlo, zhi)
        # x,y,z を同縮尺に (各軸の実寸幅 1.0m に合わせる)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zhi - zlo))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"trajectory  sample {k}/{n}\n"
                     f"x={x[k-1]:.2f} y={y[k-1]:.2f} z={z[k-1]:.2f} [m]")
        ax.legend(loc="upper left")

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000.0 / args.fps)

    if args.output:
        out = args.output
        if not os.path.dirname(out):
            os.makedirs(GIF_DIR, exist_ok=True)
            out = os.path.join(GIF_DIR, out)
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
    else:
        base = os.path.splitext(os.path.basename(path))[0]
        os.makedirs(GIF_DIR, exist_ok=True)
        out = os.path.join(GIF_DIR, f"{base}_traj.gif")
    if not out.lower().endswith(".gif"):
        out += ".gif"

    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"GIFを保存しました: {out}")
    print(f"  サンプル数 {n} / フレーム数 {len(frames)} / {args.fps} fps / z単位 {zunit}")
    print(f"  終端位置 x={x[-1]:.2f} y={y[-1]:.2f} z={z[-1]:.2f} [m]")


if __name__ == "__main__":
    main()
