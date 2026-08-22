#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飛行が破綻したログの原因診断グラフ。

log_to_csv.py が pose イベントから作った CSV を入力に、
 - 各 ToF センサーの平均距離と無効ゾーン数 (壁への接触が分かる)
 - 姿勢 (EKF) と姿勢の指令値 ctrltarget
 - 角速度 (gyro)
 - スラストと高度 (range.zrange / stateEstimate.z)
 - 水平位置 (stateEstimate.x/y) と chamber.state
を1枚に並べる。

使い方:
    python3 crash_diag.py log70_pose.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from tof_odometry import CSV_DIR, OUT_DIR
from tof_odometry_chamber import _use_jp_font, hover_spans, draw_hover_lines

D_MIN, D_MAX = 30.0, 4000.0
SENSORS = [0, 1, 2, 5, 6, 7]
SENSOR_LABEL = {0: "0 (前 左45°)", 1: "1 (前)", 2: "2 (前 右45°)",
                5: "5 (左)", 6: "6 (右)", 7: "7 (後)"}


def sensor_stats(df, s):
    """センサー s の有効ゾーン平均距離と有効ゾーン数を返す。"""
    cols = [f"vl53l8cx_s{s}.d{i}" for i in range(16)]
    if any(c not in df.columns for c in cols):
        return None, None
    v = df[cols].to_numpy(dtype=float)
    v[(v < D_MIN) | (v > D_MAX)] = np.nan
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(v, axis=1)
    return mean, np.isfinite(v).sum(axis=1)


def main():
    ap = argparse.ArgumentParser(description="飛行破綻の原因診断グラフ")
    ap.add_argument("csv", help="pose イベントの CSV (例: log70_pose.csv)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    path = args.csv
    if not os.path.isfile(path):
        path = os.path.join(CSV_DIR, os.path.basename(args.csv))
        if not os.path.isfile(path):
            sys.exit(f"ファイルが見つかりません: {args.csv}")
    df = pd.read_csv(path)
    t = (df["timestamp"].to_numpy(float) - df["timestamp"].to_numpy(float)[0]) / 1000.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_jp_font()

    hover = hover_spans(df, t)
    fig, ax = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

    # (0) ToF 平均距離
    for s in SENSORS:
        mean, _ = sensor_stats(df, s)
        if mean is None:
            continue
        ax[0].plot(t, mean, lw=1.2, marker=".", ms=3, label=SENSOR_LABEL.get(s, str(s)))
    ax[0].axhline(D_MIN, color="k", ls=":", lw=1, label=f"有効下限 {D_MIN:.0f} mm")
    ax[0].set_ylabel("平均距離 [mm]")
    ax[0].set_title("各 ToF センサーの有効ゾーン平均距離 (欠測 = 全ゾーンが無効 = 近すぎる/測距不能)")

    # (1) 有効ゾーン数
    for s in SENSORS:
        _, n = sensor_stats(df, s)
        if n is None:
            continue
        ax[1].plot(t, n, lw=1.2, label=SENSOR_LABEL.get(s, str(s)))
    ax[1].set_ylabel("有効ゾーン数 / 16")
    ax[1].set_title("有効ゾーン数 (0 に落ちる = そのセンサーは壁に密着し測距できていない)")

    # (2) 姿勢と指令値
    for nm, c in (("roll", "tab:blue"), ("pitch", "tab:orange"), ("yaw", "tab:green")):
        ax[2].plot(t, df[f"stabilizer.{nm}"], lw=1.3, color=c, label=f"EKF {nm}")
        col = f"ctrltarget.{nm}"
        if col in df.columns:
            ax[2].plot(t, df[col], lw=1.0, ls="--", color=c, label=f"指令 {nm}")
    ax[2].set_ylabel("attitude [deg]")
    ax[2].set_title("姿勢 (実線) と姿勢指令 ctrltarget (破線)")

    # (3) 角速度
    for nm, c in (("x", "tab:blue"), ("y", "tab:orange"), ("z", "tab:green")):
        ax[3].plot(t, df[f"gyro.{nm}"], lw=1.0, color=c, label=f"gyro.{nm}")
    ax[3].set_ylabel("angular rate [deg/s]")
    ax[3].set_title("角速度 (gyro.y = ピッチレート)")

    # (4) スラストと高度・水平位置
    a = ax[4]
    a.plot(t, df["stabilizer.thrust"], lw=1.3, color="tab:red", label="thrust")
    a.set_ylabel("thrust [-]")
    a.set_title("スラストと高度・水平位置")
    b = a.twinx()
    b.plot(t, df["range.zrange"], lw=1.2, color="tab:purple", label="zrange [mm]")
    b.plot(t, df["stateEstimate.z"] * 1000.0, lw=1.2, color="tab:green",
           label="EKF z [mm]")
    b.plot(t, df["stateEstimate.x"] * 1000.0, lw=1.0, ls="--", color="tab:blue",
           label="EKF x [mm]")
    b.plot(t, df["stateEstimate.y"] * 1000.0, lw=1.0, ls="--", color="tab:orange",
           label="EKF y [mm]")
    b.set_ylabel("distance [mm]")
    a.set_xlabel("time [s]")
    lines = a.get_lines() + b.get_lines()
    a.legend(lines, [l.get_label() for l in lines], fontsize=7, ncol=3, loc="upper left")

    for i, aa in enumerate(ax):
        aa.grid(alpha=0.3)
        draw_hover_lines(aa, hover, label=(i == 0))
        if i != 4:
            aa.legend(fontsize=7, ncol=4)

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_crash_diag.png")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"診断グラフを保存: {out}")


if __name__ == "__main__":
    main()
