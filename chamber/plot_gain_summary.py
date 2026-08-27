#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chamber実験: k_center条件ごとの中心オフセット・yawドリフトの平均を集計し、
論文用のサマリー棒グラフ (gain_summary) を作成する。

集計対象: chamber/csv/log*_<tag>.csv (+_pose.csv)。HOVER中に姿勢が崩れて
墜落したログ (k=0.3 の log00: HOVER開始1.6秒でroll 76°まで傾き墜落、
k=0.5 の log00: HOVER到達前に失敗) は kousatsu_wall_centering_k.md と同じ
基準で除外する (`EXCLUDE` を参照)。

ex, ey の集計は plot_chamber.py の数値サマリ (mean={np.nanmean(a):+.3f}) と
同じ定義を使う: ログ内でまず符号付き平均 mean(ex), mean(ey) を取り、その
絶対値をログ間で平均する (ログ内の振動が打ち消し合わないよう、ポイントごとの
|ex| の平均ではなく、ログごとの平均のオフセット方向で評価する)。

出力: chamber/figs/gain_summary.png, chamber/figs/gain_summary.pdf

使い方:
    python3 plot_gain_summary.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
FIGS_DIR = os.path.join(SCRIPT_DIR, "figs")

SENSOR_FRONT = "vl53l8cx.s1"
SENSOR_BACK = "vl53l8cx.s7"
SENSOR_LEFT = "vl53l8cx.s5"
SENSOR_RIGHT = "vl53l8cx.s6"
STATE_HOVER = 6

CONDITIONS = ["k_0.3", "k_0.5", "k_0.7", "without_Pcontrol"]
LABELS = ["k=0.3", "k=0.5", "k=0.7", "no P-ctrl"]

# HOVER中に姿勢が崩れて墜落し、他のログと同列に平均できないもの
# (kousatsu_wall_centering_k.md セクション1参照)
EXCLUDE = {
    ("k_0.3", "log00"),  # HOVER開始1.6秒でroll 76度まで傾き墜落
    ("k_0.5", "log00"),  # HOVER到達前に失敗
}


def per_log_stats(tag):
    """条件タグ1つぶんの (|ex| mean, |ey| mean, |yaw drift|) をログごとに返す。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(CSV_DIR, f"log*_{tag}.csv"))):
        name = os.path.basename(f).split("_")[0]  # log00
        if (tag, name) in EXCLUDE:
            continue
        pose_f = f[:-4] + "_pose.csv"
        if not os.path.exists(pose_f):
            continue
        df = pd.read_csv(f)
        pose = pd.read_csv(pose_f)
        hover = pose["chamber.state"] == STATE_HOVER
        if not hover.any():
            continue  # HOVER未到達のログは除外
        t0 = pose.loc[hover, "timestamp"].iloc[0]
        t1 = pose.loc[hover, "timestamp"].iloc[-1]
        dfh = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]
        if dfh.empty:
            continue
        # 壁距離 [m]。0以下 (無効測距) は plot_chamber.py と同じく NaN 扱いにする。
        def dist(col):
            d = dfh[col].to_numpy(dtype=float) / 1000.0
            return np.where(d > 0.0, d, np.nan)

        dF, dB = dist(SENSOR_FRONT), dist(SENSOR_BACK)
        dL, dR = dist(SENSOR_LEFT), dist(SENSOR_RIGHT)
        ex = (dF - dB) / 2.0
        ey = (dL - dR) / 2.0
        yaw_drift = (pose.loc[hover, "stabilizer.yaw"].iloc[-1]
                     - pose.loc[hover, "stabilizer.yaw"].iloc[0])
        # plot_chamber.py の数値サマリと同じ定義: ログ内はまず符号付き平均を取り、
        # その絶対値をログ間で平均する。
        rows.append((abs(np.nanmean(ex)), abs(np.nanmean(ey)), abs(yaw_drift)))
    return rows


def main():
    ex_means, ey_means, yaw_means = [], [], []
    for tag in CONDITIONS:
        rows = per_log_stats(tag)
        n = len(rows)
        ex, ey, yaw = np.mean(rows, axis=0)
        print(f"{tag:20s} n={n:2d}  |ex| mean={ex:.3f}  |ey| mean={ey:.3f}  "
              f"|yaw drift| mean={yaw:.1f}")
        ex_means.append(ex)
        ey_means.append(ey)
        yaw_means.append(yaw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))

    x = np.arange(len(LABELS))
    w = 0.35
    b1 = ax1.bar(x - w / 2, ex_means, w, label=r"$|e_x|$ (front/back)")
    b2 = ax1.bar(x + w / 2, ey_means, w, label=r"$|e_y|$ (left/right)")
    ax1.set_xticks(x, LABELS)
    ax1.set_ylabel("mean offset from\nchamber centre [m]")
    ax1.set_title("(a) centring accuracy")
    ax1.legend(frameon=False)
    for b in (*b1, *b2):
        ax1.annotate(f"{b.get_height():.3f}",
                     (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8)

    b3 = ax2.bar(x, yaw_means, color="tab:green")
    ax2.set_xticks(x, LABELS)
    ax2.set_ylabel("mean |yaw drift|\nduring HOVER [deg]")
    ax2.set_title("(b) attitude stability")
    for b in b3:
        ax2.annotate(f"{b.get_height():.1f}",
                     (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGS_DIR, f"gain_summary.{ext}")
        fig.savefig(out, dpi=300)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
