#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uSDログCSVから motion.loopUs と sys.rangingUs の時間変化をグラフ化する。

- motion.loopUs : flowdeckTask の while ループ本体の所要時間 [us]
- sys.rangingUs : Gget_Ranging() の所要時間 [us]

依存: numpy, pandas, matplotlib。

使い方:
    python3 timing_plot.py log32.csv
    python3 timing_plot.py log32.csv -o timing.png
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figs")

# 描画対象: 列名 -> (凡例ラベル, 色)
SERIES = {
    "motion.loopUs": ("flowdeck loop (motion.loopUs)", "tab:blue"),
    "sys.rangingUs": ("Gget_Ranging (sys.rangingUs)", "tab:orange"),
}


def resolve_csv(path):
    if not os.path.isfile(path):
        cand = os.path.join(CSV_DIR, os.path.basename(path))
        if os.path.isfile(cand):
            return cand
        sys.exit(f"ファイルが見つかりません: {path} (csv/ も確認しました)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="motion.loopUs / sys.rangingUs の時間変化グラフ")
    parser.add_argument("csv", help="入力CSV (例: log32.csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="出力画像パス (省略時は figs/<csv名>_timing.png)")
    args = parser.parse_args()

    path = resolve_csv(args.csv)
    df = pd.read_csv(path)

    present = [c for c in SERIES if c in df.columns]
    if not present:
        sys.exit("motion.loopUs / sys.rangingUs のどちらもCSVに含まれていません。")
    for c in SERIES:
        if c not in df.columns:
            print(f"注意: {c} がCSVに無いためスキップします。")

    # 時間軸: timestamp があれば相対秒、無ければサンプル番号
    if "timestamp" in df.columns:
        t = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0  # ms -> s
        xlabel = "time [s]"
    else:
        t = range(len(df))
        xlabel = "sample"

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in present:
        label, color = SERIES[c]
        ax.plot(t, df[c], color=color, lw=1.0, label=label)
        # 平均線
        mean = df[c].mean()
        ax.axhline(mean, color=color, ls="--", lw=0.8, alpha=0.6,
                   label=f"{label} mean={mean:.0f}us")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("duration [us]")
    ax.set_title("Task timing over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if args.output:
        out = args.output
        if not os.path.dirname(out):
            os.makedirs(FIG_DIR, exist_ok=True)
            out = os.path.join(FIG_DIR, out)
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
    else:
        base = os.path.splitext(os.path.basename(path))[0]
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, f"{base}_timing.png")

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"保存しました: {out}")
    for c in present:
        s = df[c]
        print(f"  {c}: mean={s.mean():.0f} us, max={s.max():.0f} us, min={s.min():.0f} us")


if __name__ == "__main__":
    main()
