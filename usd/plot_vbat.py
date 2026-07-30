#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_to_csv.py が生成したCSVから pm.vbatMV (バッテリ電圧) の時間変化を描画する。

使い方:
    python3 plot_vbat.py <csvfile> [--save out.png]

グラフは必ず PNG として figs/ フォルダ (このスクリプトと同じ階層) に保存される。

例:
    python3 plot_vbat.py log37.csv          # -> figs/log37_vbat.png
    python3 plot_vbat.py log37.csv --save x.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

VBAT_COL = "pm.vbatMV"


def main():
    parser = argparse.ArgumentParser(description="Plot pm.vbatMV over time from CSV")
    parser.add_argument("csvfile", help="入力CSV (log_to_csv.py の出力)")
    parser.add_argument("--save", default=None, help="画像保存パス (省略時は <csv名>_vbat.png)")
    args = parser.parse_args()

    # 入力CSVの解決: そのまま見つからなければ csv/ フォルダ内を探す
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
    csvfile = args.csvfile
    if not os.path.isfile(csvfile):
        candidate = os.path.join(csv_dir, os.path.basename(csvfile))
        if os.path.isfile(candidate):
            csvfile = candidate
        else:
            sys.exit(f"ファイルが見つかりません: {args.csvfile} (csv/ も確認しました)")

    df = pd.read_csv(csvfile)

    if VBAT_COL not in df.columns:
        sys.exit(f"'{VBAT_COL}' がCSVに含まれていません。含まれる列: {list(df.columns)}")

    # 時間軸: timestamp[ms] があれば秒に変換、なければサンプル番号
    if "timestamp" in df.columns:
        t = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0
        xlabel = "time [s]"
    else:
        t = range(len(df))
        xlabel = "sample"

    # 電圧は mV -> V に換算
    vbat_v = df[VBAT_COL] / 1000.0

    plt.rcParams["figure.facecolor"] = "w"
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 16,
    })

    plt.figure(figsize=(10, 6))
    plt.plot(t, vbat_v, "-", color="tab:red")
    plt.xlabel(xlabel)
    plt.ylabel("battery voltage [V]")
    plt.title("Battery voltage (pm.vbatMV)")
    plt.grid(True)
    plt.tight_layout()

    # 必ず figs フォルダにPNG保存する
    figs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    if args.save:
        out = args.save
        if not os.path.dirname(out):
            out = os.path.join(figs_dir, out)
    else:
        base = os.path.splitext(os.path.basename(csvfile))[0]
        out = os.path.join(figs_dir, f"{base}_vbat.png")

    plt.savefig(out, dpi=150)
    print(f"画像を保存しました: {out}")
    print(f"  サンプル数 : {len(df)}")
    print(f"  電圧範囲   : {vbat_v.min():.3f} - {vbat_v.max():.3f} V")


if __name__ == "__main__":
    main()
