#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_to_csv.py が生成したCSVから VL53L8CX の距離をグラフ描画する。

最大11センサー (vl53l8cx.s0 ... vl53l8cx.s10) に対応。
CSVに存在するセンサー列だけを自動で検出してプロットする。

使い方:
    python3 plot_csv.py <csvfile> [--save out.png] [--separate]

グラフは必ず PNG として figs/ フォルダ (このスクリプトと同じ階層) に保存される。

オプション:
    --save PATH   保存ファイル名を指定 (ディレクトリ無指定なら figs/ 下)
    --separate    センサーごとにサブプロットを分けて表示 (既定は1枚に重ね描き)

例:
    python3 plot_csv.py log00.csv              # -> figs/log00.png
    python3 plot_csv.py log00.csv --separate   # -> figs/log00_separate.png
    python3 plot_csv.py log00.csv --save x.png # -> figs/x.png
"""
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

NUM_SENSORS = 11

# グラフから除外するセンサー番号をここで指定する (例: [0, 3, 10])。
# 空リスト [] なら全センサーを描画する。
EXCLUDE_SENSORS = []  # 例: [0, 3, 10]


def main():
    parser = argparse.ArgumentParser(description="Plot VL53L8CX distances from CSV (up to 11 sensors)")
    parser.add_argument("csvfile", help="入力CSV (log_to_csv.py の出力)")
    parser.add_argument("--save", default=None, help="画像保存パス (省略時は画面表示)")
    parser.add_argument("--separate", action="store_true", help="センサーごとにサブプロット分割")
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

    # 時間軸: timestamp[ms] があれば秒に変換、なければサンプル番号
    if "timestamp" in df.columns:
        t = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0
        xlabel = "time [s]"
    else:
        t = range(len(df))
        xlabel = "sample"

    # 存在するセンサー列を検出 (EXCLUDE_SENSORS で指定した番号は除外)
    sensor_cols = [
        f"vl53l8cx.s{i}"
        for i in range(NUM_SENSORS)
        if f"vl53l8cx.s{i}" in df.columns and i not in EXCLUDE_SENSORS
    ]
    if not sensor_cols:
        sys.exit("描画対象のセンサー列がありません (CSV未検出 or 全て除外指定)。")

    if EXCLUDE_SENSORS:
        print(f"除外センサー: {sorted(EXCLUDE_SENSORS)}")

    plt.rcParams["figure.facecolor"] = "w"
    # 全テキスト要素のフォントサイズを16ptに統一
    plt.rcParams.update({
        "font.size": 16,          # 既定 (目盛り数値など)
        "axes.titlesize": 16,     # サブプロットのタイトル
        "axes.labelsize": 16,     # 軸ラベル
        "xtick.labelsize": 16,    # X軸目盛り
        "ytick.labelsize": 16,    # Y軸目盛り
        "legend.fontsize": 16,    # 凡例
        "figure.titlesize": 16,   # figure全体のタイトル
    })

    if args.separate:
        # センサーごとにサブプロット
        n = len(sensor_cols)
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2.5 * rows), squeeze=False)
        for idx, col in enumerate(sensor_cols):
            ax = axes[idx // cols][idx % cols]
            ax.plot(t, df[col], "-")
            ax.set_title(col)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("distance [mm]")
            ax.grid(True)
        # 余ったサブプロットを非表示
        for j in range(len(sensor_cols), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.tight_layout()
    else:
        # 1枚に重ね描き
        plt.figure(figsize=(10, 6))
        for col in sensor_cols:
            plt.plot(t, df[col], "-", label=col)
        plt.xlabel(xlabel)
        plt.ylabel("distance [mm]")
        plt.ylim(-50, 2500)  # 距離は0以上
        plt.title("VL53L8CX averaged distance")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()

    print(f"検出センサー: {sensor_cols}")

    # 必ず figs フォルダ (このスクリプトと同じ階層) にPNG保存する
    figs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    if args.save:
        # --save 指定時はそのファイル名を使う (ディレクトリ指定が無ければ figs 下に置く)
        out = args.save
        if not os.path.dirname(out):
            out = os.path.join(figs_dir, out)
    else:
        # 既定: <csv名>.png を figs フォルダに保存
        base = os.path.splitext(os.path.basename(csvfile))[0]
        suffix = "_separate" if args.separate else ""
        out = os.path.join(figs_dir, f"{base}{suffix}.png")

    plt.savefig(out, dpi=150)
    print(f"画像を保存しました: {out}")

    # ===== EKF推定位置 x, y の時間変化グラフ (横軸:時間, 縦軸:移動量[mm]) =====
    # stateEstimate.x / y は単位[m]なので mm に換算 (×1000)
    base = os.path.splitext(os.path.basename(csvfile))[0]
    pos_axes = [c for c in ("stateEstimate.x", "stateEstimate.y") if c in df.columns]
    if pos_axes:
        plt.figure(figsize=(10, 6))
        for col in pos_axes:
            label = col.split(".")[-1]  # "x" / "y"
            plt.plot(t, df[col] * 1000.0, "-", label=label)
        plt.xlabel(xlabel)
        plt.ylabel("displacement [mm]")
        plt.title("EKF estimated position (x, y)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        xy_out = os.path.join(figs_dir, f"{base}_xy.png")
        plt.savefig(xy_out, dpi=150)
        print(f"xy位置グラフを保存しました: {xy_out}")
    else:
        print("stateEstimate.x / y がCSVに無いため xy位置グラフはスキップしました。")


if __name__ == "__main__":
    main()
