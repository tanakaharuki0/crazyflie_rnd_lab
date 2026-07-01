#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microSDカードデッキのバイナリログ (log00 等) をCSVに変換する。

VL53L8CX を最大11個 (vl53l8cx.s0 ... vl53l8cx.s10) 想定。
ログに実際に含まれる変数だけを自動で書き出すので、
センサーが11個未満でもそのまま動作する。

使い方:
    python3 log_to_csv.py <logfile> [-o output.csv]

例:
    python3 log_to_csv.py log00
    python3 log_to_csv.py log00 -o run1.csv
"""
import argparse
import os
import sys

import pandas as pd

import cfusdlog

# 想定する最大センサー数
NUM_SENSORS = 11

# このスクリプトと同じ階層の入出力フォルダ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")  # バイナリログ置き場
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")    # CSV出力先


def main():
    parser = argparse.ArgumentParser(description="uSD log -> CSV converter (up to 11 VL53L8CX sensors)")
    parser.add_argument("logfile", help="バイナリログファイル (例: log00)")
    parser.add_argument("-o", "--output", default=None, help="出力CSVパス (省略時は <logfile>.csv)")
    parser.add_argument("--event", default="fixedFrequency", help="抽出するイベント名 (既定: fixedFrequency)")
    args = parser.parse_args()

    # 入力ログの解決: そのまま見つからなければ logs/ フォルダ内を探す
    logfile = args.logfile
    if not os.path.isfile(logfile):
        candidate = os.path.join(LOGS_DIR, os.path.basename(logfile))
        if os.path.isfile(candidate):
            logfile = candidate
        else:
            sys.exit(f"ファイルが見つかりません: {args.logfile} (logs/ も確認しました)")

    # バイナリログをデコード
    data = cfusdlog.decode(logfile)
    if data is None:
        sys.exit("デコードに失敗しました (フォーマット不正の可能性)。")

    if args.event not in data:
        sys.exit(f"イベント '{args.event}' がログに含まれていません。含まれるイベント: {list(data.keys())}")

    log = data[args.event]

    # DataFrame 化 (各変数は同じ長さの配列)
    df = pd.DataFrame(log)

    # 列の並びを整える: timestamp -> vl53l8cx.s0..s10 -> その他
    sensor_cols = [f"vl53l8cx.s{i}" for i in range(NUM_SENSORS) if f"vl53l8cx.s{i}" in df.columns]
    other_cols = [c for c in df.columns if c not in (["timestamp"] + sensor_cols)]
    ordered = (["timestamp"] if "timestamp" in df.columns else []) + sensor_cols + other_cols
    df = df[ordered]

    # 出力CSV: 既定は csv/ フォルダに <ログ名>.csv
    if args.output:
        out = args.output
        if not os.path.dirname(out):
            out = os.path.join(CSV_DIR, out)
    else:
        base = os.path.splitext(os.path.basename(logfile))[0]
        out = os.path.join(CSV_DIR, base + ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print(f"書き出し完了: {out}")
    print(f"  サンプル数 : {len(df)}")
    print(f"  検出センサー: {sensor_cols if sensor_cols else '(なし)'}")
    print(f"  全列        : {list(df.columns)}")


if __name__ == "__main__":
    main()
