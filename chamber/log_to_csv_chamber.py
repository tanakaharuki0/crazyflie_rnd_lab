#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chamber 実験用の一括ログ変換。

usd/log_to_csv.py と中身はほぼ同じだが、こちらは
「条件ごとにフォルダ分けされたログ群」をまとめて chamber/csv/ に出す。

  chamber/logs_k_0.7/log00        -> chamber/csv/log00_k_0.7.csv
                                     chamber/csv/log00_k_0.7_pose.csv
  chamber/logs_k_0.5/log03        -> chamber/csv/log03_k_0.5.csv  ...
  chamber/logs_without_Pcontrol/log00 -> chamber/csv/log00_without_Pcontrol.csv ...

フォルダ名から先頭の "logs_" を取り除いたものが条件タグになる。
ログに含まれるイベントは全部書き出す。fixedFrequency は接尾辞なし、
それ以外のイベント (pose など) は _<イベント名> が付く
(既存の plot_chamber.py が期待する命名と同じ)。

使い方:
    python3 log_to_csv_chamber.py                 # logs_* を全部変換
    python3 log_to_csv_chamber.py logs_k_0.7      # そのフォルダだけ
    python3 log_to_csv_chamber.py logs_k_0.7/log03  # 1ファイルだけ
    python3 log_to_csv_chamber.py --list          # 変換せず対象を一覧
"""
import argparse
import glob
import os
import sys

import pandas as pd

# usd/ の cfusdlog.py を使う
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "usd")
sys.path.insert(0, USD_DIR)
import cfusdlog  # noqa: E402

CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
LOGS_GLOB = os.path.join(SCRIPT_DIR, "logs_*")

NUM_SENSORS = 11
FIXED_FREQ_EVENT = "fixedFrequency"


def tag_of(logdir):
    """logs_k_0.7 -> k_0.7 / logs_without_Pcontrol -> without_Pcontrol"""
    base = os.path.basename(os.path.normpath(logdir))
    return base[len("logs_"):] if base.startswith("logs_") else base


def order_columns(df):
    """timestamp -> vl53l8cx.s0..s10 -> その他 の順に並べ替える。"""
    sensors = [f"vl53l8cx.s{i}" for i in range(NUM_SENSORS) if f"vl53l8cx.s{i}" in df.columns]
    others = [c for c in df.columns if c not in (["timestamp"] + sensors)]
    return df[(["timestamp"] if "timestamp" in df.columns else []) + sensors + others]


def convert(logfile, tag, outdir):
    """1つのバイナリログを、含まれるイベントごとにCSVへ書き出す。"""
    name = os.path.splitext(os.path.basename(logfile))[0]  # log00

    if os.path.getsize(logfile) == 0:
        print(f"  skip {name:8s} ({tag}): 空ファイル")
        return []

    try:
        data = cfusdlog.decode(logfile)
    except Exception as e:
        print(f"  skip {name:8s} ({tag}): デコード失敗 ({e})")
        return []
    if not data:
        print(f"  skip {name:8s} ({tag}): イベントなし")
        return []

    written = []
    for event, log in data.items():
        if not log:
            continue
        df = order_columns(pd.DataFrame(log))
        suffix = "" if event == FIXED_FREQ_EVENT else f"_{event}"
        out = os.path.join(outdir, f"{name}_{tag}{suffix}.csv")
        df.to_csv(out, index=False)
        written.append((event, out, len(df)))

    for event, out, n in written:
        print(f"  {os.path.basename(out):40s}  {event:16s} {n:5d} サンプル")
    return written


def main():
    ap = argparse.ArgumentParser(description="chamber ログ一括変換 (条件フォルダ -> csv/)")
    ap.add_argument("targets", nargs="*",
                    help="変換対象のフォルダかファイル (省略時は chamber/logs_* を全部)")
    ap.add_argument("-o", "--outdir", default=CSV_DIR, help=f"出力先 (既定: {CSV_DIR})")
    ap.add_argument("--list", action="store_true", help="変換せず対象を一覧表示")
    args = ap.parse_args()

    # 対象の解決: 引数なし -> logs_* を全部
    if args.targets:
        targets = []
        for t in args.targets:
            p = t if os.path.exists(t) else os.path.join(SCRIPT_DIR, t)
            if not os.path.exists(p):
                sys.exit(f"見つかりません: {t}")
            targets.append(p)
    else:
        targets = sorted(glob.glob(LOGS_GLOB))
        if not targets:
            sys.exit(f"条件フォルダが見つかりません: {LOGS_GLOB}")

    # (ログファイル, タグ) のリストに展開
    jobs = []
    for t in targets:
        if os.path.isdir(t):
            tag = tag_of(t)
            for f in sorted(glob.glob(os.path.join(t, "log*"))):
                if os.path.isfile(f):
                    jobs.append((f, tag))
        else:
            jobs.append((t, tag_of(os.path.dirname(t))))

    if args.list:
        for f, tag in jobs:
            size = os.path.getsize(f)
            mark = " (空)" if size == 0 else ""
            print(f"{tag:20s} {os.path.basename(f):8s} {size:7d} B{mark}")
        print(f"\n合計 {len(jobs)} ファイル")
        return

    os.makedirs(args.outdir, exist_ok=True)

    total = 0
    current_tag = None
    for f, tag in jobs:
        if tag != current_tag:
            print(f"\n=== {tag} ===")
            current_tag = tag
        total += len(convert(f, tag, args.outdir))

    print(f"\n書き出し完了: {total} ファイル -> {args.outdir}")


if __name__ == "__main__":
    main()
