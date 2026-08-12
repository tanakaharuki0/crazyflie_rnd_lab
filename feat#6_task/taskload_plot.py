#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microSDに記録した tasklog (taskload.* ) をCSVに変換し、
各タスクの実行時間を時系列でプロットする。

ファームウェア側 (src/modules/src/taskload.c) の仕様:
    taskload.<TASK>  直近のサンプリング区間でそのタスクが消費した CPU 時間 [us]
                     (既定の周期 1000ms なら「1秒あたりの CPU マイクロ秒」)
    taskload.period  実測したサンプリング区間 [us]  ← これで割ると占有率になる
    taskload.total   全タスクの CPU 時間の合計 [us]
    taskload.cpuLoad CPU 使用率 [%] (= 100 * (1 - idle/period))

使い方:
    # バイナリログを変換してプロット (CSVも同時に書き出す)
    python3 taskload_plot.py tasklog01

    # フォルダ内の tasklog* をまとめて処理
    python3 taskload_plot.py --all

    # 変換済みCSVから描き直す / 占有率[%]で描く / 表示範囲を絞る
    python3 taskload_plot.py csv/tasklog01.csv --percent --tmax 30
"""
import argparse
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "log")   # バイナリログ置き場
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "fig")

# cfusdlog.py は usd/ にあるので import できるようにする
USD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "usd")

PREFIX = "taskload."
# タスクの実行時間ではない (集計用の) 列
META_COLS = {"cpuLoad", "total", "period"}
# IDLE は「CPUの余り」であって処理時間ではない。既定では他のタスクが潰れるので外す。
IDLE_COL = "IDLE"

# --- メモリ関連 (memload.*) ---
MEM_PREFIX = "memload."
# タスク別スタック残量 (taskstack.*)
STACK_PREFIX = "taskstack."
# FreeRTOS ヒープ総量 [byte]。src/config/config.h の FREERTOS_HEAP_SIZE と一致させること。
HEAP_TOTAL_DEFAULT = 50000
# stackMinId が指すタスク名 (taskload.c の entries[] の並び順)。
STACK_TASK_NAMES = [
    "STABILIZER", "KALMAN", "SENSORS", "PWRMGNT", "CMDHL", "APP",
    "FLOW", "ZRANGER2", "VL53L8CX", "USDLOG", "USDWRITE",
    "SYSTEM", "CRTP_TX", "CRTP_RX", "SYSLINK", "USBLINK", "LOGT",
    "PARAM", "MEM", "WORKER", "PLATFORM", "CRTP_SRV", "IDLE",
]


def decode_to_csv(logfile, out_csv, event="fixedFrequency"):
    """バイナリログをCSVに変換する。すでにCSVならそのまま返す。"""
    import pandas as pd

    if USD_DIR not in sys.path:
        sys.path.insert(0, USD_DIR)
    try:
        import cfusdlog
    except ImportError:
        sys.exit(f"cfusdlog.py が見つかりません ({USD_DIR} を確認してください)")

    data = cfusdlog.decode(logfile)
    if data is None:
        sys.exit(f"デコードに失敗しました: {logfile}")
    if event not in data:
        sys.exit(f"イベント '{event}' がありません。含まれるイベント: {list(data.keys())}")

    df = pd.DataFrame(data[event])
    cols = (["timestamp"] if "timestamp" in df.columns else []) + \
           sorted(c for c in df.columns if c != "timestamp")
    df = df[cols]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"CSVに変換: {out_csv}  ({len(df)} サンプル, {len(df.columns)} 列)")
    return df


def load(path, event="fixedFrequency"):
    """入力がバイナリログでもCSVでも DataFrame を返す。"""
    import pandas as pd

    if path.lower().endswith(".csv"):
        if not os.path.isfile(path):
            cand = os.path.join(CSV_DIR, os.path.basename(path))
            if not os.path.isfile(cand):
                sys.exit(f"ファイルが見つかりません: {path}")
            path = cand
        return pd.read_csv(path), os.path.splitext(os.path.basename(path))[0]

    if not os.path.isfile(path):
        for d in (LOGS_DIR, SCRIPT_DIR):
            cand = os.path.join(d, os.path.basename(path))
            if os.path.isfile(cand):
                path = cand
                break
        else:
            sys.exit(f"ファイルが見つかりません: {path}")

    base = os.path.splitext(os.path.basename(path))[0]
    return decode_to_csv(path, os.path.join(CSV_DIR, base + ".csv"), event), base


def task_columns(df, keep_zero=False, with_idle=False):
    """taskload.<TASK> の列を、平均実行時間の大きい順に返す。"""
    cols = [c for c in df.columns
            if c.startswith(PREFIX) and c[len(PREFIX):] not in META_COLS]
    if not cols:
        sys.exit(
            "taskload.* の列がありません。\n"
            "config.txt に taskload.STABILIZER 等が入っているか確認してください。"
        )
    if not with_idle:
        cols = [c for c in cols if c[len(PREFIX):] != IDLE_COL]
    means = {c: np.nanmean(df[c].to_numpy(dtype=float)) for c in cols}
    if not keep_zero:
        # 一度も走っていないタスク (常に0) は既定では省く
        cols = [c for c in cols if means[c] > 0.0]
    return sorted(cols, key=lambda c: means[c], reverse=True)


def plot(df, cols, base, out_path, percent=False, tmax=None, top=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "IPAGothic", "TakaoGothic", "VL Gothic",
                 "Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False

    t = df["timestamp"].to_numpy(dtype=float) / 1000.0
    t = t - t[0]

    # 占有率に直すときは実測のサンプリング区間で割る (周期が揺れても正しくなる)
    scale = np.ones(len(df))
    if percent:
        pcol = PREFIX + "period"
        if pcol not in df.columns:
            sys.exit("taskload.period が無いので --percent は使えません")
        period = df[pcol].to_numpy(dtype=float)
        scale = np.where(period > 0, 100.0 / period, np.nan)

    shown = cols if top is None else cols[:top]

    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})

    # 色は tab20 で最大20タスクまで区別する
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(shown):
        v = df[c].to_numpy(dtype=float) * scale
        ax[0].plot(t, v, lw=1, color=cmap(i % 20), label=c[len(PREFIX):])

    ax[0].set_ylabel("CPU占有率 [%]" if percent else "実行時間 [us / 区間]")
    ax[0].set_title(f"{base}: タスクごとの実行時間"
                    + ("(占有率)" if percent else "")
                    + (f"  上位{top}件" if top is not None else ""))
    ax[0].legend(fontsize=7, ncol=4, loc="upper right")
    ax[0].grid(alpha=0.3)

    # 下段: 全体のCPU使用率と、サンプリング周期の揺れ
    cpu = PREFIX + "cpuLoad"
    if cpu in df.columns:
        ax[1].plot(t, df[cpu].to_numpy(dtype=float), lw=1, color="tab:red",
                   label="cpuLoad [%]")
    pcol = PREFIX + "period"
    if pcol in df.columns:
        a2 = ax[1].twinx()
        a2.plot(t, df[pcol].to_numpy(dtype=float) / 1000.0, lw=0.8, ls=":",
                color="tab:gray", label="period [ms]")
        a2.set_ylabel("period [ms]", color="tab:gray")
        a2.tick_params(axis="y", labelcolor="tab:gray")
    ax[1].set_ylabel("cpuLoad [%]", color="tab:red")
    ax[1].tick_params(axis="y", labelcolor="tab:red")
    ax[1].set_xlabel("time [s]")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8, loc="upper left")

    if tmax is not None:
        for a in ax:
            a.set_xlim(0, tmax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"グラフを保存: {out_path}")


def has_mem(df):
    """memload.* が記録されているか。"""
    return (MEM_PREFIX + "heapFree") in df.columns


def _set_jp_font():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "IPAGothic", "TakaoGothic", "VL Gothic",
                 "Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_mem(df, base, out_path, heap_total=HEAP_TOTAL_DEFAULT, tmax=None):
    """メモリ消費率を描く。
    上段: FreeRTOS ヒープ使用率 [%] (瞬時値と、起動以降の最悪値)。
    下段: 全タスク中で最も余裕の無いスタック残量 [byte]。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_jp_font()

    t = df["timestamp"].to_numpy(dtype=float) / 1000.0
    t = t - t[0]

    heap_free = df[MEM_PREFIX + "heapFree"].to_numpy(dtype=float)
    heap_min = df[MEM_PREFIX + "heapMin"].to_numpy(dtype=float)
    # 消費率 = (総量 - 空き) / 総量
    heap_use_pct = 100.0 * (heap_total - heap_free) / heap_total
    heap_peak_pct = 100.0 * (heap_total - heap_min) / heap_total

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})

    ax[0].plot(t, heap_use_pct, lw=1.4, color="tab:blue",
               label="ヒープ使用率 [%] (瞬時)")
    ax[0].plot(t, heap_peak_pct, lw=1.2, ls="--", color="tab:red",
               label="ヒープ最大使用率 [%] (起動以降の最悪値)")
    ax[0].set_ylabel("ヒープ消費率 [%]")
    ax[0].set_title(f"{base}: メモリ消費率  (ヒープ総量 {heap_total} byte)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=9, loc="best")
    # 右軸に実バイトも併記
    a0b = ax[0].twinx()
    a0b.set_ylim(heap_total * (1 - ax[0].get_ylim()[1] / 100.0),
                 heap_total * (1 - ax[0].get_ylim()[0] / 100.0))
    a0b.set_ylabel("空きヒープ [byte]", color="tab:gray")
    a0b.tick_params(axis="y", labelcolor="tab:gray")

    # 下段: スタック残量。taskstack.* があればタスク別に、無ければ最小値だけ描く。
    stack_cols = [c for c in df.columns if c.startswith(STACK_PREFIX)]
    if stack_cols:
        # 記録された最小値 (時間方向の最小) が小さい順 = 危険な順に描く
        cmap = plt.get_cmap("tab20")
        order = sorted(stack_cols,
                       key=lambda c: np.nanmin(df[c].to_numpy(dtype=float)))
        for i, c in enumerate(order):
            v = df[c].to_numpy(dtype=float)
            ax[1].plot(t, v, lw=1, color=cmap(i % 20),
                       label=c[len(STACK_PREFIX):])
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_ylabel("スタック残量 [byte]")
        ax[1].set_title("各タスクのスタック残量 (起動以降の最小値。0 に近づくとオーバーフロー危険)")
        ax[1].grid(alpha=0.3)
        ax[1].legend(fontsize=7, ncol=4, loc="upper right")
    elif (MEM_PREFIX + "stackMin") in df.columns:
        smin = df[MEM_PREFIX + "stackMin"].to_numpy(dtype=float)
        ax[1].plot(t, smin, lw=1.2, color="tab:green")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_ylabel("最小スタック残量 [byte]")
        ax[1].set_title("全タスク中で最も余裕の無いスタック (0 に近づくとオーバーフロー危険)")
        ax[1].grid(alpha=0.3)
        # どのタスクが最小かを注記
        if (MEM_PREFIX + "stackMinId") in df.columns:
            ids = df[MEM_PREFIX + "stackMinId"].to_numpy(dtype=int)
            worst_i = int(np.nanargmin(smin))
            wid = ids[worst_i]
            wname = STACK_TASK_NAMES[wid] if 0 <= wid < len(STACK_TASK_NAMES) else f"id{wid}"
            ax[1].annotate(f"最悪: {wname} ({smin[worst_i]:.0f} byte)",
                           xy=(t[worst_i], smin[worst_i]),
                           xytext=(0.98, 0.9), textcoords="axes fraction",
                           ha="right", fontsize=9,
                           arrowprops=dict(arrowstyle="->", color="tab:green"))
    ax[1].set_xlabel("time [s]")

    if tmax is not None:
        for a in ax:
            a.set_xlim(0, tmax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"メモリグラフを保存: {out_path}")


def summary_mem(df, heap_total=HEAP_TOTAL_DEFAULT):
    heap_free = df[MEM_PREFIX + "heapFree"].to_numpy(dtype=float)
    heap_min = df[MEM_PREFIX + "heapMin"].to_numpy(dtype=float)
    use = 100.0 * (heap_total - heap_free) / heap_total
    peak = 100.0 * (heap_total - heap_min.min()) / heap_total
    print(f"  ヒープ: 使用率 平均 {np.nanmean(use):.1f}% / 最大 {np.nanmax(use):.1f}%"
          f"  (最悪の空き {heap_min.min():.0f} / {heap_total} byte, 使用率 {peak:.1f}%)")
    stack_cols = [c for c in df.columns if c.startswith(STACK_PREFIX)]
    if stack_cols:
        print("  タスク別スタック残量 [byte] (小さい順):")
        mins = sorted(((c[len(STACK_PREFIX):],
                        np.nanmin(df[c].to_numpy(dtype=float)))
                       for c in stack_cols), key=lambda x: x[1])
        for name, mn in mins:
            print(f"    {name:<12}{mn:8.0f}")
        return
    if (MEM_PREFIX + "stackMin") in df.columns:
        smin = df[MEM_PREFIX + "stackMin"].to_numpy(dtype=float)
        msg = f"  最小スタック残量: {smin.min():.0f} byte"
        if (MEM_PREFIX + "stackMinId") in df.columns:
            ids = df[MEM_PREFIX + "stackMinId"].to_numpy(dtype=int)
            wid = ids[int(np.nanargmin(smin))]
            wname = STACK_TASK_NAMES[wid] if 0 <= wid < len(STACK_TASK_NAMES) else f"id{wid}"
            msg += f"  (タスク {wname})"
        print(msg)


def summary(df, cols, percent=False):
    pcol = PREFIX + "period"
    period = df[pcol].to_numpy(dtype=float) if pcol in df.columns else None

    t = df["timestamp"].to_numpy(dtype=float) / 1000.0
    print(f"  サンプル数 {len(df)}, 記録時間 {t[-1] - t[0]:.1f}s")
    if period is not None:
        print(f"  サンプリング区間 median {np.median(period) / 1000.0:.1f} ms")

    print(f"  {'task':<12}{'平均[us]':>10}{'最大[us]':>10}{'平均[%]':>9}{'最大[%]':>9}")
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        name = c[len(PREFIX):]
        if period is not None:
            pct = np.where(period > 0, 100.0 * v / period, np.nan)
            print(f"  {name:<12}{np.nanmean(v):10.0f}{np.nanmax(v):10.0f}"
                  f"{np.nanmean(pct):9.2f}{np.nanmax(pct):9.2f}")
        else:
            print(f"  {name:<12}{np.nanmean(v):10.0f}{np.nanmax(v):10.0f}"
                  f"{'-':>9}{'-':>9}")

    cpu = PREFIX + "cpuLoad"
    if cpu in df.columns:
        v = df[cpu].to_numpy(dtype=float)
        print(f"  cpuLoad: 平均 {np.nanmean(v):.1f}% / 最大 {np.nanmax(v):.1f}%")


def main():
    ap = argparse.ArgumentParser(
        description="tasklog をCSVに変換し、タスクごとの実行時間をプロットする")
    ap.add_argument("logs", nargs="*",
                    help="tasklog のバイナリ (例: tasklog01) または変換済みCSV")
    ap.add_argument("--all", action="store_true",
                    help="フォルダ内の tasklog* をすべて処理する")
    ap.add_argument("--percent", action="store_true",
                    help="実行時間ではなくCPU占有率[%%]で描く")
    ap.add_argument("--top", type=int, default=None,
                    help="平均実行時間の上位N件だけ描く")
    ap.add_argument("--keep-zero", action="store_true",
                    help="実行時間が常に0のタスクも描く")
    ap.add_argument("--with-idle", action="store_true",
                    help="IDLE (CPUの余り) も描く。他のタスクが潰れるので既定は非表示")
    ap.add_argument("--tmax", type=float, default=None, help="表示上限時刻 [s]")
    ap.add_argument("--heap-total", type=int, default=HEAP_TOTAL_DEFAULT,
                    help=f"FreeRTOSヒープ総量[byte] (既定 {HEAP_TOTAL_DEFAULT})")
    ap.add_argument("--no-mem", action="store_true",
                    help="メモリ消費率グラフを描かない")
    ap.add_argument("--event", default="fixedFrequency", help="抽出するイベント名")
    ap.add_argument("-o", "--output", default=None,
                    help="出力PNGパス (入力が1件のときのみ)")
    args = ap.parse_args()

    targets = list(args.logs)
    if args.all or not targets:
        found = sorted(glob.glob(os.path.join(LOGS_DIR, "tasklog*")) +
                       glob.glob(os.path.join(SCRIPT_DIR, "tasklog*")))
        found = [f for f in found if os.path.isfile(f)]
        if not found:
            found = sorted(glob.glob(os.path.join(CSV_DIR, "tasklog*.csv")))
        if not found:
            sys.exit(f"tasklog* が見つかりません ({LOGS_DIR})")
        targets = found

    if args.output and len(targets) > 1:
        sys.exit("-o は入力が1件のときだけ使えます")

    for path in targets:
        df, base = load(path, args.event)
        cols = task_columns(df, keep_zero=args.keep_zero,
                            with_idle=args.with_idle)
        print(f"入力: {path}")
        summary(df, cols, percent=args.percent)
        out = args.output or os.path.join(
            FIG_DIR, f"{base}_taskload{'_pct' if args.percent else ''}.png")
        plot(df, cols, base, out, percent=args.percent, tmax=args.tmax,
             top=args.top)

        # メモリ消費率 (memload.* が記録されていれば)
        if not args.no_mem and has_mem(df):
            summary_mem(df, heap_total=args.heap_total)
            mem_out = os.path.join(FIG_DIR, f"{base}_mem.png")
            plot_mem(df, base, mem_out, heap_total=args.heap_total,
                     tmax=args.tmax)
        elif not args.no_mem:
            print("  (memload.* が無いのでメモリグラフはスキップ)")
        print()


if __name__ == "__main__":
    main()
