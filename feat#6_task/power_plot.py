#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microSD に記録した pwrlog (config_power.txt) を読み込み、消費電力・モーター出力
まわりを可視化する。「電力不足で目標高度に届かない」の切り分けが目的。

見るポイント:
  1. motor.mN (実PWM 0..65535) が上限に張り付く = モーター飽和。
     motor.mNreq (無制限の要求値) が 65535 を超えていれば、機体は出せる以上の
     推力を要求している = パワー不足の決定的証拠。
  2. pm.vbat が離陸負荷で大きく降下(sag)し、飽和を招いていないか。
  3. 目標高度 (ctrltargetZ.z) と実測 (stateEstimateZ.z) の差が、飽和中に
     開いたままか。
  4. 電流計デッキがあれば pm.extCurr から実電力 P = vbat*extCurr。

使い方:
    python3 power_plot.py pwrlog01           # log/ 内のバイナリを変換+描画
    python3 power_plot.py csv/pwrlog01.csv    # 変換済みCSVから描画
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "log")
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "fig")
USD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "usd")

PWM_MAX = 65535   # モーターPWMの上限 (=100% duty)


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


def load(path, event="fixedFrequency"):
    import pandas as pd
    if path.lower().endswith(".csv"):
        if not os.path.isfile(path):
            path = os.path.join(CSV_DIR, os.path.basename(path))
        return pd.read_csv(path), os.path.splitext(os.path.basename(path))[0]

    if not os.path.isfile(path):
        for d in (LOGS_DIR, SCRIPT_DIR):
            cand = os.path.join(d, os.path.basename(path))
            if os.path.isfile(cand):
                path = cand
                break
        else:
            sys.exit(f"ファイルが見つかりません: {path}")

    if USD_DIR not in sys.path:
        sys.path.insert(0, USD_DIR)
    try:
        import cfusdlog
    except ImportError:
        sys.exit(f"cfusdlog.py が見つかりません ({USD_DIR})")

    data = cfusdlog.decode(path)
    if not data or event not in data:
        sys.exit(f"イベント '{event}' がありません: {list(data.keys()) if data else None}")
    df = pd.DataFrame(data[event])
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(CSV_DIR, exist_ok=True)
    out_csv = os.path.join(CSV_DIR, base + ".csv")
    df.to_csv(out_csv, index=False)
    print(f"CSVに変換: {out_csv} ({len(df)} サンプル)")
    return df, base


def col(df, name):
    return df[name].to_numpy(dtype=float) if name in df.columns else None


def analyse(df):
    """飽和・電圧降下・追従誤差の要約を出す。"""
    motors = [col(df, f"motor.m{i}") for i in (1, 2, 3, 4)]
    reqs = [col(df, f"motor.m{i}req") for i in (1, 2, 3, 4)]

    print("  --- モーター飽和 ---")
    if all(m is not None for m in motors):
        for i, m in enumerate(motors, 1):
            sat = np.mean(m >= PWM_MAX * 0.999) * 100.0
            print(f"    m{i}: 平均 {np.mean(m)/PWM_MAX*100:5.1f}% duty, "
                  f"最大 {np.max(m)/PWM_MAX*100:5.1f}%, 飽和時間 {sat:4.1f}%")
    if all(r is not None for r in reqs):
        over = np.zeros(len(df), dtype=bool)
        for r in reqs:
            over |= (r > PWM_MAX)
        print(f"    要求が上限超過 (パワー不足) の時間: {np.mean(over)*100:.1f}%")

    vbat = col(df, "pm.vbat")
    if vbat is not None:
        print("  --- バッテリー ---")
        print(f"    vbat: 開始 {vbat[0]:.2f}V, 最小 {vbat.min():.2f}V, "
              f"降下量 {vbat[0]-vbat.min():.2f}V")
        lvl = col(df, "pm.batteryLevel")
        if lvl is not None:
            print(f"    batteryLevel: {lvl[0]:.0f}% -> {lvl[-1]:.0f}%")
    cur = col(df, "pm.extCurr")
    if cur is not None and np.nanmax(cur) > 0.01:
        p = vbat * cur if vbat is not None else None
        print(f"    extCurr: 平均 {np.nanmean(cur):.2f}A, 最大 {np.nanmax(cur):.2f}A")
        if p is not None:
            print(f"    電力 P=vbat*I: 平均 {np.nanmean(p):.1f}W, 最大 {np.nanmax(p):.1f}W")
    elif cur is not None:
        print("    extCurr: ~0 (電流計デッキ無し。実電力は測れません)")

    tgt = col(df, "ctrltargetZ.z")
    est = col(df, "stateEstimateZ.z")
    if tgt is not None and est is not None:
        err = (tgt - est) / 1000.0
        print("  --- 高度追従 ---")
        print(f"    目標 最大 {tgt.max()/1000:.2f}m, 実測 最大 {est.max()/1000:.2f}m, "
              f"定常誤差 (末尾平均) {np.mean(err[-min(50,len(err)):]):+.3f}m")


def plot(df, base, out_path, tmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_jp_font()

    t = df["timestamp"].to_numpy(dtype=float) / 1000.0
    t = t - t[0]

    fig, ax = plt.subplots(4, 1, figsize=(13, 12), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    cmap = plt.get_cmap("tab10")

    # (1) 実PWM duty と要求 duty
    for i in range(1, 5):
        m = col(df, f"motor.m{i}")
        if m is not None:
            ax[0].plot(t, m / PWM_MAX * 100.0, lw=1.0, color=cmap(i - 1),
                       label=f"m{i} 実PWM")
    for i in range(1, 5):
        r = col(df, f"motor.m{i}req")
        if r is not None:
            ax[0].plot(t, r / PWM_MAX * 100.0, lw=0.8, ls=":", color=cmap(i - 1),
                       label=f"m{i} 要求")
    ax[0].axhline(100.0, color="k", lw=0.8, ls="--", label="上限 100% (飽和)")
    ax[0].set_ylabel("モーター duty [%]")
    ax[0].set_title(f"{base}: モーター出力 (点線=要求値。100%超え=パワー不足)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=3, loc="upper right")

    # (2) 指令推力
    thr = col(df, "stabilizer.thrust")
    if thr is not None:
        ax[1].plot(t, thr, lw=1.0, color="tab:purple")
    ax[1].set_ylabel("指令推力")
    ax[1].grid(alpha=0.3)

    # (3) バッテリー電圧 (+電流があれば電力)
    vbat = col(df, "pm.vbat")
    if vbat is not None:
        ax[2].plot(t, vbat, lw=1.2, color="tab:red", label="vbat [V]")
    ax[2].set_ylabel("vbat [V]", color="tab:red")
    ax[2].tick_params(axis="y", labelcolor="tab:red")
    ax[2].grid(alpha=0.3)
    cur = col(df, "pm.extCurr")
    if cur is not None and np.nanmax(cur) > 0.01:
        a2 = ax[2].twinx()
        a2.plot(t, vbat * cur, lw=0.9, color="tab:orange", label="電力 [W]")
        a2.set_ylabel("電力 [W]", color="tab:orange")
        a2.tick_params(axis="y", labelcolor="tab:orange")
    ax[2].legend(fontsize=8, loc="upper right")

    # (4) 目標 vs 実測 高度
    tgt = col(df, "ctrltargetZ.z")
    est = col(df, "stateEstimateZ.z")
    if tgt is not None:
        ax[3].plot(t, tgt / 1000.0, lw=1.2, ls="--", color="tab:gray", label="目標 z [m]")
    if est is not None:
        ax[3].plot(t, est / 1000.0, lw=1.2, color="tab:blue", label="実測 z [m]")
    ax[3].set_ylabel("高度 [m]")
    ax[3].set_xlabel("time [s]")
    ax[3].grid(alpha=0.3)
    ax[3].legend(fontsize=8, loc="upper right")

    if tmax is not None:
        for a in ax:
            a.set_xlim(0, tmax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"グラフを保存: {out_path}")


def is_scope_csv(path):
    """オシロスコープ (Tektronix MDO) が書き出した波形CSVか。"""
    if not (path.lower().endswith(".csv") and os.path.isfile(path)):
        return False
    with open(path, "r", errors="replace") as f:
        return f.readline().startswith("Model,")


def load_scope(path):
    """オシロの波形CSV (先頭に機器情報、'TIME,CH1' 以降が波形) を読む。"""
    import pandas as pd
    with open(path, "r", errors="replace") as f:
        for skip, line in enumerate(f):
            if line.startswith("TIME,"):
                break
        else:
            sys.exit(f"'TIME,' の行がありません: {path}")
    df = pd.read_csv(path, skiprows=skip)
    return df, os.path.splitext(os.path.basename(path))[0]


def scope_power(df, shunt, supply):
    """シャント抵抗の電圧降下から 電流 [A] と 機体側の消費電力 [W] を求める。"""
    v = df.iloc[:, 1].to_numpy(dtype=float)
    cur = v / shunt
    # 電源電圧のうちシャントで落ちた分は機体に届かない
    return v, cur, (supply - v) * cur


def analyse_scope(df, shunt, supply):
    t = df.iloc[:, 0].to_numpy(dtype=float)
    v, cur, p = scope_power(df, shunt, supply)
    print(f"  記録時間 {t[-1]-t[0]:.1f}s, シャント {shunt}Ω, 電源 {supply}V")
    print(f"    シャント電圧: 平均 {v.mean()*1000:.1f}mV, 最小 {v.min()*1000:.0f}mV, "
          f"最大 {v.max()*1000:.0f}mV")
    print(f"    電流: 平均 {cur.mean():.3f}A, 最大 {cur.max():.3f}A")
    print(f"    機体電力 P=(Vs-Vsh)*I: 平均 {p.mean():.3f}W, 最大 {p.max():.3f}W")
    print(f"    電源電力 P=Vs*I      : 平均 {(supply*cur).mean():.3f}W")


def plot_scope(df, base, out_path, shunt, supply, tmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_jp_font()

    t = df.iloc[:, 0].to_numpy(dtype=float)
    t = t - t[0]
    v, cur, p = scope_power(df, shunt, supply)
    # オシロの量子化ノイズが大きいので 0.1s の移動平均も重ねる
    n = max(1, int(round(0.1 / np.median(np.diff(t)))))
    smooth = lambda x: np.convolve(x, np.ones(n) / n, mode="same")

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for a, y, color, ylabel in (
            (ax[0], v * 1000.0, "tab:blue", "シャント電圧 [mV]"),
            (ax[1], cur, "tab:green", "電流 [A]"),
            (ax[2], p, "tab:orange", "機体電力 [W]")):
        a.plot(t, y, lw=0.5, color=color, alpha=0.4, label="生値")
        a.plot(t[n:-n], smooth(y)[n:-n], lw=1.2, color=color, label="移動平均 0.1s")
        a.axhline(y.mean(), color="k", lw=0.8, ls="--",
                  label=f"平均 {y.mean():.3f}")
        a.set_ylabel(ylabel)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"{base}: 消費電力 (シャント {shunt}Ω, 電源 {supply}V)")
    ax[2].set_xlabel("time [s]")

    if tmax is not None:
        for a in ax:
            a.set_xlim(0, tmax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"グラフを保存: {out_path}")


def main():
    global FIG_DIR, CSV_DIR
    ap = argparse.ArgumentParser(description="pwrlog を解析して電力/モーター出力を描く")
    ap.add_argument("log", help="pwrlog バイナリ (例: pwrlog01) または CSV "
                                "(オシロの波形CSVならシャント電圧から電力を描く)")
    ap.add_argument("--tmax", type=float, default=None, help="表示上限時刻 [s]")
    ap.add_argument("-o", "--output", default=None, help="出力PNGパス")
    ap.add_argument("--fig-dir", default=None, help=f"PNGの出力先 (既定 {FIG_DIR})")
    ap.add_argument("--csv-dir", default=None, help=f"CSVの出力先 (既定 {CSV_DIR})")
    ap.add_argument("--shunt", type=float, default=0.22,
                    help="シャント抵抗 [Ω] (オシロCSV用, 既定 0.22)")
    ap.add_argument("--supply", type=float, default=3.7,
                    help="外部電源の電圧 [V] (オシロCSV用, 既定 3.7)")
    args = ap.parse_args()
    if args.fig_dir:
        FIG_DIR = os.path.abspath(args.fig_dir)
    if args.csv_dir:
        CSV_DIR = os.path.abspath(args.csv_dir)

    if is_scope_csv(args.log):
        df, base = load_scope(args.log)
        print(f"入力: {args.log}  ({len(df)} サンプル)")
        analyse_scope(df, args.shunt, args.supply)
        out = args.output or os.path.join(FIG_DIR, f"{base}_power.png")
        plot_scope(df, base, out, args.shunt, args.supply, tmax=args.tmax)
        return

    df, base = load(args.log)
    print(f"入力: {args.log}  ({len(df)} サンプル)")
    analyse(df)
    out = args.output or os.path.join(FIG_DIR, f"{base}_power.png")
    plot(df, base, out, tmax=args.tmax)


if __name__ == "__main__":
    main()
