#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chamber.planeOk の内訳を解析する。

app_cf_brushless_chamber_takeoff_Pcontrol_wall.c は、PCA+RANSAC の平面フィットが
どの壁で成功したかを 4bit のマスクで記録している:

    bit0 = front (s1)   bit1 = back (s7)   bit2 = left (s5)   bit3 = right (s6)

壁センタリングは対向ペアの両方が有効なときだけ駆動されるので、
「センサーごとの成功率」と「軸ごとの成功率 (= 実際に制御できた割合)」の
両方を出す。

使い方:
    python3 plot_chamber_planeok.py log200_pose.csv
    python3 plot_chamber_planeok.py log200_pose.csv --save log200_planeok.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIRS = [os.path.join(SCRIPT_DIR, "csv"),
            os.path.join(os.path.dirname(SCRIPT_DIR), "usd", "csv")]
FIGS_DIR = os.path.join(SCRIPT_DIR, "figs")

# app 側の planeOkLog と同じ並び
BITS = [("front", 0, "s1"), ("back", 1, "s7"), ("left", 2, "s5"), ("right", 3, "s6")]
AXES = [("x (front/back)", 0, 1), ("y (left/right)", 2, 3)]

STATE_NAMES = {5: "TAKEOFF", 6: "HOVER", 7: "LAND"}
STATE_COLORS = {5: "#cfe8ff", 6: "#d8f5d0", 7: "#ffe0cc"}
STATE_HOVER = 6


def resolve_csv(path):
    if os.path.isfile(path):
        return path
    for d in CSV_DIRS:
        cand = os.path.join(d, os.path.basename(path))
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"CSVが見つかりません: {path}\n探した場所: {CSV_DIRS}")


def bit(mask, b):
    return ((mask.astype(int) >> b) & 1).astype(bool)


def report(label, mask, out):
    n = len(mask)
    if n == 0:
        return
    out.append(f"\n--- {label}  (n={n}) ---")
    out.append("  センサーごと:")
    for name, b, sid in BITS:
        ok = bit(mask, b)
        out.append(f"    {name:6s} ({sid}): {100*ok.mean():6.1f} %")
    out.append("  軸ごと (対向ペアの両方が有効 = 制御が働いた割合):")
    for name, b0, b1 in AXES:
        both = bit(mask, b0) & bit(mask, b1)
        out.append(f"    {name:15s}: {100*both.mean():6.1f} %")
    nseen = sum(bit(mask, b).astype(int) for _, b, _ in BITS)
    out.append("  同時に取れた壁の枚数:")
    for k in range(5):
        out.append(f"    {k} 枚: {100*np.mean(nseen == k):6.1f} %")
    both_axes = (bit(mask, 0) & bit(mask, 1)) & (bit(mask, 2) & bit(mask, 3))
    out.append(f"  両軸とも制御可能: {100*both_axes.mean():.1f} %")


def main():
    ap = argparse.ArgumentParser(description="chamber.planeOk の内訳解析")
    ap.add_argument("pose_csv")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    p = pd.read_csv(resolve_csv(args.pose_csv))
    if "chamber.planeOk" not in p.columns:
        raise SystemExit("chamber.planeOk が入っていません (config.txt に追加が必要)")

    t0 = p["timestamp"].iloc[0]
    t = (p["timestamp"].to_numpy() - t0) / 1000.0
    st = p["chamber.state"].to_numpy()
    mask = p["chamber.planeOk"].to_numpy()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(f"chamber.planeOk breakdown  ({os.path.basename(args.pose_csv)})",
                 fontsize=15, fontweight="bold")

    def shade(ax):
        edges = np.flatnonzero(np.diff(st)) + 1
        for s, e in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(st)]))):
            c = STATE_COLORS.get(int(st[s]))
            if c:
                ax.axvspan(t[s], t[e] if e < len(t) else t[-1], color=c, alpha=0.55, lw=0, zorder=0)
        ax.grid(alpha=0.3)
        ax.set_xlabel("time since takeoff [s]")

    hov = st == STATE_HOVER

    # (0,0) センサーごとの成功率 (全体 / HOVER)
    ax = axes[0, 0]
    names = [f"{n}\n({s})" for n, _, s in BITS]
    all_r = [100 * bit(mask, b).mean() for _, b, _ in BITS]
    hov_r = [100 * bit(mask[hov], b).mean() for _, b, _ in BITS]
    x = np.arange(len(BITS))
    ax.bar(x - 0.2, all_r, 0.4, label="all phases")
    ax.bar(x + 0.2, hov_r, 0.4, label="HOVER only")
    for i, (a, h) in enumerate(zip(all_r, hov_r)):
        ax.text(i - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.2, h + 1, f"{h:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("plane fit success [%]"); ax.set_ylim(0, 108)
    ax.set_title("Per-sensor plane-fit success rate")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # (0,1) 軸ごと
    ax = axes[0, 1]
    anames = [n for n, _, _ in AXES]
    all_a = [100 * (bit(mask, b0) & bit(mask, b1)).mean() for _, b0, b1 in AXES]
    hov_a = [100 * (bit(mask[hov], b0) & bit(mask[hov], b1)).mean() for _, b0, b1 in AXES]
    x = np.arange(len(AXES))
    ax.bar(x - 0.2, all_a, 0.4, label="all phases")
    ax.bar(x + 0.2, hov_a, 0.4, label="HOVER only")
    for i, (a, h) in enumerate(zip(all_a, hov_a)):
        ax.text(i - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.2, h + 1, f"{h:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(anames)
    ax.set_ylabel("both sensors valid [%]"); ax.set_ylim(0, 108)
    ax.set_title("Per-axis: fraction where centering could actually run")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # (1,0) 時系列 (センサーごとの ON/OFF)
    ax = axes[1, 0]
    shade(ax)
    for i, (name, b, sid) in enumerate(BITS):
        v = bit(mask, b).astype(float)
        ax.plot(t, v * 0.8 + i, lw=1.2, drawstyle="steps-post", label=f"{name} ({sid})")
    ax.set_yticks([i + 0.4 for i in range(len(BITS))])
    ax.set_yticklabels([n for n, _, _ in BITS])
    ax.set_title("Per-sensor validity over time (high = plane fitted)")
    ax.set_ylim(-0.2, len(BITS))

    # (1,1) 同時に取れた枚数
    ax = axes[1, 1]
    shade(ax)
    nseen = sum(bit(mask, b).astype(int) for _, b, _ in BITS)
    ax.plot(t, nseen, lw=1.2, drawstyle="steps-post", color="tab:purple")
    ax.set_ylabel("walls fitted simultaneously")
    ax.set_yticks(range(5)); ax.set_ylim(-0.2, 4.2)
    ax.set_title("How many of the four walls were fitted at once")

    # (2,0) 枚数のヒストグラム
    ax = axes[2, 0]
    for lab, m in (("all phases", np.ones(len(mask), bool)), ("HOVER", hov)):
        cnt = [100 * np.mean(nseen[m] == k) for k in range(5)]
        ax.bar(np.arange(5) + (-0.2 if lab == "all phases" else 0.2), cnt, 0.4, label=lab)
    ax.set_xticks(range(5)); ax.set_xlabel("number of walls fitted")
    ax.set_ylabel("time share [%]")
    ax.set_title("Distribution of simultaneously fitted walls")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # (2,1) 制御が使った距離の有効性 (参考)
    ax = axes[2, 1]
    shade(ax)
    for c, lab in (("chamber.dFront", "front"), ("chamber.dBack", "back"),
                   ("chamber.dLeft", "left"), ("chamber.dRight", "right")):
        if c in p.columns:
            d = p[c].to_numpy(dtype=float)
            ax.plot(t, np.where(d > 0, d, np.nan), lw=1.1, label=lab)
    ax.axhline(0.25, color="k", ls="--", lw=0.9)
    ax.set_ylabel("distance used by control [m]")
    ax.set_title("chamber.dXxx (gaps = axis undriven)")
    ax.legend(fontsize=8, ncol=2)

    out = args.save or (os.path.splitext(os.path.basename(args.pose_csv))[0].replace("_pose", "") + "_planeok.png")
    if not os.path.dirname(out):
        out = os.path.join(FIGS_DIR, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"保存しました: {out}")

    lines = []
    report("全フェーズ", mask, lines)
    for sv in (5, 6, 7):
        m = st == sv
        if m.any():
            report(STATE_NAMES[sv], mask[m], lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
