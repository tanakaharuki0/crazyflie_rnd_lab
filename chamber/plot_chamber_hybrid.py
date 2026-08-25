#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid 版 chamber ミッション (logs_k_0.3_hybrid / log200~) のログを時系列で可視化する。

hybrid ファームは壁までの距離を「平面フィット (planeOk)」と「素の測距」の
2系統から作り、軸ごとにどちらを使ったかを chamber.srcX / chamber.srcY に残す。
そのため plot_chamber.py の診断に加えて、

  - srcX / srcY の切り替わり (どちらのソースで飛んでいたか)
  - planeOk 4bit の内訳 (front/back/left/right)
  - 壁距離 dFront/dBack/dLeft/dRight と、その差 (センタリング誤差)
  - 速度指令 vxCmd/vyCmd と holdX/holdY
  - yaw 整列 (wallYaw / yawCmd / yawAlignOk)
  - 位置・姿勢・推力・下向き測距
  - fixedFrequency ストリームの生センサー値

を1枚にまとめる。

使い方:
    python3 plot_chamber_hybrid.py log200_k_0.3_hybrid.csv log200_k_0.3_hybrid_pose.csv
    python3 plot_chamber_hybrid.py --all          # csv/log2*_hybrid を全部処理して figs/ に保存
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
FIGS_DIR = os.path.join(SCRIPT_DIR, "figs")

STATE_NAMES = {
    0: "IDLE", 1: "WAIT_DECK", 2: "PREARM", 3: "ARMING",
    4: "ARM_DWELL", 5: "TAKEOFF", 6: "HOVER", 7: "LAND",
}
STATE_COLORS = {5: "#cfe8ff", 6: "#d8f5d0", 7: "#ffe0cc"}
STATE_HOVER = 6

# app 側の planeOkLog と同じ並び
PLANE_BITS = [("front", 0, "s1"), ("back", 1, "s7"),
              ("left", 2, "s5"), ("right", 3, "s6")]

# fixedFrequency ストリームのセンサー割り当て (app_cf_brushless_chamber.c)
FF_SENSORS = [("front", "vl53l8cx.s1"), ("back", "vl53l8cx.s7"),
              ("left", "vl53l8cx.s5"), ("right", "vl53l8cx.s6"),
              ("up", "vl53l8cx.s8")]

HALF_BOX_M = 0.25  # 50cm チャンバの中心から壁まで


def resolve_csv(path):
    if os.path.isfile(path):
        return path
    cand = os.path.join(CSV_DIR, os.path.basename(path))
    if os.path.isfile(cand):
        return cand
    raise SystemExit(f"CSVが見つかりません: {path} ({CSV_DIR} も確認しました)")


def k_from_filename(path):
    m = re.search(r"_k_([0-9]*\.?[0-9]+)", os.path.basename(path))
    return float(m.group(1)) if m else None


def rel_t(df, t0):
    return (df["timestamp"].to_numpy() - t0) / 1000.0


def hover_bounds(pose, t0):
    h = pose["chamber.state"] == STATE_HOVER
    if not h.any():
        return None
    t = (pose.loc[h, "timestamp"].to_numpy() - t0) / 1000.0
    return t[0], t[-1]


def shade_states(ax, pose, t0, hover=None):
    st = pose["chamber.state"].to_numpy()
    t = rel_t(pose, t0)
    edges = np.flatnonzero(np.diff(st)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(st)]))
    for s, e in zip(starts, ends):
        color = STATE_COLORS.get(int(st[s]))
        if color is None:
            continue
        t_end = t[e] if e < len(t) else t[-1]
        ax.axvspan(t[s], t_end, color=color, alpha=0.55, lw=0, zorder=0)
    if hover is not None:
        for tv in hover:
            ax.axvline(tv, color="k", lw=1.4, zorder=5)


def state_legend_handles(hover=None):
    h = [mpatches.Patch(color=c, alpha=0.55, label=STATE_NAMES[s])
         for s, c in STATE_COLORS.items()]
    if hover is not None:
        h.append(Line2D([0], [0], color="k", lw=1.4, label="HOVER start/end"))
    return h


def col(df, name):
    """列があれば numpy 配列、無ければ None。"""
    return df[name].to_numpy() if name in df.columns else None


def plot_one(ff_csv, pose_csv, save=None, k=None):
    ff = pd.read_csv(resolve_csv(ff_csv)) if ff_csv else None
    pose = pd.read_csv(resolve_csv(pose_csv))

    # 時刻原点は両ストリームの早い方に揃える
    t0 = pose["timestamp"].iloc[0]
    if ff is not None and len(ff):
        t0 = min(t0, ff["timestamp"].iloc[0])
    tp = rel_t(pose, t0)
    hover = hover_bounds(pose, t0)
    if k is None:
        k = k_from_filename(pose_csv)

    nrows = 8
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 3.0 * nrows), sharex=True)
    title = os.path.basename(pose_csv).replace("_pose.csv", "")
    fig.suptitle(f"chamber hybrid diagnostics — {title}"
                 + (f"   (kCenter={k})" if k is not None else ""),
                 fontsize=15, y=0.997)

    # --- 1. FSM state ------------------------------------------------------
    ax = axes[0]
    ax.step(tp, pose["chamber.state"], where="post", color="k", lw=1.5)
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("chamber.state")
    ax.set_yticks(sorted(STATE_NAMES))
    ax.set_yticklabels([STATE_NAMES[s] for s in sorted(STATE_NAMES)], fontsize=8)
    ax.legend(handles=state_legend_handles(hover), fontsize=8, ncol=4, loc="upper left")
    ax.grid(alpha=0.3)

    # --- 2. 距離ソース srcX / srcY ----------------------------------------
    ax = axes[1]
    for name, color in (("chamber.srcX", "#1f77b4"), ("chamber.srcY", "#d62728")):
        c = col(pose, name)
        if c is not None:
            ax.step(tp, c, where="post", lw=1.6, color=color, label=name)
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("distance source")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (raw)", "1 (plane)"])
    ax.set_ylim(-0.2, 1.2)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # --- 3. planeOk 4bit の内訳 -------------------------------------------
    ax = axes[2]
    mask = col(pose, "chamber.planeOk")
    if mask is not None:
        mask = mask.astype(int)
        for i, (name, bit, sid) in enumerate(PLANE_BITS):
            ok = (mask >> bit) & 1
            ax.fill_between(tp, i, i + ok * 0.8, step="post",
                            color="#4c9f70", alpha=0.85, lw=0)
            ax.text(tp[0], i + 0.4, f" {name} ({sid})  {100*ok.mean():.0f}%",
                    va="center", fontsize=8)
        ay = col(pose, "chamber.yawAlignOk")
        if ay is not None:
            i = len(PLANE_BITS)
            ax.fill_between(tp, i, i + ay * 0.8, step="post",
                            color="#c78d3a", alpha=0.85, lw=0)
            ax.text(tp[0], i + 0.4, f" yawAlignOk  {100*ay.mean():.0f}%",
                    va="center", fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(0, len(PLANE_BITS) + 1)
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("planeOk bits")
    ax.grid(alpha=0.3, axis="x")

    # --- 4. 壁までの距離 ---------------------------------------------------
    ax = axes[3]
    walls = [("chamber.dFront", "#1f77b4"), ("chamber.dBack", "#7fb2d8"),
             ("chamber.dLeft", "#d62728"), ("chamber.dRight", "#f0928f")]
    for name, color in walls:
        c = col(pose, name)
        if c is not None:
            ax.plot(tp, c, lw=1.3, color=color, label=name.split(".")[1])
    ax.axhline(HALF_BOX_M, color="gray", ls="--", lw=1,
               label=f"half box {HALF_BOX_M} m")
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("wall distance [m]")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    ax.grid(alpha=0.3)

    # --- 5. センタリング誤差 (対向差) --------------------------------------
    ax = axes[4]
    df_, db_ = col(pose, "chamber.dFront"), col(pose, "chamber.dBack")
    dl_, dr_ = col(pose, "chamber.dLeft"), col(pose, "chamber.dRight")
    if df_ is not None and db_ is not None:
        ax.plot(tp, df_ - db_, lw=1.4, color="#1f77b4", label="dFront - dBack (x)")
    if dl_ is not None and dr_ is not None:
        ax.plot(tp, dl_ - dr_, lw=1.4, color="#d62728", label="dLeft - dRight (y)")
    ax.axhline(0, color="k", lw=0.8)
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("centering error [m]")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # --- 6. 速度指令と hold フラグ ----------------------------------------
    ax = axes[5]
    for name, color in (("chamber.vxCmd", "#1f77b4"), ("chamber.vyCmd", "#d62728")):
        c = col(pose, name)
        if c is not None:
            ax.plot(tp, c, lw=1.4, color=color, label=name.split(".")[1])
    ax.axhline(0, color="k", lw=0.8)
    ax2 = ax.twinx()
    for name, color, off in (("chamber.holdX", "#1f77b4", 0.0),
                             ("chamber.holdY", "#d62728", 0.06)):
        c = col(pose, name)
        if c is not None:
            ax2.step(tp, c + off, where="post", lw=1.0, ls=":", color=color,
                     label=name.split(".")[1])
    ax2.set_ylim(-0.2, 1.4)
    ax2.set_yticks([0, 1])
    ax2.set_ylabel("hold flag")
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("velocity cmd [m/s]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=4, loc="upper left")
    ax.grid(alpha=0.3)

    # --- 7. yaw 整列 -------------------------------------------------------
    ax = axes[6]
    for name, color in (("stabilizer.yaw", "k"), ("chamber.wallYaw", "#2a9d8f"),
                        ("chamber.yawCmd", "#e76f51")):
        c = col(pose, name)
        if c is not None:
            ax.plot(tp, c, lw=1.3, color=color, label=name.split(".")[1])
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("yaw [deg]")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(alpha=0.3)

    # --- 8. 位置 / 高度 ----------------------------------------------------
    ax = axes[7]
    for name, color in (("stateEstimate.x", "#1f77b4"),
                        ("stateEstimate.y", "#d62728"),
                        ("stateEstimate.z", "#2a9d8f")):
        c = col(pose, name)
        if c is not None:
            ax.plot(tp, c, lw=1.3, color=color, label=name.split(".")[1])
    zr = col(pose, "range.zrange")
    if zr is not None:
        ax.plot(tp, zr / 1000.0, lw=1.0, ls="--", color="#888",
                label="range.zrange [m]")
    shade_states(ax, pose, t0, hover)
    ax.set_ylabel("position [m]")
    ax.set_xlabel("time since log start [s]")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.995))

    if save is None:
        save = title + "_diag.png"
    if not os.path.dirname(save):
        os.makedirs(FIGS_DIR, exist_ok=True)
        save = os.path.join(FIGS_DIR, save)
    fig.savefig(save, dpi=130)
    plt.close(fig)
    return save


def plot_sensors(ff_csv, pose_csv, save=None):
    """fixedFrequency ストリームの生センサー値 (約50Hz) を別図で出す。"""
    ff = pd.read_csv(resolve_csv(ff_csv))
    pose = pd.read_csv(resolve_csv(pose_csv))
    t0 = min(ff["timestamp"].iloc[0], pose["timestamp"].iloc[0])
    tf = rel_t(ff, t0)
    hover = hover_bounds(pose, t0)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    title = os.path.basename(ff_csv).replace(".csv", "")
    for name, c in FF_SENSORS:
        if c in ff.columns:
            ax.plot(tf, ff[c].to_numpy() / 1000.0, lw=1.1, label=f"{name} ({c})")
    shade_states(ax, pose, t0, hover)
    ax.axhline(HALF_BOX_M, color="gray", ls="--", lw=1)
    ax.set_ylabel("raw range [m]")
    ax.set_xlabel("time since log start [s]")
    ax.set_title(f"raw VL53L8CX ranges (fixedFrequency) — {title}")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + state_legend_handles(hover),
              fontsize=8, ncol=5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save is None:
        save = title + "_sensors.png"
    if not os.path.dirname(save):
        os.makedirs(FIGS_DIR, exist_ok=True)
        save = os.path.join(FIGS_DIR, save)
    fig.savefig(save, dpi=130)
    plt.close(fig)
    return save


def main():
    ap = argparse.ArgumentParser(description="chamber hybrid ログの時系列可視化")
    ap.add_argument("ff_csv", nargs="?", help="fixedFrequency CSV (例 log200_k_0.3_hybrid.csv)")
    ap.add_argument("pose_csv", nargs="?", help="pose CSV (例 log200_k_0.3_hybrid_pose.csv)")
    ap.add_argument("--all", action="store_true",
                    help="csv/ の *_hybrid ログを全部処理する")
    ap.add_argument("--save", default=None, help="保存名 (ディレクトリ未指定なら figs/ 下)")
    ap.add_argument("--k", type=float, default=None, help="kCenter の明示指定")
    args = ap.parse_args()

    if args.all:
        pairs = []
        for pose_csv in sorted(glob.glob(os.path.join(CSV_DIR, "log*_hybrid_pose.csv"))):
            ff_csv = pose_csv.replace("_pose.csv", ".csv")
            pairs.append((ff_csv if os.path.isfile(ff_csv) else None, pose_csv))
        if not pairs:
            raise SystemExit(f"対象CSVが見つかりません: {CSV_DIR}/log*_hybrid_pose.csv")
        for ff_csv, pose_csv in pairs:
            out = plot_one(ff_csv, pose_csv, k=args.k)
            print(f"保存: {out}")
            if ff_csv:
                print(f"保存: {plot_sensors(ff_csv, pose_csv)}")
        return

    if not args.pose_csv:
        raise SystemExit("ff_csv と pose_csv の2つ、または --all を指定してください。")
    print(f"保存: {plot_one(args.ff_csv, args.pose_csv, args.save, args.k)}")
    if args.ff_csv:
        print(f"保存: {plot_sensors(args.ff_csv, args.pose_csv)}")


if __name__ == "__main__":
    main()
