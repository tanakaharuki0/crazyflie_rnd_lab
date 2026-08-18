#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chamber ミッション (app_cf_brushless_chamber) のuSDログを診断用にプロットする。

入力は log_to_csv.py が出した2つのCSV:
  - fixedFrequency ストリーム (vl53l8cx.s0..s10)  ... 壁までの距離、約50Hz
  - pose ストリーム (chamber.state, stateEstimate.*, stabilizer.*) ... 約10Hz

ファームの壁センタリング則を同じ式で再現して重ねるので、
「センサが何を見て」「制御が何を指令して」「機体がどう動いたか」を
1枚で突き合わせられる。

使い方:
    python3 plot_chamber.py log10.csv log10_pose.csv
    python3 plot_chamber.py log10.csv log10_pose.csv --save log10_diag.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "usd", "csv")
FIGS_DIR = os.path.join(SCRIPT_DIR, "figs")

# app_cf_brushless_chamber.c の WALL_SENSOR_* と同じ割り当て
SENSOR_FRONT = "vl53l8cx.s1"
SENSOR_BACK = "vl53l8cx.s7"
SENSOR_LEFT = "vl53l8cx.s5"
SENSOR_RIGHT = "vl53l8cx.s6"
SENSOR_UP = "vl53l8cx.s8"

# ファームのセンタリング則のパラメータ (chamber.kCenter / chamber.centerMaxV)
K_CENTER = 0.7
CENTER_MAX_V = 0.05

HALF_BOX_M = 0.25  # 50cm チャンバの中心から壁まで

# app_cf_brushless_chamber.c の appState_t
STATE_NAMES = {
    0: "IDLE", 1: "WAIT_DECK", 2: "PREARM", 3: "ARMING",
    4: "ARM_DWELL", 5: "TAKEOFF", 6: "HOVER", 7: "LAND",
}
STATE_COLORS = {5: "#cfe8ff", 6: "#d8f5d0", 7: "#ffe0cc"}


def resolve_csv(path):
    if os.path.isfile(path):
        return path
    cand = os.path.join(CSV_DIR, os.path.basename(path))
    if os.path.isfile(cand):
        return cand
    raise SystemExit(f"CSVが見つかりません: {path} (({CSV_DIR}) も確認しました)")


STATE_HOVER = 6


def hover_bounds(pose, t0):
    """APP_HOVER の開始/終了時刻 [s] を返す。無ければ None。"""
    h = pose["chamber.state"] == STATE_HOVER
    if not h.any():
        return None
    t = (pose.loc[h, "timestamp"].to_numpy() - t0) / 1000.0
    return t[0], t[-1]


def shade_states(ax, pose, t0, hover=None):
    """FSMの状態ごとに背景を塗り分け、HOVER の開始/終了に黒の縦線を引く。"""
    st = pose["chamber.state"].to_numpy()
    t = (pose["timestamp"].to_numpy() - t0) / 1000.0
    edges = np.flatnonzero(np.diff(st)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(st)]))
    for s, e in zip(starts, ends):
        val = int(st[s])
        color = STATE_COLORS.get(val)
        if color is None:
            continue
        t_end = t[e] if e < len(t) else t[-1]
        ax.axvspan(t[s], t_end, color=color, alpha=0.55, lw=0, zorder=0)

    if hover is not None:
        for tv in hover:
            ax.axvline(tv, color="k", lw=1.6, zorder=5)


def state_legend_handles(hover=None):
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    h = [mpatches.Patch(color=c, alpha=0.55, label=STATE_NAMES[s])
         for s, c in STATE_COLORS.items()]
    if hover is not None:
        h.append(Line2D([0], [0], color="k", lw=1.6, label="HOVER start/end"))
    return h


def main():
    ap = argparse.ArgumentParser(description="chamber ログ診断プロット")
    ap.add_argument("ff_csv", help="fixedFrequency ストリームのCSV (例 log10.csv)")
    ap.add_argument("pose_csv", help="pose ストリームのCSV (例 log10_pose.csv)")
    ap.add_argument("--save", default=None, help="保存名 (ディレクトリ未指定なら figs/ 下)")
    args = ap.parse_args()

    ff = pd.read_csv(resolve_csv(args.ff_csv))
    pose = pd.read_csv(resolve_csv(args.pose_csv))

    # 時間軸は「pose の最初のサンプル = 離陸開始」を 0 s とする
    t0 = pose["timestamp"].iloc[0]
    t_ff = (ff["timestamp"].to_numpy() - t0) / 1000.0
    t_ps = (pose["timestamp"].to_numpy() - t0) / 1000.0
    hover = hover_bounds(pose, t0)   # APP_HOVER の開始/終了 [s]

    # 壁距離 [m]。ドライバは中央4ゾーン平均を mm で入れる。0以下は無効。
    def dist(col):
        d = ff[col].to_numpy(dtype=float) / 1000.0
        return np.where(d > 0.0, d, np.nan)

    dF, dB = dist(SENSOR_FRONT), dist(SENSOR_BACK)
    dL, dR = dist(SENSOR_LEFT), dist(SENSOR_RIGHT)
    dU = dist(SENSOR_UP)

    # ファームと同じ式で指令速度を再現 (無効測距の軸は 0)
    def cmd(a, b):
        v = np.clip(K_CENTER * (a - b), -CENTER_MAX_V, CENTER_MAX_V)
        return np.where(np.isnan(a) | np.isnan(b), 0.0, v)

    vx_cmd, vy_cmd = cmd(dF, dB), cmd(dL, dR)

    # 中心からのずれ [m]。+x は中心が前方 = 機体は後ろ寄り。
    ex, ey = (dF - dB) / 2.0, (dL - dR) / 2.0
    span_x, span_y = dF + dB, dL + dR

    fig, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    fig.suptitle(f"chamber wall-centering diagnosis  ({os.path.basename(args.ff_csv)})",
                 fontsize=15, fontweight="bold")

    # --- (0,0) 壁距離 ---
    ax = axes[0, 0]
    shade_states(ax, pose, t0, hover)
    ax.plot(t_ff, dF, label="front (s1)", lw=1.2)
    ax.plot(t_ff, dB, label="back (s7)", lw=1.2)
    ax.plot(t_ff, dL, label="left (s5)", lw=1.2)
    ax.plot(t_ff, dR, label="right (s6)", lw=1.2)
    ax.axhline(HALF_BOX_M, color="k", ls="--", lw=0.9, label="0.25 m (centred)")
    ax.set_ylabel("wall distance [m]")
    ax.set_title("VL53L8CX central-4-zone distances")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + state_legend_handles(hover),
              fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # --- (0,1) 上向き ---
    ax = axes[0, 1]
    shade_states(ax, pose, t0, hover)
    ax.plot(t_ff, dU, color="tab:purple", lw=1.2, label="up (s8)")
    ax.set_ylabel("up distance [m]")
    ax.set_title("Up-facing sensor (ceiling / hand trigger)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (1,0) 中心からのずれ ---
    ax = axes[1, 0]
    shade_states(ax, pose, t0, hover)
    ax.plot(t_ff, ex, label="ex = (front-back)/2", lw=1.3)
    ax.plot(t_ff, ey, label="ey = (left-right)/2", lw=1.3)
    ax.axhline(0, color="k", lw=0.8)
    for s in (+1, -1):
        ax.axhline(s * CENTER_MAX_V / K_CENTER / 2, color="tab:red", ls=":", lw=1.0)
    ax.text(0.01, 0.03, "outside red dotted lines = command saturated (|e| > 3.6 cm)",
            transform=ax.transAxes, fontsize=8, color="tab:red")
    ax.set_ylabel("offset from centre [m]")
    ax.set_title("Centering error (0 = centred)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (1,1) 対向センサの和 = 見かけの箱の幅 ---
    ax = axes[1, 1]
    shade_states(ax, pose, t0, hover)
    ax.plot(t_ff, span_x, label="front+back", lw=1.2)
    ax.plot(t_ff, span_y, label="left+right", lw=1.2)
    ax.axhline(2 * HALF_BOX_M, color="k", ls="--", lw=0.9, label="0.50 m (box)")
    ax.set_ylabel("opposing pair sum [m]")
    ax.set_title("Sanity check: should stay ~0.5 m while inside the box")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (2,0) 再現した指令速度 ---
    ax = axes[2, 0]
    shade_states(ax, pose, t0, hover)
    ax.plot(t_ff, vx_cmd, label="vx cmd (body fwd)", lw=1.3)
    ax.plot(t_ff, vy_cmd, label="vy cmd (body left)", lw=1.3)
    for s in (+1, -1):
        ax.axhline(s * CENTER_MAX_V, color="tab:red", ls=":", lw=1.0)
    ax.set_ylabel("commanded velocity [m/s]")
    ax.set_title(f"Recomputed centering command  (k={K_CENTER}, clamp=±{CENTER_MAX_V} m/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (2,1) 位置推定 ---
    ax = axes[2, 1]
    shade_states(ax, pose, t0, hover)
    ax.plot(t_ps, pose["stateEstimate.x"], label="x", lw=1.3)
    ax.plot(t_ps, pose["stateEstimate.y"], label="y", lw=1.3)
    ax.plot(t_ps, pose["stateEstimate.z"], label="z", lw=1.3)
    for s in (+1, -1):
        ax.axhline(s * HALF_BOX_M, color="k", ls="--", lw=0.8)
    ax.set_ylabel("estimator position [m]")
    ax.set_title("stateEstimate (dashed = box wall at ±0.25 m from takeoff point)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (3,0) 姿勢 ---
    ax = axes[3, 0]
    shade_states(ax, pose, t0, hover)
    ax.plot(t_ps, pose["stabilizer.roll"], label="roll", lw=1.2)
    ax.plot(t_ps, pose["stabilizer.pitch"], label="pitch", lw=1.2)
    ax.plot(t_ps, pose["stabilizer.yaw"], label="yaw", lw=1.2)
    ax.set_xlabel("time since takeoff [s]")
    ax.set_ylabel("attitude [deg]")
    ax.set_title("Attitude (large tilt corrupts the ranges and means loss of control)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- (3,1) XY 軌跡 ---
    ax = axes[3, 1]
    x = pose["stateEstimate.x"].to_numpy()
    y = pose["stateEstimate.y"].to_numpy()
    sc = ax.scatter(x, y, c=t_ps, cmap="viridis", s=12, zorder=3)
    ax.plot(x, y, color="gray", lw=0.7, alpha=0.7, zorder=2)
    ax.plot(x[0], y[0], "go", ms=9, label="takeoff", zorder=4)
    ax.plot(x[-1], y[-1], "rx", ms=11, mew=2, label="end", zorder=4)
    ax.add_patch(plt.Rectangle((-HALF_BOX_M, -HALF_BOX_M), 2 * HALF_BOX_M, 2 * HALF_BOX_M,
                               fill=False, ec="k", ls="--", lw=1.2, label="50 cm chamber"))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory (estimator frame, origin = takeoff)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="t [s]")

    for r in range(3):
        for c in range(2):
            axes[r, c].set_xlabel("time since takeoff [s]")

    out = args.save or (os.path.splitext(os.path.basename(args.ff_csv))[0] + "_diag.png")
    if not os.path.dirname(out):
        out = os.path.join(FIGS_DIR, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"保存しました: {out}")

    # ---- 数値サマリ ----
    hover = pose["chamber.state"] == 6
    if hover.any():
        th0, th1 = pose.loc[hover, "timestamp"].iloc[0], pose.loc[hover, "timestamp"].iloc[-1]
        m = (ff["timestamp"] >= th0) & (ff["timestamp"] <= th1)
        print("\n=== HOVER 区間の統計 ===")
        print(f"  区間: {(th0 - t0)/1000:.2f} - {(th1 - t0)/1000:.2f} s")
        for name, arr in (("front", dF), ("back", dB), ("left", dL), ("right", dR)):
            a = arr[m.to_numpy()]
            print(f"  {name:5s}: min={np.nanmin(a):.3f}  mean={np.nanmean(a):.3f}  max={np.nanmax(a):.3f} m")
        for name, arr in (("ex", ex), ("ey", ey)):
            a = arr[m.to_numpy()]
            sat = np.nanmean(np.abs(a) > CENTER_MAX_V / K_CENTER / 2) * 100
            print(f"  {name}: mean={np.nanmean(a):+.3f} m, |max|={np.nanmax(np.abs(a)):.3f} m, 飽和率={sat:.0f}%")
        print(f"  |roll|max={pose.loc[hover,'stabilizer.roll'].abs().max():.1f} deg, "
              f"|pitch|max={pose.loc[hover,'stabilizer.pitch'].abs().max():.1f} deg, "
              f"yaw drift={pose.loc[hover,'stabilizer.yaw'].iloc[-1] - pose.loc[hover,'stabilizer.yaw'].iloc[0]:+.1f} deg")


if __name__ == "__main__":
    main()
