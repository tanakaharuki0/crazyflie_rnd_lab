#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_cf_brushless_chamber.c のホバリング時 yaw 補正 (tofWallAngle ベース) を診断する。

plot_chamber.py が壁センタリング (並進) 用なのに対し、こちらは yaw 補正専用。
ファームの制御則
    desired = yawDeg + yawAlignSign * wallYaw
    yawCmd += clamp(desired - yawCmd, +-yawRateMax*dt)
が実際に収束しているかを、符号の検証まで含めて確認する。

符号の判定方法:
  wallYaw は「壁法線とのヨーずれ」で、機体が +psi 回ったときに
    slope = d(wallYaw)/d(yaw)
  が -1 なら desired = yaw + wallYaw が絶対方位として一定になり収束する。
  +1 なら desired = yaw - wallYaw が一定で、今の符号は発散する。
  実測の回帰係数を出して、両方の候補のばらつきを比較する。

使い方:
    python3 plot_chamber_yaw.py log73_pose.csv
    python3 plot_chamber_yaw.py log73_pose.csv --ff log73.csv --save log73_yaw.png
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

STATE_NAMES = {0: "IDLE", 1: "WAIT_DECK", 2: "PREARM", 3: "ARMING",
               4: "ARM_DWELL", 5: "TAKEOFF", 6: "HOVER", 7: "LAND"}
STATE_COLORS = {5: "#cfe8ff", 6: "#d8f5d0", 7: "#ffe0cc"}
STATE_HOVER = 6

YAW_RATE_MAX = 5.0     # chamber.yawRateMax の設定値 [deg/s]
HALF_BOX_M = 0.25


def resolve_csv(path):
    if os.path.isfile(path):
        return path
    for d in CSV_DIRS:
        cand = os.path.join(d, os.path.basename(path))
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"CSVが見つかりません: {path}\n探した場所: {CSV_DIRS}")


def wrap180(deg):
    while deg >= 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def hover_bounds(pose, t0):
    h = pose["chamber.state"] == STATE_HOVER
    if not h.any():
        return None
    t = (pose.loc[h, "timestamp"].to_numpy() - t0) / 1000.0
    return t[0], t[-1]


def shade(ax, pose, t0, hover):
    st = pose["chamber.state"].to_numpy()
    t = (pose["timestamp"].to_numpy() - t0) / 1000.0
    edges = np.flatnonzero(np.diff(st)) + 1
    for s, e in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(st)]))):
        col = STATE_COLORS.get(int(st[s]))
        if col:
            ax.axvspan(t[s], t[e] if e < len(t) else t[-1], color=col, alpha=0.55, lw=0, zorder=0)
    if hover:
        for tv in hover:
            ax.axvline(tv, color="k", lw=1.6, zorder=5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("time since takeoff [s]")


def main():
    ap = argparse.ArgumentParser(description="chamber yaw 補正の診断プロット")
    ap.add_argument("pose_csv", help="pose ストリームのCSV (例 log73_pose.csv)")
    ap.add_argument("--ff", default=None, help="fixedFrequency のCSV (省略時は pose 名から推測)")
    ap.add_argument("--save", default=None, help="保存名 (ディレクトリ未指定なら figs/ 下)")
    ap.add_argument("--rate-max", type=float, default=YAW_RATE_MAX, help="chamber.yawRateMax [deg/s]")
    args = ap.parse_args()

    pose = pd.read_csv(resolve_csv(args.pose_csv))
    ff = None
    ff_name = args.ff or os.path.basename(args.pose_csv).replace("_pose.csv", ".csv")
    try:
        ff = pd.read_csv(resolve_csv(ff_name))
    except SystemExit:
        pass

    t0 = pose["timestamp"].iloc[0]
    t = (pose["timestamp"].to_numpy() - t0) / 1000.0
    hov = hover_bounds(pose, t0)

    yaw = pose["stabilizer.yaw"].to_numpy()
    wall = pose["chamber.wallYaw"].to_numpy()
    cmd = pose["chamber.yawCmd"].to_numpy()
    ok = pose["chamber.yawAlignOk"].to_numpy()
    conf = pose["tofwall.conf"].to_numpy()
    angF = pose["tofwall.angleFilt"].to_numpy()
    ang = pose["tofwall.angle"].to_numpy()

    # 解析対象は HOVER かつ推定が有効なサンプルのみ
    m = (pose["chamber.state"].to_numpy() == STATE_HOVER) & (ok == 1)
    ty, yy, ww, cc = t[m], yaw[m], wall[m], cmd[m]

    # --- 符号の判定 ---
    slope, icept = np.polyfit(yy, ww, 1)
    corr = np.corrcoef(yy, ww)[0, 1]
    cand_plus = yy + ww     # desired if yawAlignSign = +1
    cand_minus = yy - ww    # desired if yawAlignSign = -1
    sd_plus, sd_minus = cand_plus.std(), cand_minus.std()
    good_sign = -1.0 if sd_minus < sd_plus else +1.0

    # --- ファームが実際にどちらの符号で動いていたかを推定する ---
    # yawCmd は desired に向かってスルーするので、(desired - yawCmd) の符号と
    # 実際のステップ方向の一致率が高い方が、そのフライトで使われた符号。
    dcmd = np.diff(cc)
    agree_plus = np.mean(np.sign((cand_plus - cc)[:-1]) == np.sign(dcmd)) * 100
    agree_minus = np.mean(np.sign((cand_minus - cc)[:-1]) == np.sign(dcmd)) * 100
    used_sign = -1.0 if agree_minus > agree_plus else +1.0

    # yaw 追従誤差 (指令に姿勢がついてきているか)
    trk = np.array([wrap180(v) for v in (yy - cc)])

    # --- スルーレート ---
    dt = np.diff(ty)
    slew = np.diff(cc) / dt
    sat = np.mean(np.abs(np.diff(cc)) >= args.rate_max * np.median(dt) * 0.995) * 100

    fig, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    fig.suptitle(f"chamber yaw-alignment diagnosis  ({os.path.basename(args.pose_csv)})",
                 fontsize=15, fontweight="bold")

    # (0,0) yaw 系の時系列
    ax = axes[0, 0]
    shade(ax, pose, t0, hov)
    ax.plot(t, yaw, label="stabilizer.yaw (actual)", lw=1.4)
    ax.plot(t, cmd, label="chamber.yawCmd (commanded)", lw=1.4)
    ax.plot(t, wall, label="chamber.wallYaw (measured offset)", lw=1.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("angle [deg]")
    ax.set_title("Yaw: actual / commanded / measured wall offset")
    ax.legend(fontsize=8)

    # (0,1) |wallYaw| の推移 = 収束しているか
    ax = axes[0, 1]
    shade(ax, pose, t0, hov)
    ax.plot(t, np.abs(wall), color="tab:red", lw=1.4, label="|wallYaw|")
    if len(ty) > 2:
        k = np.polyfit(ty, np.abs(ww), 1)
        ax.plot(ty, np.polyval(k, ty), "k--", lw=1.6,
                label=f"trend {k[0]:+.2f} deg/s")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("|wall offset| [deg]")
    ax.set_title("Convergence check: should trend to 0")
    ax.legend(fontsize=8)

    # (1,0) ★符号の検証: wallYaw vs yaw
    ax = axes[1, 0]
    sc = ax.scatter(yy, ww, c=ty, cmap="viridis", s=14)
    xs = np.linspace(yy.min(), yy.max(), 10)
    ax.plot(xs, np.polyval([slope, icept], xs), "r-", lw=2,
            label=f"fit: slope={slope:+.2f} (corr {corr:+.2f})")
    ax.plot(xs, -xs + (ww.mean() + yy.mean()), "b--", lw=1.5,
            label="slope -1 → sign +1 is correct")
    ax.plot(xs, xs + (ww.mean() - yy.mean()), "m--", lw=1.5,
            label="slope +1 → sign -1 is correct")
    ax.set_xlabel("stabilizer.yaw [deg]")
    ax.set_ylabel("chamber.wallYaw [deg]")
    ax.set_title("SIGN TEST (hover, valid samples)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="t [s]")

    # (1,1) desired 候補の安定性
    ax = axes[1, 1]
    shade(ax, pose, t0, hov)
    ax.plot(ty, cand_plus, lw=1.4, label=f"yaw + wallYaw  (sign +1, sd={sd_plus:.1f})")
    ax.plot(ty, cand_minus, lw=1.4, label=f"yaw - wallYaw  (sign -1, sd={sd_minus:.1f})")
    ax.set_ylabel("candidate absolute heading [deg]")
    ax.set_title("Which candidate is a CONSTANT wall-referenced heading?")
    ax.legend(fontsize=8)

    # (2,0) スルーレート
    ax = axes[2, 0]
    shade(ax, pose, t0, hov)
    ax.plot(ty[1:], slew, lw=1.0, label="d(yawCmd)/dt")
    for s in (+1, -1):
        ax.axhline(s * args.rate_max, color="tab:red", ls=":", lw=1.2)
    ax.set_ylabel("yaw setpoint slew [deg/s]")
    ax.set_title(f"Rate limit (±{args.rate_max:.0f} deg/s). saturated {sat:.0f}% of hover")
    ax.legend(fontsize=8)

    # (2,1) yaw 追従誤差 + 推定の健全性
    ax = axes[2, 1]
    shade(ax, pose, t0, hov)
    ax.plot(ty, trk, color="tab:purple", lw=1.3, label="yaw - yawCmd (tracking error)")
    ax.plot(ty, ww, color="tab:red", lw=1.0, alpha=0.7, label="wallYaw")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("[deg]")
    ax.set_title("Does the attitude loop follow the yaw setpoint?")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t, conf, color="tab:green", lw=1.0, alpha=0.5, label="tofwall.conf")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_ylabel("tofwall.conf", color="tab:green")
    ax2.legend(fontsize=7, loc="lower right")

    # (3,0) 生値 vs 平滑値
    ax = axes[3, 0]
    shade(ax, pose, t0, hov)
    ax.plot(t, ang, lw=0.9, alpha=0.6, label="tofwall.angle (raw)")
    ax.plot(t, angF, lw=1.5, label="tofwall.angleFilt (EMA)")
    ax.plot(t, wall, lw=1.2, ls="--", label="chamber.wallYaw (used)")
    ax.axhline(0, color="k", lw=0.8)
    for s in (+45, -45):
        ax.axhline(s, color="gray", ls=":", lw=1.0)
    ax.set_ylabel("wall offset [deg]")
    ax.set_title("Raw vs filtered wall angle (±45 deg = wrap boundary)")
    ax.legend(fontsize=8)

    # (3,1) 並進側: 壁距離
    ax = axes[3, 1]
    shade(ax, pose, t0, hov)
    for c, lab in (("chamber.dFront", "front"), ("chamber.dBack", "back"),
                   ("chamber.dLeft", "left"), ("chamber.dRight", "right")):
        d = pose[c].to_numpy().astype(float)
        ax.plot(t, np.where(d > 0, d, np.nan), lw=1.2, label=lab)
    ax.axhline(HALF_BOX_M, color="k", ls="--", lw=0.9, label="0.25 m")
    ax.set_ylabel("wall distance [m]")
    ax.set_title("Translation loop (for context)")
    ax.legend(fontsize=8, ncol=2)

    out = args.save or (os.path.splitext(os.path.basename(args.pose_csv))[0].replace("_pose", "") + "_yaw.png")
    if not os.path.dirname(out):
        out = os.path.join(FIGS_DIR, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"保存しました: {out}\n")

    # ---------------- 数値まとめ ----------------
    print("=== HOVER 区間 ===")
    print(f"  {ty[0]:.2f} - {ty[-1]:.2f} s ({ty[-1]-ty[0]:.2f} s), 有効サンプル {m.sum()}/{(pose['chamber.state']==STATE_HOVER).sum()}")
    print(f"  loop dt median {np.median(dt)*1000:.1f} ms")
    print()
    print("=== 符号の判定 ===")
    print(f"  wallYaw = {slope:+.3f} * yaw {icept:+.2f}   (corr {corr:+.3f})")
    print(f"  desired = yaw + wallYaw : sd {sd_plus:6.2f} deg, drift {np.polyfit(ty,cand_plus,1)[0]:+.2f} deg/s")
    print(f"  desired = yaw - wallYaw : sd {sd_minus:6.2f} deg, drift {np.polyfit(ty,cand_minus,1)[0]:+.2f} deg/s")
    print(f"  データが示す正しい符号     : {good_sign:+.0f}")
    print(f"  このフライトで使われた符号 : {used_sign:+.0f}"
          f"   (一致率 +1:{agree_plus:.0f}%  -1:{agree_minus:.0f}%)")
    print(f"  --> {'一致。符号は正しい' if used_sign == good_sign else '不一致。符号を反転すべき'}")
    print()
    print("=== 収束 ===")
    n2 = ty < ty[0] + 2.0
    l2 = ty > ty[-1] - 2.0
    print(f"  |wallYaw| 最初の2s {np.abs(ww[n2]).mean():5.2f} deg -> 最後の2s {np.abs(ww[l2]).mean():5.2f} deg")
    print(f"  |wallYaw| 平均 {np.abs(ww).mean():5.2f} deg, 中央値 {np.median(np.abs(ww)):5.2f} deg, 最大 {np.abs(ww).max():5.2f} deg")
    print(f"  |wallYaw| <5deg の時間割合 {100*np.mean(np.abs(ww)<5):.0f}%,  <10deg {100*np.mean(np.abs(ww)<10):.0f}%")
    print(f"  yaw 追従誤差 |yaw-yawCmd| 平均 {np.abs(trk).mean():5.2f} deg, 最大 {np.abs(trk).max():5.2f} deg")
    print(f"  yawCmd {cc[0]:+.2f} -> {cc[-1]:+.2f} deg (平均 {np.polyfit(ty,cc,1)[0]:+.2f} deg/s)")
    print(f"  スルーレート飽和 {sat:.0f}%, |max| {np.abs(slew).max():.2f} deg/s")
    print()
    print("=== 推定の健全性 ===")
    print(f"  tofwall.conf 平均 {conf[m].mean():.3f} (min {conf[m].min():.3f})")
    print(f"  yawAlignOk    {100*ok[m if m.any() else slice(None)].mean():.1f}% (hover 全体では "
          f"{100*ok[pose['chamber.state'].to_numpy()==STATE_HOVER].mean():.1f}%)")
    for c in ("chamber.dFront", "chamber.dBack", "chamber.dLeft", "chamber.dRight"):
        d = pose[c].to_numpy()[pose["chamber.state"].to_numpy() == STATE_HOVER]
        print(f"  {c:18s} 無効 {100*np.mean(d <= 0):5.1f}%")


if __name__ == "__main__":
    main()
