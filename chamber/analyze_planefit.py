#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chamber.planeOk を深掘りする: 平面フィットが「なぜ」失敗したかまで切り分ける。

plot_chamber_planeok.py が成功率の集計だけなのに対し、こちらは
tof_odometry.c の平面フィットを Python で再現し、ログに入っている
生ゾーン距離 (vl53l8cx_sN.dM) から各フレーム・各センサーについて

    OK              : 平面が取れた
    FAIL_NVALID     : 有効ゾーンが MIN_INLIERS 未満 (測距そのものが取れていない)
    FAIL_INLIERS    : RANSAC の最大インライアが MIN_INLIERS 未満 (面が割れている)
    FAIL_PLANARITY  : 平面性 lambda3/sum > PLANARITY_EPS (床や角が混入)

を判定する。再現の妥当性は chamber.planeOk との一致率で検証する。

さらに
  - 検出壁数の時間変化と、高度・姿勢との関係
  - センサーごとの有効/無効の継続時間 (ちらつきの粒度)
  - 制御が止まっていた時間のうち、原因が「測距なし」か「不感帯」かの内訳
を出す。

使い方:
    python3 analyze_planefit.py log200_pose.csv
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

# ---- tof_odometry.c と同じ定数 ----
FOV_DEG, GRID, NZONE = 45.0, 4, 16
D_MIN_MM, D_MAX_MM = 30.0, 4000.0
RANSAC_ITERS, RANSAC_THRESH_MM, MIN_INLIERS = 60, 20.0, 10
PLANARITY_EPS = 0.006
NZ_MAX = 0.5   # tof_wall_angle.c: |n_z| がこれを超えると方位角には使わない

SENSOR_MOUNT_T = {0: (56.6, -9.5, 0.0), 1: (59.2, 0.0, 0.0), 2: (56.6, 9.5, 0.0),
                  5: (0.0, -21.7, 0.0), 6: (0.0, 21.7, 0.0), 7: (-59.2, 0.0, 0.0)}
SENSOR_MOUNT_YAW = {0: -22.5, 1: 0.0, 2: 22.5, 5: -90.0, 6: 90.0, 7: 180.0}

# planeOk のビット割り当て (app 側と同じ)
WALLS = [("front", 1, 0), ("back", 7, 1), ("left", 5, 2), ("right", 6, 3)]
AXES = [("x (front/back)", 0, 1), ("y (left/right)", 2, 3)]

STATE_COLORS = {5: "#cfe8ff", 6: "#d8f5d0", 7: "#ffe0cc"}
STATE_NAMES = {5: "TAKEOFF", 6: "HOVER", 7: "LAND"}
HOVER = 6

REASONS = ["OK", "FAIL_NVALID", "FAIL_INLIERS", "FAIL_PLANARITY"]


def resolve_csv(path):
    if os.path.isfile(path):
        return path
    for d in CSV_DIRS:
        c = os.path.join(d, os.path.basename(path))
        if os.path.isfile(c):
            return c
    raise SystemExit(f"CSVが見つかりません: {path}")


def rot_yaw(deg):
    y = np.deg2rad(deg)
    return np.array([[np.cos(y), -np.sin(y), 0.0],
                     [np.sin(y),  np.cos(y), 0.0],
                     [0.0,        0.0,       1.0]])


def build_rays(sensor):
    """tofOdometryInit() と同じ手順でゾーン方向の単位ベクトル(ボディ系)を作る。"""
    step = np.deg2rad(FOV_DEG) / GRID
    R = rot_yaw(SENSOR_MOUNT_YAW[sensor])   # pitch/roll は 0 のセンサーのみ扱う
    rays = np.zeros((NZONE, 3))
    for i in range(NZONE):
        row, col = i // GRID, i % GRID
        r = (GRID - 1) - row
        c = (GRID - 1) - col
        az = (c - (GRID - 1) / 2.0) * step
        el = (r - (GRID - 1) / 2.0) * step
        ray = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        rays[i] = R @ ray
    return rays


def pca_plane(P):
    """最小固有値方向を法線とする平面フィット。(normal, d, planarity) を返す。"""
    pbar = P.mean(axis=0)
    Q = P - pbar
    C = (Q.T @ Q) / len(P)
    w, V = np.linalg.eigh(C)          # 昇順
    n = V[:, 0]
    planarity = w[0] / max(w.sum(), 1e-12)
    d = float(n @ pbar)
    if d < 0:
        n, d = -n, -d
    return n, d, planarity


def fit_plane_ransac(P, rng):
    """fitPlaneRansac() の再現。(ok, reason, normal, d, planarity, n_inl)"""
    N = len(P)
    if N < MIN_INLIERS:
        return False, "FAIL_NVALID", None, np.nan, np.nan, 0

    best_inl, best_cnt = None, 0
    for _ in range(RANSAC_ITERS):
        i0, i1, i2 = rng.choice(N, 3, replace=False)
        e1, e2 = P[i1] - P[i0], P[i2] - P[i0]
        nrm = np.cross(e1, e2)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm /= nn
        inl = np.abs((P - P[i0]) @ nrm) < RANSAC_THRESH_MM
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
            if cnt == N:
                break

    if best_cnt < MIN_INLIERS:
        return False, "FAIL_INLIERS", None, np.nan, np.nan, best_cnt

    idx = best_inl
    for _ in range(2):                     # インライアで2回リファイン
        n, d, pl = pca_plane(P[idx])
        new = np.abs(P @ n - d) < RANSAC_THRESH_MM
        if int(new.sum()) < MIN_INLIERS:
            break
        idx = new
    n, d, pl = pca_plane(P[idx])
    if pl > PLANARITY_EPS:
        return False, "FAIL_PLANARITY", n, d, pl, int(idx.sum())
    return True, "OK", n, d, pl, int(idx.sum())


def runs(flags):
    """True/False の連続長を (値, 長さ) のリストで返す。"""
    out, cur, ln = [], flags[0], 1
    for v in flags[1:]:
        if v == cur:
            ln += 1
        else:
            out.append((cur, ln)); cur, ln = v, 1
    out.append((cur, ln))
    return out


def main():
    ap = argparse.ArgumentParser(description="平面フィット失敗要因の解析")
    ap.add_argument("pose_csv")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    p = pd.read_csv(resolve_csv(args.pose_csv))
    t = (p["timestamp"].to_numpy() - p["timestamp"].iloc[0]) / 1000.0
    st = p["chamber.state"].to_numpy()
    mask = p["chamber.planeOk"].to_numpy().astype(int)
    n = len(p)
    rng = np.random.default_rng(12345)

    rays = {s: build_rays(s) for _, s, _ in WALLS}
    mount = {s: np.array(SENSOR_MOUNT_T[s]) for _, s, _ in WALLS}

    # ---- 全フレーム・全センサーで平面フィットを再現 ----
    res_ok = {name: np.zeros(n, bool) for name, _, _ in WALLS}
    res_reason = {name: np.empty(n, object) for name, _, _ in WALLS}
    res_pl = {name: np.full(n, np.nan) for name, _, _ in WALLS}
    res_d = {name: np.full(n, np.nan) for name, _, _ in WALLS}
    res_nz = {name: np.full(n, np.nan) for name, _, _ in WALLS}
    res_nvalid = {name: np.zeros(n, int) for name, _, _ in WALLS}

    for name, sid, _ in WALLS:
        cols = [f"vl53l8cx_s{sid}.d{z}" for z in range(NZONE)]
        if not all(c in p.columns for c in cols):
            raise SystemExit(f"生ゾーン列が足りません: s{sid}")
        Z = p[cols].to_numpy(dtype=float)
        for k in range(n):
            d = Z[k]
            good = (d > D_MIN_MM) & (d < D_MAX_MM)
            res_nvalid[name][k] = int(good.sum())
            P = rays[sid][good] * d[good, None] + mount[sid]
            ok, reason, nrm, dd, pl, _ = fit_plane_ransac(P, rng)
            res_ok[name][k] = ok
            res_reason[name][k] = reason
            res_pl[name][k] = pl
            res_d[name][k] = dd / 1000.0 if np.isfinite(dd) else np.nan
            res_nz[name][k] = abs(nrm[2]) if nrm is not None else np.nan

    # ---- ファームの planeOk との一致率 ----
    agree = {}
    for name, _, b in WALLS:
        fw = ((mask >> b) & 1).astype(bool)
        agree[name] = 100.0 * np.mean(fw == res_ok[name])

    nseen_fw = sum(((mask >> b) & 1) for _, _, b in WALLS)
    hov = st == HOVER
    z = p["stateEstimate.z"].to_numpy()
    roll = p["stabilizer.roll"].to_numpy()
    pitch = p["stabilizer.pitch"].to_numpy()
    tilt = np.hypot(roll, pitch)

    # ================= 作図 =================
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
    fig.suptitle(f"plane-fit deep dive  ({os.path.basename(args.pose_csv)})",
                 fontsize=15, fontweight="bold")

    def shade(ax):
        edges = np.flatnonzero(np.diff(st)) + 1
        for s, e in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(st)]))):
            c = STATE_COLORS.get(int(st[s]))
            if c:
                ax.axvspan(t[s], t[e] if e < len(t) else t[-1], color=c, alpha=0.55, lw=0, zorder=0)
        ax.grid(alpha=0.3); ax.set_xlabel("time since takeoff [s]")

    # (0,0) 壁数と高度
    ax = axes[0, 0]; shade(ax)
    ax.plot(t, nseen_fw, lw=1.3, drawstyle="steps-post", color="tab:purple", label="walls fitted")
    ax.set_ylabel("walls fitted"); ax.set_ylim(-0.2, 4.2); ax.set_yticks(range(5))
    ax2 = ax.twinx()
    ax2.plot(t, z, color="tab:green", lw=1.2, alpha=0.8, label="estZ")
    ax2.set_ylabel("height [m]", color="tab:green")
    ax.set_title("Wall count vs height")
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower right")

    # (0,1) 壁数 vs 高度 の散布
    ax = axes[0, 1]
    sc = ax.scatter(z, nseen_fw + np.random.uniform(-0.15, 0.15, n), c=t, cmap="viridis", s=12)
    ax.set_xlabel("height estZ [m]"); ax.set_ylabel("walls fitted (jittered)")
    ax.set_yticks(range(5)); ax.grid(alpha=0.3)
    ax.set_title("Wall count vs height (scatter)")
    fig.colorbar(sc, ax=ax, label="t [s]")

    # (1,0) 失敗要因の内訳 (HOVER)
    ax = axes[1, 0]
    names = [w[0] for w in WALLS]
    bottom = np.zeros(len(WALLS))
    colors = {"OK": "tab:green", "FAIL_NVALID": "tab:red",
              "FAIL_INLIERS": "tab:orange", "FAIL_PLANARITY": "tab:blue"}
    for r in REASONS:
        vals = [100 * np.mean(res_reason[nm][hov] == r) for nm in names]
        ax.bar(names, vals, bottom=bottom, label=r, color=colors[r])
        for i, v in enumerate(vals):
            if v > 4:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=9)
        bottom += np.array(vals)
    ax.set_ylabel("time share [%]"); ax.set_ylim(0, 100)
    ax.set_title("Why the fit failed (HOVER, reproduced offline)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # (1,1) 平面性スコア
    ax = axes[1, 1]; shade(ax)
    for nm, _, _ in WALLS:
        ax.plot(t, res_pl[nm], lw=1.0, label=nm)
    ax.axhline(PLANARITY_EPS, color="k", ls="--", lw=1.2, label=f"PLANARITY_EPS={PLANARITY_EPS}")
    ax.set_yscale("log"); ax.set_ylabel("planarity  $\\lambda_3/\\Sigma\\lambda$")
    ax.set_title("Planarity score (above the line = rejected)")
    ax.legend(fontsize=8, ncol=2)

    # (2,0) 有効ゾーン数
    ax = axes[2, 0]; shade(ax)
    for nm, _, _ in WALLS:
        ax.plot(t, res_nvalid[nm], lw=1.0, drawstyle="steps-post", label=nm)
    ax.axhline(MIN_INLIERS, color="k", ls="--", lw=1.2, label=f"MIN_INLIERS={MIN_INLIERS}")
    ax.set_ylabel("valid zones (of 16)"); ax.set_ylim(-0.5, 16.5)
    ax.set_title("Zones with a usable range")
    ax.legend(fontsize=8, ncol=3)

    # (2,1) 継続時間 (ちらつきの粒度)
    ax = axes[2, 1]
    dt = float(np.median(np.diff(t)))
    for i, (nm, _, b) in enumerate(WALLS):
        fw = (((mask >> b) & 1) == 1)[hov]
        rl = runs(fw)
        on = [ln * dt for v, ln in rl if v]
        off = [ln * dt for v, ln in rl if not v]
        ax.scatter(np.full(len(on), i - 0.15), on, s=18, alpha=0.6, color="tab:green")
        ax.scatter(np.full(len(off), i + 0.15), off, s=18, alpha=0.6, color="tab:red")
    ax.set_xticks(range(len(WALLS))); ax.set_xticklabels(names)
    ax.set_yscale("log"); ax.set_ylabel("run length [s]")
    ax.set_title("Dwell time: valid (green, left) / invalid (red, right), HOVER")
    ax.grid(alpha=0.3)

    # (3,0) 姿勢と壁数
    ax = axes[3, 0]; shade(ax)
    ax.plot(t, tilt, lw=1.1, color="tab:brown", label="|tilt| = hypot(roll,pitch)")
    ax.set_ylabel("tilt [deg]")
    ax2 = ax.twinx()
    ax2.plot(t, nseen_fw, lw=1.0, color="tab:purple", alpha=0.6, drawstyle="steps-post")
    ax2.set_ylabel("walls", color="tab:purple"); ax2.set_ylim(-0.2, 4.2)
    ax.set_title("Attitude vs wall count")
    ax.legend(fontsize=8)

    # (3,1) 制御が止まった理由
    ax = axes[3, 1]
    labels, vals = [], []
    for aname, b0, b1 in AXES:
        both = ((((mask >> b0) & 1) & ((mask >> b1) & 1)) == 1)
        cmdcol = "chamber.vxCmd" if b0 == 0 else "chamber.vyCmd"
        holdcol = "chamber.holdX" if b0 == 0 else "chamber.holdY"
        cmd = p[cmdcol].to_numpy()[hov]
        hold = p[holdcol].to_numpy()[hov].astype(bool)
        bb = both[hov]
        labels += [f"{aname}\ndriven", f"{aname}\ndeadband", f"{aname}\nno data"]
        vals += [100 * np.mean(bb & ~hold & (np.abs(cmd) > 1e-6)),
                 100 * np.mean(bb & hold),
                 100 * np.mean(~bb)]
    ax.bar(labels, vals, color=["tab:green", "tab:blue", "tab:red"] * 2)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("time share [%]"); ax.set_ylim(0, 108)
    ax.set_title("Why each axis was / was not being driven (HOVER)")
    ax.grid(alpha=0.3, axis="y"); ax.tick_params(axis="x", labelsize=8)

    out = args.save or (os.path.splitext(os.path.basename(args.pose_csv))[0].replace("_pose", "") + "_planefit.png")
    if not os.path.dirname(out):
        out = os.path.join(FIGS_DIR, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"保存しました: {out}\n")

    # ================= 数値 =================
    print("=== オフライン再現とファームの planeOk の一致率 ===")
    for nm, _, _ in WALLS:
        print(f"  {nm:6s}: {agree[nm]:5.1f} %")

    print("\n=== 失敗要因の内訳 (HOVER) ===")
    print(f"  {'wall':6s} {'OK':>7s} {'NVALID':>8s} {'INLIERS':>8s} {'PLANARITY':>10s}  {'有効ゾーン中央値':>10s}")
    for nm, _, _ in WALLS:
        r = res_reason[nm][hov]
        print(f"  {nm:6s} " + "".join(f"{100*np.mean(r==x):7.1f}%" if x=='OK' else f"{100*np.mean(r==x):8.1f}" for x in REASONS)
              + f"   {np.median(res_nvalid[nm][hov]):.0f}/16")

    print("\n=== 平面性スコア (HOVER, フィット到達分のみ) ===")
    for nm, _, _ in WALLS:
        v = res_pl[nm][hov]
        v = v[np.isfinite(v)]
        if len(v):
            print(f"  {nm:6s}: median {np.median(v):.5f}  p90 {np.percentile(v,90):.5f}  "
                  f"(閾値 {PLANARITY_EPS})  閾値超え {100*np.mean(v>PLANARITY_EPS):.0f}%")

    print("\n=== 継続時間 (HOVER) ===")
    for nm, _, b in WALLS:
        fw = (((mask >> b) & 1) == 1)[hov]
        rl = runs(fw)
        on = np.array([ln * dt for v, ln in rl if v])
        off = np.array([ln * dt for v, ln in rl if not v])
        print(f"  {nm:6s}: 有効 中央値 {np.median(on) if len(on) else 0:.2f} s / 最長 {on.max() if len(on) else 0:.2f} s,"
              f"  無効 中央値 {np.median(off) if len(off) else 0:.2f} s / 最長 {off.max() if len(off) else 0:.2f} s,"
              f"  切替 {len(rl)-1} 回")

    print("\n=== 高度との関係 ===")
    for lo, hi in [(0.0, 0.10), (0.10, 0.18), (0.18, 0.24), (0.24, 1.0)]:
        m = (z >= lo) & (z < hi)
        if m.sum() > 3:
            print(f"  z {lo:.2f}-{hi:.2f} m (n={m.sum():3d}): 平均壁数 {nseen_fw[m].mean():.2f}, "
                  + ", ".join(f"{nm} {100*np.mean(((mask[m]>>b)&1)==1):3.0f}%" for nm, _, b in WALLS))

    print("\n=== 姿勢との関係 (HOVER) ===")
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 90)]:
        m = hov & (tilt >= lo) & (tilt < hi)
        if m.sum() > 3:
            print(f"  tilt {lo:2d}-{hi:2d} deg (n={m.sum():3d}): 平均壁数 {nseen_fw[m].mean():.2f}")


if __name__ == "__main__":
    main()
