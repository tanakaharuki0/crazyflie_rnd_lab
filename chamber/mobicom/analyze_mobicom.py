#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobiCom experiment analysis: ex1..ex4 chamber flights.

Produces publication-ready figures (English labels, Times New Roman) in figs/
and prints the numbers used in kousatu.md.

Data sources per flight:
  <log>_<ex>.csv       fixedFrequency stream: chamber.* control state,
                       tofwall.*, tofodo.*  (~25 Hz)
  <log>_<ex>_pose.csv  pose stream: stateEstimate.*, stabilizer.*, IMU,
                       raw per-zone ToF (~30 Hz)

Manually scored outcomes (collisions / landing offset / landing yaw / crash)
are parsed straight out of mobicom_experiment.md so the two never diverge.

Usage:
    python3 analyze_mobicom.py
"""
import os
import re
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(HERE, "mobicom_csv")
FIGS_DIR = os.path.join(HERE, "figs")
MD = os.path.join(HERE, "mobicom_experiment.md")
os.makedirs(FIGS_DIR, exist_ok=True)

# experiment id -> (csv tag, first log number, short label, long label)
EXPS = [
    ("ex1", "ex1_flow_flow_flow", 360, "E1: flow/flow/flow",
     "Flow-deck position hold in all phases"),
    ("ex2", "ex2_flow_P_flow", 340, "E2: flow/P/flow",
     "Flow hold on takeoff and landing, wall P control in hover"),
    ("ex3", "ex3_P_P_P", 320, "E3: P/P/P",
     "Wall P control in all phases (zone-average distance)"),
    ("ex4", "ex4_P_hybridP_P", 300, "E4: P/hybrid/P",
     "Wall P control in all phases, plane-based hybrid + yaw alignment in hover"),
]
COLORS = {"ex1": "#8c8c8c", "ex2": "#4c72b0", "ex3": "#dd8452", "ex4": "#55a868"}

HALF_BOX = 0.25      # m, chamber half width
STATE_TAKEOFF, STATE_HOVER, STATE_LAND = 5, 6, 7


# ---------------------------------------------------------------- md parsing
def parse_md():
    """mobicom_experiment.md の手動スコアを {ex: DataFrame} で返す。"""
    txt = open(MD, encoding="utf-8").read()
    out = {}
    for exid, tag, first, _, _ in EXPS:
        m = re.search(rf"^# {re.escape(tag)}.*?$(.*?)(?=^# |\Z)", txt,
                      re.S | re.M)
        if not m:
            raise SystemExit(f"section for {tag} not found in md")
        rows = []
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            body = line[2:].strip()
            no_log = "logナシ" in body
            crash = "crash" in body
            coll = re.search(r"(\d+)\s*回", body)
            off = re.search(r"(\d+)\s*cm", body)
            # trailing field: yaw angle, 'a' (hit a wall on landing) or empty
            tail = body.split(",")[-1].strip()
            hit_wall = tail == "a"
            yaw = None
            ym = re.search(r"(-?\d+)\s*$", tail)
            if ym and not hit_wall and not crash:
                yaw = float(ym.group(1))
            rows.append(dict(
                collisions=float(coll.group(1)) if coll else np.nan,
                offset_cm=float(off.group(1)) if off else np.nan,
                land_yaw_deg=yaw,
                wall_hit_on_landing=hit_wall,
                crash=crash,
                no_log=no_log,
            ))
        df = pd.DataFrame(rows)
        df.insert(0, "log", [first + i for i in range(len(df))])
        out[exid] = df
    return out


# ---------------------------------------------------------------- log loading
def load_flights(tag):
    """そのグループの (log番号, fixedFrequency df, pose df) を返す。"""
    out = []
    for f in sorted(glob.glob(os.path.join(CSV_DIR, f"*_{tag}.csv"))):
        base = os.path.basename(f)
        num = int(re.match(r"log(\d+)_", base).group(1))
        pose_f = f.replace(".csv", "_pose.csv")
        pose = pd.read_csv(pose_f) if os.path.isfile(pose_f) else None
        out.append((num, pd.read_csv(f), pose))
    return out


def hover_slice(df):
    if "chamber.state" not in df.columns:
        return None
    m = df["chamber.state"].to_numpy() == STATE_HOVER
    return df[m] if m.sum() > 5 else None


def wall_xy(df):
    """壁距離から求めた中心基準の位置 [m]。無効サンプルは NaN。

    dFront = h - x, dBack = h + x  =>  x = (dBack - dFront)/2
    dLeft  = h + y, dRight = h - y =>  y = (dLeft - dRight)/2
    """
    f = df["chamber.dFront"].to_numpy(float)
    b = df["chamber.dBack"].to_numpy(float)
    l = df["chamber.dLeft"].to_numpy(float)
    r = df["chamber.dRight"].to_numpy(float)
    okx = (f > 0) & (b > 0)
    oky = (l > 0) & (r > 0)
    x = np.where(okx, (b - f) / 2.0, np.nan)
    y = np.where(oky, (l - r) / 2.0, np.nan)
    return x, y, okx, oky


def min_clearance(df):
    d = np.vstack([df[f"chamber.d{k}"].to_numpy(float)
                   for k in ("Front", "Back", "Left", "Right")])
    d = np.where(d > 0, d, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmin(d, axis=0)


# ---------------------------------------------------------------- collect
def collect():
    scores = parse_md()
    per_flight, per_sample = [], {}
    for exid, tag, _, short, _ in EXPS:
        sc = scores[exid].set_index("log")
        samples = []
        # 母集団は md の全エントリ (ログが残らなかった飛行も目視スコアはある)
        metrics = {}
        for num, ff, pose in load_flights(tag):
            row = {}
            h = hover_slice(ff)
            if h is not None and len(h) > 5:
                x, y, okx, oky = wall_xy(h)
                e = np.hypot(x, y)
                clr = min_clearance(h)
                row["hover_s"] = (h["timestamp"].iloc[-1] - h["timestamp"].iloc[0]) / 1000.0
                row["err_rms_cm"] = 100 * np.sqrt(np.nanmean(e ** 2)) if np.isfinite(e).any() else np.nan
                row["err_med_cm"] = 100 * np.nanmedian(e) if np.isfinite(e).any() else np.nan
                row["err_max_cm"] = 100 * np.nanmax(e) if np.isfinite(e).any() else np.nan
                row["clr_min_cm"] = 100 * np.nanmin(clr) if np.isfinite(clr).any() else np.nan
                row["t_near5_pct"] = 100 * np.nanmean(clr < 0.05)
                row["valid_x_pct"] = 100 * okx.mean()
                row["valid_y_pct"] = 100 * oky.mean()
                # tofwall.usedMask: どのセンサーで平面が取れたか
                # bit0=s0 bit1=s1 bit2=s2 bit3=s5 bit4=s6 bit5=s7
                if "tofwall.usedMask" in h:
                    um = h["tofwall.usedMask"].to_numpy(int)
                    for lab, b in (("front", 1), ("back", 5), ("left", 3), ("right", 4)):
                        row[f"plane_{lab}_pct"] = 100 * np.mean((um >> b) & 1)
                # ヨー誤差は tofwall.angleFilt を使う。これはアプリが yaw 補正を
                # 使うかどうかに関係なく常に計算されているので、4条件を同じ物差しで
                # 比較できる (chamber.wallYaw は補正が無効だと 0 で埋まる)。
                if "tofwall.angleFilt" in h and "tofwall.conf" in h:
                    w = h["tofwall.angleFilt"].to_numpy(float)
                    ok = np.isfinite(w) & (h["tofwall.conf"].to_numpy(float) >= 0.70)
                    if ok.sum() > 5:
                        row["wallyaw_med"] = float(np.median(np.abs(w[ok])))
                        row["wallyaw_rms"] = float(np.sqrt(np.mean(w[ok] ** 2)))
                        row["wallyaw_p90"] = float(np.percentile(np.abs(w[ok]), 90))
                if "chamber.srcX" in h:
                    row["src_plane_pct"] = 100 * np.mean(
                        (h["chamber.srcX"].to_numpy() == 1) | (h["chamber.srcY"].to_numpy() == 1))
                    row["srcx_plane_pct"] = 100 * np.mean(h["chamber.srcX"].to_numpy() == 1)
                    row["srcy_plane_pct"] = 100 * np.mean(h["chamber.srcY"].to_numpy() == 1)
                if "tofwall.nPlanes" in h:
                    row["nplanes_mean"] = float(h["tofwall.nPlanes"].mean())
                    row["conf_mean"] = float(h["tofwall.conf"].mean())
                t = (h["timestamp"].to_numpy() - h["timestamp"].iloc[0]) / 1000.0
                samples.append(dict(t=t, x=x, y=y, e=e, clr=clr, log=num))
            metrics[num] = row
        for num, srow in sc.iterrows():
            row = dict(ex=exid, log=num)
            row.update(srow.to_dict())
            row.update(metrics.get(num, {}))
            per_flight.append(row)
        per_sample[exid] = samples
    return pd.DataFrame(per_flight), per_sample


def save(fig, name):
    p = os.path.join(FIGS_DIR, name)
    fig.savefig(p, bbox_inches="tight")
    pdf_p = os.path.splitext(p)[0] + ".pdf"
    fig.savefig(pdf_p, bbox_inches="tight")
    plt.close(fig)
    print("  saved", p, "and", pdf_p)


# ---------------------------------------------------------------- figures
def fig_outcomes(df):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    labels = [s for _, _, _, s, _ in EXPS]
    ids = [e for e, _, _, _, _ in EXPS]
    cs = [COLORS[e] for e in ids]

    ax = axes[0]
    v = [100 * df[df.ex == e]["crash"].mean() for e in ids]
    ax.bar(range(4), v, color=cs)
    ax.set_ylabel("Crash rate [%]"); ax.set_title("(a) Crash rate")

    ax = axes[1]
    v = [100 * df[(df.ex == e) & (~df.crash)]["wall_hit_on_landing"].mean() for e in ids]
    ax.bar(range(4), v, color=cs)
    ax.set_ylabel("Wall contact at landing [%]"); ax.set_title("(b) Wall contact on landing")

    ax = axes[2]
    data = [df[(df.ex == e) & (~df.crash)]["collisions"].dropna().to_numpy() for e in ids]
    bp = ax.boxplot(data, positions=range(4), widths=0.6, patch_artist=True, showmeans=True)
    for p, c in zip(bp["boxes"], cs):
        p.set_facecolor(c); p.set_alpha(0.65)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.12, .12, len(d)), d,
                   s=14, color="k", alpha=0.55, zorder=3)
    ax.set_ylabel("Wall collisions per flight"); ax.set_title("(c) Collisions during flight")

    ax = axes[3]
    data = [df[(df.ex == e) & (~df.crash)]["offset_cm"].dropna().to_numpy() for e in ids]
    bp = ax.boxplot(data, positions=range(4), widths=0.6, patch_artist=True, showmeans=True)
    for p, c in zip(bp["boxes"], cs):
        p.set_facecolor(c); p.set_alpha(0.65)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.12, .12, len(d)), d,
                   s=14, color="k", alpha=0.55, zorder=3)
    ax.axhline(25, ls="--", lw=1, color="firebrick")
    ax.text(3.45, 25.4, "wall", color="firebrick", fontsize=8, ha="right")
    ax.set_ylabel("Landing offset from centre [cm]"); ax.set_title("(d) Landing accuracy")

    for ax in axes:
        ax.set_xticks(range(4))
        ax.set_xticklabels([l.split(":")[0] for l in labels])
        ax.set_xlim(-0.6, 3.6)
    fig.legend(handles=[plt.Line2D([], [], color=COLORS[e], lw=6, label=s)
                        for (e, _, _, s, _) in EXPS],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), frameon=False)
    save(fig, "fig1_outcomes.png")


def fig_hover_error(df, samples):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ids = [e for e, _, _, _, _ in EXPS]
    cs = [COLORS[e] for e in ids]

    ax = axes[0]
    data = [df[(df.ex == e) & (~df.crash)]["err_rms_cm"].dropna().to_numpy() for e in ids]
    bp = ax.boxplot(data, positions=range(4), widths=0.6, patch_artist=True, showmeans=True)
    for p, c in zip(bp["boxes"], cs):
        p.set_facecolor(c); p.set_alpha(0.65)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.12, .12, len(d)), d, s=14,
                   color="k", alpha=0.55, zorder=3)
    ax.set_ylabel(r"RMS centring error $|e|$ [cm]")
    ax.set_title("(a) Hover centring error")

    ax = axes[1]
    data = [df[(df.ex == e) & (~df.crash)]["clr_min_cm"].dropna().to_numpy() for e in ids]
    bp = ax.boxplot(data, positions=range(4), widths=0.6, patch_artist=True, showmeans=True)
    for p, c in zip(bp["boxes"], cs):
        p.set_facecolor(c); p.set_alpha(0.65)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.12, .12, len(d)), d, s=14,
                   color="k", alpha=0.55, zorder=3)
    ax.axhline(5, ls="--", lw=1, color="firebrick")
    ax.set_ylabel("Minimum wall clearance [cm]")
    ax.set_title("(b) Closest approach during hover")

    ax = axes[2]
    for e in ids:
        S = samples[e]
        if not S:
            continue
        grid = np.linspace(0, 9.5, 60)
        M = []
        for s in S:
            M.append(np.interp(grid, s["t"], 100 * s["e"], left=np.nan, right=np.nan))
        M = np.array(M, dtype=float)
        med = np.nanmedian(M, axis=0)
        ax.plot(grid, med, color=COLORS[e], lw=1.8, label=e.upper())
        ax.fill_between(grid, np.nanpercentile(M, 25, axis=0),
                        np.nanpercentile(M, 75, axis=0), color=COLORS[e], alpha=0.15)
    ax.set_xlabel("Time since hover start [s]")
    ax.set_ylabel(r"$|e|$ [cm]")
    ax.set_title("(c) Centring error over hover (median, IQR)")
    ax.legend(ncol=2)

    for ax in axes[:2]:
        ax.set_xticks(range(4)); ax.set_xticklabels([e.upper() for e in ids])
        ax.set_xlim(-0.6, 3.6)
    save(fig, "fig2_hover_error.png")


def fig_traj(samples):
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharex=True, sharey=True)
    for ax, (exid, _, _, short, _) in zip(axes, EXPS):
        for s in samples[exid]:
            ax.plot(100 * s["x"], 100 * s["y"], lw=0.8, alpha=0.55, color=COLORS[exid])
        ax.add_patch(plt.Rectangle((-25, -25), 50, 50, fill=False, ec="k", lw=1.3))
        ax.plot(0, 0, "k+", ms=9)
        ax.set_aspect("equal")
        ax.set_xlim(-28, 28); ax.set_ylim(-28, 28)
        ax.set_title(short, fontsize=11)
        ax.set_xlabel("x (front +) [cm]")
    axes[0].set_ylabel("y (left +) [cm]")
    fig.suptitle("Hover trajectories in the chamber frame, reconstructed from wall ranges", y=1.03)
    save(fig, "fig3_trajectories.png")


def fig_sensing(df, samples):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ids = [e for e, _, _, _, _ in EXPS]
    cs = [COLORS[e] for e in ids]

    ax = axes[0]
    walls = ["front", "back", "left", "right"]
    w = 0.2
    for j, e in enumerate(ids):
        d = df[(df.ex == e) & (~df.crash)]
        v = [np.nanmean(d[f"plane_{k}_pct"]) if f"plane_{k}_pct" in d else np.nan for k in walls]
        ax.bar(np.arange(4) + (j - 1.5) * w, v, w, label=e.upper(), color=COLORS[e])
    ax.set_xticks(range(4)); ax.set_xticklabels(walls)
    ax.set_ylabel("Plane fitted [% of hover]")
    ax.set_title("(a) Per-wall plane-fit success"); ax.legend(ncol=2, fontsize=8)
    ax.set_ylim(0, 108)

    ax = axes[1]
    data = [df[(df.ex == e) & (~df.crash)]["wallyaw_med"].dropna().to_numpy() for e in ids]
    bp = ax.boxplot(data, positions=range(4), widths=0.6, patch_artist=True, showmeans=True)
    for p, c in zip(bp["boxes"], cs):
        p.set_facecolor(c); p.set_alpha(0.65)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.12, .12, len(d)), d, s=14,
                   color="k", alpha=0.55, zorder=3)
    ax.set_ylabel(r"Median $|\psi_\mathrm{wall}|$ during hover [deg]")
    ax.set_title("(b) Yaw misalignment to the walls")

    ax = axes[2]
    e4 = df[(df.ex == "ex4") & (~df.crash)]
    vals = [np.nanmean(e4["srcx_plane_pct"]), np.nanmean(e4["srcy_plane_pct"])]
    ax.bar([0, 1], vals, 0.5, color=["#4c72b0", "#dd8452"])
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["x (front/back)", "y (left/right)"])
    ax.set_ylabel("Plane-based distance used [% of hover]")
    ax.set_title("(c) E4: how often the plane fit was usable")
    ax.set_ylim(0, 108)

    ax = axes[1]
    ax.set_xticks(range(4)); ax.set_xticklabels([e.upper() for e in ids])
    ax.set_xlim(-0.6, 3.6)
    save(fig, "fig4_sensing.png")


# ---------------------------------------------------------------- main
def main():
    df, samples = collect()
    df.to_csv(os.path.join(HERE, "mobicom_summary.csv"), index=False)
    print("figures:")
    fig_outcomes(df)
    fig_hover_error(df, samples)
    fig_traj(samples)
    fig_sensing(df, samples)

    print("\n================ SUMMARY ================")
    for exid, _, _, short, _ in EXPS:
        d = df[df.ex == exid]
        ok = d[~d.crash]
        n, nc = len(d), int(d.crash.sum())
        print(f"\n{short}   n={n}, crash={nc} ({100*nc/n:.0f}%)")
        if len(ok):
            print(f"  collisions/flight   mean {ok.collisions.mean():.2f}  median {ok.collisions.median():.1f}"
                  f"  zero-collision {100*np.mean(ok.collisions==0):.0f}%")
            print(f"  landing offset [cm] mean {ok.offset_cm.mean():.1f}  median {ok.offset_cm.median():.1f}"
                  f"  wall-contact {100*ok.wall_hit_on_landing.mean():.0f}%")
            y = ok.land_yaw_deg.dropna()
            if len(y):
                print(f"  landing yaw [deg]   mean|.| {y.abs().mean():.1f}  median|.| {y.abs().median():.1f}  n={len(y)}")
            for k, lab in (("err_rms_cm", "hover |e| RMS [cm]"), ("clr_min_cm", "min clearance [cm]"),
                           ("t_near5_pct", "time <5cm [%]"), ("valid_x_pct", "x measurable [%]"),
                           ("valid_y_pct", "y measurable [%]"), ("wallyaw_med", "|psi_wall| med [deg]"),
                           ("nplanes_mean", "planes fitted (of 6)"), ("conf_mean", "tofwall.conf"),
                           ("srcx_plane_pct", "x on plane [%]"), ("srcy_plane_pct", "y on plane [%]"),
                           ("plane_front_pct", "plane front [%]"), ("plane_back_pct", "plane back [%]"),
                           ("plane_left_pct", "plane left [%]"), ("plane_right_pct", "plane right [%]")):
                if k in ok and ok[k].notna().any():
                    print(f"  {lab:22s} mean {ok[k].mean():7.2f}  median {ok[k].median():7.2f}")


if __name__ == "__main__":
    main()
