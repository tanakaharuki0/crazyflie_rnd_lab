#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid 実装 (log200-208) と、平面フィットのみの takeoff_Pcontrol 実装
(log180-192)、および MobiCom 時の素の測距実装 (k=0.3, log100-115) を
同じ指標で突き合わせる。

指標はすべて HOVER 区間 (chamber.state == 6) のみで計算し、
ゲイン非依存の物理量
    ex = (dFront - dBack)/2,  ey = (dLeft - dRight)/2   [m]
    r  = hypot(ex, ey)    (チャンバ中心からの水平距離)
を使う (kousatsu_mobicom_chamber.md と同じ定義)。

出力:
    csv/summary_hybrid_compare.csv   ログ別の生の統計
    figs/hybrid_compare.png          条件比較の図

使い方:
    python3 compare_hybrid.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
FIGS_DIR = os.path.join(SCRIPT_DIR, "figs")

STATE_HOVER = 6
ACC_TH = 0.2      # detect_impact.py と同じ衝突しきい値 [g]
IMPACT_RESET_S = 0.3

# (条件名, csv のグロブ, ログ番号の範囲)  範囲で他実験の混入を除く
CONDS = [
    ("raw (k=0.3, MobiCom)", "log*_k_0.3_pose.csv", (100, 116)),
    ("planefit (takeoff_Pcontrol)", "log*_k_0.3_takeoff_Pcontrol_pose.csv", (180, 192)),
    ("hybrid", "log*_k_0.3_hybrid_pose.csv", (200, 208)),
]
COLORS = {"raw (k=0.3, MobiCom)": "#7f7f7f",
          "planefit (takeoff_Pcontrol)": "#1f77b4",
          "hybrid": "#d62728"}


def lognum(path):
    b = os.path.basename(path)
    return int(b[3:b.index("_")])


def hover_slice(p):
    return p[p["chamber.state"] == STATE_HOVER].reset_index(drop=True)


def count_impacts(h):
    """水平加速度スパイクの数 (detect_impact.py の既定と同じ判定)。"""
    t = h["timestamp"].to_numpy() / 1000.0
    a = np.hypot(h["acc.x"].to_numpy(), h["acc.y"].to_numpy())
    n, last = 0, -1e9
    for ti, ai in zip(t, a):
        if ai > ACC_TH and ti - last > IMPACT_RESET_S:
            n += 1
            last = ti
    return n, float(a.max()) if len(a) else np.nan


def analyze(path):
    p = pd.read_csv(path)
    h = hover_slice(p)
    row = {"log": lognum(path), "file": os.path.basename(path),
           "n_hover": len(h)}
    if len(h) < 2:
        row["dur"] = 0.0
        row["crash"] = True
        return row

    t = (h["timestamp"].to_numpy() - h["timestamp"].iloc[0]) / 1000.0
    row["dur"] = float(t[-1])

    ex = (h["chamber.dFront"] - h["chamber.dBack"]).to_numpy() / 2.0
    ey = (h["chamber.dLeft"] - h["chamber.dRight"]).to_numpy() / 2.0
    r = np.hypot(ex, ey)
    row.update(
        ex_mean=float(ex.mean()), ey_mean=float(ey.mean()),
        abs_ex=float(np.abs(ex).mean()), abs_ey=float(np.abs(ey).mean()),
        r_mean=float(r.mean()), r_rms=float(np.sqrt((r**2).mean())),
        r_max=float(r.max()),
        r_first2=float(r[t <= 2.0].mean()),
        r_last2=float(r[t >= t[-1] - 2.0].mean()),
        wall_min=float(np.nanmin(h[["chamber.dFront", "chamber.dBack",
                                    "chamber.dLeft", "chamber.dRight"]].to_numpy())),
        roll_max=float(np.abs(p["stabilizer.roll"]).max()),
        pitch_max=float(np.abs(p["stabilizer.pitch"]).max()),
        yaw_drift=float(abs(h["stabilizer.yaw"].iloc[-1] - h["stabilizer.yaw"].iloc[0])),
        yaw_span=float(h["stabilizer.yaw"].max() - h["stabilizer.yaw"].min()),
        z_mean=float(h["stateEstimate.z"].mean()), z_sd=float(h["stateEstimate.z"].std()),
        hold_x=float(h["chamber.holdX"].mean()), hold_y=float(h["chamber.holdY"].mean()),
        yaw_ok=float(h["chamber.yawAlignOk"].mean()),
        vx_rms=float(np.sqrt((h["chamber.vxCmd"].to_numpy()**2).mean())),
        vy_rms=float(np.sqrt((h["chamber.vyCmd"].to_numpy()**2).mean())),
    )
    # 実際に壁距離が動いたサンプル比 (= 距離推定が更新できていた割合の代理)
    for ax_, a, b in (("x", "chamber.dFront", "chamber.dBack"),
                      ("y", "chamber.dLeft", "chamber.dRight")):
        d = np.abs(np.diff(h[a].to_numpy())) + np.abs(np.diff(h[b].to_numpy()))
        row[f"upd_{ax_}"] = float((d > 1e-9).mean())

    if "chamber.srcX" in h.columns:
        row["srcX"] = float(h["chamber.srcX"].mean())
        row["srcY"] = float(h["chamber.srcY"].mean())
    if "chamber.planeOk" in h.columns:
        m = h["chamber.planeOk"].to_numpy().astype(int)
        for name, bit in (("front", 0), ("back", 1), ("left", 2), ("right", 3)):
            row[f"plane_{name}"] = float(((m >> bit) & 1).mean())
        row["plane_x"] = float((((m >> 0) & 1) & ((m >> 1) & 1)).mean())
        row["plane_y"] = float((((m >> 2) & 1) & ((m >> 3) & 1)).mean())
    if "tofwall.conf" in h.columns:
        row["wall_conf"] = float(h["tofwall.conf"].mean())

    # 壁距離の推定ノイズ: 更新のあったサンプルの増分の標準偏差 / sqrt(2)
    # (指令速度は最大 0.05 m/s、更新間隔 ~0.1 s なので実運動は <5 mm、
    #  増分のばらつきはほぼ推定ノイズと見なせる)
    nz = []
    for w in ("chamber.dFront", "chamber.dBack", "chamber.dLeft", "chamber.dRight"):
        d = np.diff(h[w].to_numpy())
        d = d[np.abs(d) > 1e-9]
        if len(d) > 3:
            nz.append(np.std(d) / np.sqrt(2))
    row["noise"] = float(np.mean(nz)) if nz else np.nan
    v = h[["chamber.dFront", "chamber.dBack", "chamber.dLeft", "chamber.dRight"]].to_numpy()
    row["implausible"] = float(((v <= 0.0) | (v > 0.60)).mean())

    # 収束時間: r < 0.05 m を初めて満たし、その後 2 秒間 0.08 m 未満を保つ時刻
    row["t_conv"] = np.nan
    for i in range(len(t)):
        if r[i] < 0.05 and (r[(t >= t[i]) & (t <= t[i] + 2.0)] < 0.08).all():
            row["t_conv"] = float(t[i])
            break
    row["frac_in5"] = float((r < 0.05).mean())

    row["impacts"], row["ah_max"] = count_impacts(h)
    # kousatsu_mobicom_chamber.md と同じ墜落判定
    row["crash"] = bool(len(h) < 50 or row["dur"] < 9.0
                        or row["roll_max"] > 40 or row["pitch_max"] > 40)
    return row


def main():
    rows = []
    for cond, pattern, (lo, hi) in CONDS:
        for f in sorted(glob.glob(os.path.join(CSV_DIR, pattern))):
            n = lognum(f)
            if not (lo <= n <= hi):
                continue
            r = analyze(f)
            r["cond"] = cond
            rows.append(r)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(CSV_DIR, "summary_hybrid_compare.csv")
    df.to_csv(out_csv, index=False)

    ok = df[~df["crash"]]
    agg = ok.groupby("cond").agg(
        n=("log", "count"), abs_ex=("abs_ex", "mean"), abs_ey=("abs_ey", "mean"),
        r_mean=("r_mean", "mean"), r_rms=("r_rms", "mean"),
        first2=("r_first2", "mean"), last2=("r_last2", "mean"),
        wall_min=("wall_min", "mean"), roll=("roll_max", "mean"),
        pitch=("pitch_max", "mean"), yaw_drift=("yaw_drift", "mean"),
        yaw_span=("yaw_span", "mean"), impacts=("impacts", "mean"),
        hold_x=("hold_x", "mean"), hold_y=("hold_y", "mean"),
        upd_x=("upd_x", "mean"), upd_y=("upd_y", "mean"),
        z=("z_mean", "mean"), z_sd=("z_sd", "mean"),
        noise=("noise", "mean"), implausible=("implausible", "mean"),
        t_conv=("t_conv", "mean"), frac_in5=("frac_in5", "mean"),
    )
    crash = df.groupby("cond").agg(logs=("log", "count"), crashes=("crash", "sum"))
    print("=== 墜落判定 ===")
    print(crash.assign(rate=lambda d: (d.crashes / d.logs).round(2)))
    print("\n=== HOVER統計 (墜落を除く) ===")
    print(agg.round(3).to_string())
    if "srcX" in ok.columns:
        h = ok[ok["cond"] == "hybrid"]
        print("\n=== hybrid: 距離ソースと平面フィット ===")
        print(h[["log", "srcX", "srcY", "plane_x", "plane_y",
                 "plane_front", "plane_back", "plane_left", "plane_right",
                 "r_mean", "impacts"]].round(3).to_string(index=False))

    # ---- 図 ---- (matplotlib の既定フォントに日本語が無いのでラベルは英語)
    conds = [c for c, _, _ in CONDS if c in ok["cond"].values]
    short = {c: c.split(" ")[0] for c in conds}
    x = np.arange(len(conds))
    w = 0.35
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    for cond in conds:
        for f in ok.loc[ok["cond"] == cond, "file"]:
            h = hover_slice(pd.read_csv(os.path.join(CSV_DIR, f)))
            t = (h["timestamp"].to_numpy() - h["timestamp"].iloc[0]) / 1000.0
            ex = (h["chamber.dFront"] - h["chamber.dBack"]).to_numpy() / 2
            ey = (h["chamber.dLeft"] - h["chamber.dRight"]).to_numpy() / 2
            ax.plot(t, np.hypot(ex, ey), color=COLORS[cond], alpha=0.45, lw=1.0)
    ax.set_xlabel("time in HOVER [s]")
    ax.set_ylabel("center distance r [m]")
    ax.set_title("(a) distance from chamber center")
    ax.legend(handles=[plt.Line2D([], [], color=COLORS[c], label=c) for c in conds],
              fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.boxplot([ok.loc[ok["cond"] == c, "r_mean"] for c in conds],
               labels=[short[c] for c in conds])
    for i, c in enumerate(conds, 1):
        y = ok.loc[ok["cond"] == c, "r_mean"]
        ax.scatter(np.full(len(y), i) + np.random.uniform(-.06, .06, len(y)), y,
                   color=COLORS[c], zorder=3, s=22)
    ax.set_ylabel("mean r [m]")
    ax.set_title("(b) per-flight mean centering error")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.boxplot([ok.loc[ok["cond"] == c, "noise"] * 1000 for c in conds],
               labels=[short[c] for c in conds])
    for i, c in enumerate(conds, 1):
        y = ok.loc[ok["cond"] == c, "noise"] * 1000
        ax.scatter(np.full(len(y), i) + np.random.uniform(-.06, .06, len(y)), y,
                   color=COLORS[c], zorder=3, s=22)
    ax.set_yscale("log")
    ax.set_ylabel("wall-distance noise [mm, 1 sigma]")
    ax.set_title("(c) wall-distance estimate noise (log scale)")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for cond in conds:
        s_ = ok[ok["cond"] == cond]
        ax.scatter(s_["ex_mean"], s_["ey_mean"], color=COLORS[cond], s=40, label=cond)
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("mean ex = (front-back)/2 [m]")
    ax.set_ylabel("mean ey = (left-right)/2 [m]")
    ax.set_title("(d) steady-state offset direction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    ax.bar(x - w / 2, [ok.loc[ok["cond"] == c, "r_first2"].mean() for c in conds],
           w, label="first 2 s", color="#bbb", edgecolor="k")
    ax.bar(x + w / 2, [ok.loc[ok["cond"] == c, "r_last2"].mean() for c in conds],
           w, label="last 2 s", color=[COLORS[c] for c in conds], edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels([short[c] for c in conds])
    ax.set_ylabel("r [m]")
    ax.set_title("(e) convergence (first vs last 2 s of HOVER)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 2]
    crash_rate = df.groupby("cond")["crash"].mean()
    ax.bar(x - w / 2, [crash_rate.get(c, np.nan) for c in conds], w,
           color="#e0a0c0", edgecolor="k", label="crash rate")
    ax.bar(x + w / 2, [ok.loc[ok["cond"] == c, "impacts"].mean() / 4 for c in conds], w,
           color="#66a3c2", edgecolor="k", label="wall impacts / flight (/4)")
    for i, c in enumerate(conds):
        ax.text(i + w / 2, ok.loc[ok["cond"] == c, "impacts"].mean() / 4 + 0.01,
                f"{ok.loc[ok['cond'] == c, 'impacts'].mean():.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([short[c] for c in conds])
    ax.set_ylabel("rate")
    ax.set_title("(f) crash rate and wall impacts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("hybrid vs plane-fit vs raw ranging - chamber centering (k=0.3)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(FIGS_DIR, exist_ok=True)
    out_png = os.path.join(FIGS_DIR, "hybrid_compare.png")
    fig.savefig(out_png, dpi=130)
    print(f"\n保存: {out_csv}\n保存: {out_png}")


if __name__ == "__main__":
    main()
