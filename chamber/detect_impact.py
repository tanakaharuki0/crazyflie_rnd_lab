#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU (acc / gyro) の急変から「壁への衝突」を検出する。

原理
----
ホバー中の機体は body 系で acc.x/acc.y ≒ 0 (acc.z ≒ 1g が重力)。
壁に当たると水平方向に衝撃が入るので、水平加速度の大きさ

    a_h = hypot(acc.x, acc.y)   [g]

がスパイクする。実測 (chamber/csv, 27 Hz) では

    接触の無いホバー : a_h max = 0.06 g (中央値 0.02 g)
    軽い接触         : a_h = 0.2〜0.9 g
    強い接触         : a_h = 1.5〜4.9 g

と、接触の有無で 3倍以上あいており、しきい値 0.2 g できれいに切れる。
補助的にジャイロの急変 (角加速度) も見られるが、単独では使えない
(接触時 3000〜8000 deg/s^2 に対し、通常の姿勢制御でも 2000〜5000 出る)。

既定の判定は加速度のみ (a_h > ACC_TH)。実測では a_h が最もよく分離する
(平常 0.05〜0.3 g / 接触 1.5〜5 g) のに対し、角加速度は接触でも
3000〜8000 と幅が広く、旋回でも同程度出るため単独では切れない。
--mode and / or でジャイロを併用できる。

検出はレート 27 Hz なので、1回の接触が複数サンプルに散る。
RESET_S 以内の検出は 1 イベントにまとめる (不応期)。

誤検出の除去
------------
**着地の接地は必ず 0.3〜3.5 g のスパイクを出す**ので、既定では
HOVER (state==6) だけを見る。--all-states を付けると TAKEOFF/LAND も
含めるが、LAND のスパイクは壁ではなく床なので解釈に注意。

急旋回でも a_h は跳ねる。イベント時刻の壁距離
(chamber.dFront/dBack/dLeft/dRight) を見て、
どの壁からも NEAR_M より遠ければ「壁ではない」として落とす。
最も近い壁をその接触の相手として報告する。

使い方
------
    python3 detect_impact.py csv/log167_without_Pcontrol_pose.csv
    python3 detect_impact.py csv/log1*_pose.csv --summary
    python3 detect_impact.py csv/log167_*_pose.csv --plot figs/
    python3 detect_impact.py ... --all-states   # HOVER以外も含める

較正結果 (目視記録と対応が確実な k=0.3 / k=0.7 の非墜落15本)
----------------------------------------------------------
HOVER中に壁へ 5 cm 以内まで近づいた 6本では、接触の有無が 6/6 一致、
接触回数も 目視9回 に対し 検出10回 とほぼ一致した。
残る9本は HOVER中に壁から 7〜15 cm 以上離れたままで、
目視で数えた接触は離陸・着陸フェーズで起きたものと考えられる。
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

STATE_HOVER = 6

ACC_TH = 0.20     # [g]  水平加速度のしきい値 (実測で較正)
GYRO_TH = 4000.0  # [deg/s^2] 角加速度のしきい値
RESET_S = 0.30    # [s]  この時間内の検出は同一イベント
NEAR_M = 0.10     # [m]  これより近い壁があるときだけ「壁接触」とみなす
STRONG_G = 1.5    # [g]  これ以上は壁距離によらず接触とみなす (他に説明がつかない)
MAX_DT = 0.20     # [s]  これより大きいサンプル間隔をまたぐ微分は捨てる

WALLS = {"chamber.dFront": "前", "chamber.dBack": "後",
         "chamber.dLeft": "左", "chamber.dRight": "右"}


def load(path, all_states=False):
    d = pd.read_csv(path)
    if "chamber.state" in d.columns and not all_states:
        d = d[d["chamber.state"] == STATE_HOVER]
    d = d.reset_index(drop=True).copy()
    if len(d) < 5:
        return None
    d["t"] = (d["timestamp"] - d["timestamp"].iloc[0]) / 1000.0
    return d


def features(d):
    """水平加速度 a_h [g] と 角加速度の大きさ |dw/dt| [deg/s^2] を返す。"""
    a_h = np.hypot(d["acc.x"].to_numpy(), d["acc.y"].to_numpy())

    t = d["t"].to_numpy()
    w = d[["gyro.x", "gyro.y", "gyro.z"]].to_numpy()
    dt = np.diff(t)
    dw = np.linalg.norm(np.diff(w, axis=0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = dw / dt
    rate[dt > MAX_DT] = 0.0          # ログの穴をまたぐ差分は無効
    gj = np.zeros_like(a_h)
    gj[1:] = rate                    # サンプル i の値は i-1 -> i の変化
    return a_h, gj


def nearest_wall(row):
    """そのサンプルで一番近い壁とその距離。-1 は無効値なので無視。"""
    best, bestd = None, np.inf
    for col, name in WALLS.items():
        v = row.get(col, np.nan)
        if pd.isna(v) or v < 0:      # -1.0 = 測距失敗
            continue
        if v < bestd:
            best, bestd = name, v
    return best, bestd


def detect(d, acc_th=ACC_TH, gyro_th=GYRO_TH, near_m=NEAR_M,
           reset_s=RESET_S, mode="acc", require_wall=True):
    a_h, gj = features(d)
    hit_a, hit_g = a_h > acc_th, gj > gyro_th
    hit = {"acc": hit_a, "and": hit_a & hit_g, "or": hit_a | hit_g}[mode]

    t = d["t"].to_numpy()
    events, last_t = [], -np.inf
    for i in np.flatnonzero(hit):
        if t[i] - last_t < reset_s:
            # 同一イベント: より強いピークで代表を上書き
            if a_h[i] > events[-1]["a_h"]:
                events[-1].update(t=t[i], a_h=a_h[i], gyro_jerk=gj[i], i=i)
            last_t = t[i]
            continue
        events.append(dict(i=i, t=t[i], a_h=a_h[i], gyro_jerk=gj[i]))
        last_t = t[i]

    for e in events:
        wall, dist = nearest_wall(d.iloc[e["i"]])
        e["wall"], e["dist"] = wall, dist
        # 壁距離が少し大きくても、この強さのスパイクは接触以外に説明がつかない
        # (接触の無いホバーの a_h max は 0.06 g)。測距の遅れ/欠測を救う。
        e["strong"] = e["a_h"] >= STRONG_G
        e["is_wall"] = e["strong"] or ((wall is not None) and (dist < near_m))

    if require_wall:
        rejected = [e for e in events if not e["is_wall"]]
        events = [e for e in events if e["is_wall"]]
    else:
        rejected = []
    return events, rejected, a_h, gj


def name_of(path):
    m = re.match(r"(log\d+)_(.+?)(?:_pose)?\.csv", os.path.basename(path))
    return (m.group(1), m.group(2)) if m else (os.path.basename(path), "")


def plot(d, a_h, gj, events, rejected, path, outdir, acc_th, gyro_th):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    log, tag = name_of(path)
    t = d["t"].to_numpy()
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axs[0].plot(t, a_h, lw=.9, color="tab:red")
    axs[0].axhline(acc_th, ls="--", c="k", lw=.7, label=f"しきい値 {acc_th} g")
    axs[0].set_ylabel("水平加速度 $a_h$ [g]"); axs[0].legend(fontsize=8)

    axs[1].plot(t, gj, lw=.9, color="tab:purple")
    axs[1].axhline(gyro_th, ls="--", c="k", lw=.7, label=f"しきい値 {gyro_th:.0f}")
    axs[1].set_ylabel("角加速度 [deg/s²]"); axs[1].legend(fontsize=8)

    for col, nm in WALLS.items():
        if col in d.columns:
            v = d[col].to_numpy().copy(); v[v < 0] = np.nan
            axs[2].plot(t, v, lw=.9, label=nm)
    axs[2].axhline(NEAR_M, ls="--", c="k", lw=.7)
    axs[2].set_ylabel("壁までの距離 [m]"); axs[2].set_xlabel("HOVER経過 [s]")
    axs[2].legend(fontsize=8, ncol=4)

    for ax in axs:
        for e in events:
            ax.axvline(e["t"], color="tab:red", alpha=.35, lw=2)
        for e in rejected:
            ax.axvline(e["t"], color="gray", alpha=.35, lw=2, ls=":")
        ax.grid(alpha=.3)

    axs[0].set_title(f"{log} ({tag})  衝突検出 {len(events)} 回"
                     f"{f' / 棄却 {len(rejected)}' if rejected else ''}",
                     fontsize=13, fontweight="bold")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{log}_{tag}_impact.png")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="IMUの急変から壁衝突を検出する")
    ap.add_argument("csvs", nargs="+", help="pose CSV (chamber/csv/log*_pose.csv)")
    ap.add_argument("--acc-th", type=float, default=ACC_TH)
    ap.add_argument("--gyro-th", type=float, default=GYRO_TH)
    ap.add_argument("--near", type=float, default=NEAR_M, help="壁とみなす距離 [m]")
    ap.add_argument("--mode", choices=["acc", "and", "or"], default="acc",
                    help="判定則: acc=加速度のみ(既定) / and / or")
    ap.add_argument("--no-wall-check", action="store_true", help="壁距離による棄却をしない")
    ap.add_argument("--all-states", action="store_true", help="HOVER以外の区間も含める")
    ap.add_argument("--summary", action="store_true", help="1行/ログ の要約のみ")
    ap.add_argument("--plot", metavar="DIR", help="検出結果の図を出す")
    args = ap.parse_args()

    rows = []
    for path in args.csvs:
        d = load(path, args.all_states)
        if d is None:
            print(f"{os.path.basename(path)}: サンプル不足", file=sys.stderr)
            continue
        ev, rej, a_h, gj = detect(d, args.acc_th, args.gyro_th, args.near,
                                  mode=args.mode,
                                  require_wall=not args.no_wall_check)
        log, tag = name_of(path)
        rows.append(dict(log=log, tag=tag, n=len(ev), rejected=len(rej),
                         a_h_max=a_h.max(), t=d["t"].iloc[-1],
                         walls="".join(e["wall"] or "?" for e in ev)))
        if not args.summary:
            print(f"\n=== {log} ({tag})  {d['t'].iloc[-1]:.2f}s, {len(d)}サンプル ===")
            if not ev:
                print("  衝突なし")
            for e in ev:
                print(f"  t={e['t']:6.2f}s  a_h={e['a_h']:5.2f}g  "
                      f"|dω/dt|={e['gyro_jerk']:8.0f}  {e['wall']}壁 {e['dist']*100:5.1f}cm"
                      f"{'  [強]' if e['strong'] else ''}")
            for e in rej:
                print(f"  (棄却) t={e['t']:6.2f}s  a_h={e['a_h']:5.2f}g  "
                      f"最近傍 {e['wall']}壁 {e['dist']*100:5.1f}cm > {args.near*100:.0f}cm")
        if args.plot:
            print(f"  -> {plot(d, a_h, gj, ev, rej, path, args.plot, args.acc_th, args.gyro_th)}")

    if len(rows) > 1:
        df = pd.DataFrame(rows)
        print("\n=== 要約 ===")
        print(df.round(2).to_string(index=False))
        print("\n=== 条件別 ===")
        print(df.groupby("tag").agg(logs=("log", "size"), 衝突計=("n", "sum"),
                                    平均=("n", "mean"), 無接触=("n", lambda s: (s == 0).sum()),
                                    棄却=("rejected", "sum")).round(2).to_string())


if __name__ == "__main__":
    main()
