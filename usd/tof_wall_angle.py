#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToF センサーだけで「機体が前後・左右の壁に対して何度傾いているか」を求める。

ジャイロは一切使わない (積分もしないので原理的にドリフトしない)。

手順:
  1. 各センサーの 16 ゾーンを PCA+RANSAC で平面フィットし、壁の法線 n を得る
     (tof_odometry.py の points_body() / fit_plane_ransac() をそのまま利用)。
  2. 機体座標での法線の方位角 a = atan2(n_y, n_x) を求める。壁は 90 度おきに
     並んでいるので、exp(4ja) の重み付き円環平均をとれば「どの壁を見ているか」
     を識別せずに全センサーの観測を1つの角度に融合できる。得られる角度は
     [-45, +45) 度の範囲で、「最も近い壁面に対する機体の傾き」を表す。
  3. 平均から大きく外れた面は捨てて再計算する (ジャイロ整合ではなく、
     その場のセンサー同士の整合だけで外れ値を除去する)。
  4. 5Hz の単発測定はばらつくので、複素平面のまま時間方向に移動平均する。

ジャイロ/EKF の姿勢は推定には一切使わず、検証用にグラフへ重ねるだけ。

使い方:
    python3 tof_wall_angle.py log58_pose.csv
    python3 tof_wall_angle.py log58_pose.csv -s 0 1 2 5 6 7 --csv-out csv/log58_wall.csv
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tof
from tof_odometry import (CSV_DIR, OUT_DIR, PLANARITY_EPS,
                          points_body, fit_plane_ransac, sensor_arrays, tof_frames)

# 融合後の平均から この角度以上ずれた面は外れ値として捨てる [deg]
REJECT_DEG = 15.0
REJECT_ITERS = 2

# 時間方向の移動平均の窓 [s] (ToF は約5Hz なので 1.0s ≒ 5フレーム)
SMOOTH_WINDOW_S = 1.0

# 法線がこれ以上鉛直に近い面は「垂直な壁ではない」(天井/床) として使わない
NZ_MAX = 0.5      # |n_z| の上限 (0.5 = 鉛直から60度以上傾いた法線は除外)

# --- sensor0 / sensor2 が別々の壁を見てしまったときの棄却ルール ---
# 斜め 22.5 度を向いた s0/s2 は、狭いチャンバーでは隣接する2面の角に差し掛かり
# やすく、フレームごとにどちらの壁にフィットするかが切り替わる (実測で s2 は
# 32% のフレームで他面から 30 度以上ずれていた)。
CONFLICT_DEG = 30.0    # s0 と s2 がこれ以上ずれていたら「別の面を見ている」
SAME_PLANE_DEG = 10.0  # 2つの面がこれ未満なら「同じ面」とみなす
# s0/s2 にとっての「sensor1 ではない方の隣」
OUTER_NEIGHBOR = {0: 5, 2: 6}

# 面の重み付け。実測では 1/planarity は planarity が桁で振れるため一面が
# 支配してしまい、融合角のフレーム間ばらつきが 4度 → 19度 に悪化した。
# 均等重みが最も安定するので既定とする。
WEIGHT_MODE = "uniform"       # "uniform" | "inv_planarity" | "inv_sqrt_planarity"


def plane_weight(planarity):
    if WEIGHT_MODE == "inv_planarity":
        return 1.0 / max(planarity, 1e-6)
    if WEIGHT_MODE == "inv_sqrt_planarity":
        return 1.0 / np.sqrt(max(planarity, 1e-6))
    return 1.0

# 壁の周期 [deg]。直方体なので 90 度おき
WALL_PERIOD_DEG = 90.0
K = 360.0 / WALL_PERIOD_DEG      # 4 (4倍角にして円環平均する)


def wrap_wall(deg):
    """角度を [-45, +45) 度 (壁の周期 90 度の代表値) に畳む。"""
    return (np.asarray(deg) + WALL_PERIOD_DEG / 2) % WALL_PERIOD_DEG - WALL_PERIOD_DEG / 2


def fuse_angles(angles, weights):
    """複数の面の方位角を 4倍角の円環平均で融合し、外れ値を反復除去する。

    Returns: (融合角[-45,45) deg, 集中度 0〜1, 採用された面のマスク)
    集中度 |z| は面同士の整合度で、1 に近いほど全ての面が同じ答えを出している。
    """
    a = np.radians(np.asarray(angles, dtype=float))
    w = np.asarray(weights, dtype=float)
    use = np.ones(len(a), dtype=bool)

    for _ in range(REJECT_ITERS + 1):
        if use.sum() == 0:
            return np.nan, 0.0, use
        z = np.sum(w[use] * np.exp(1j * K * a[use])) / np.sum(w[use])
        if use.sum() <= 2 or abs(z) < 1e-9:
            break
        # 平均からの残差 [deg] (壁の周期で畳まれている)
        resid = np.degrees(np.angle(np.exp(1j * K * a) / (z / abs(z)))) / K
        new_use = np.abs(resid) <= REJECT_DEG
        if new_use.sum() < 2 or (new_use == use).all():
            break
        use = new_use

    z = np.sum(w[use] * np.exp(1j * K * a[use])) / np.sum(w[use])
    return float(np.degrees(np.angle(z)) / K), float(np.abs(z)), use


def _angdiff(a, b):
    """2つの方位角の差の絶対値 [0,180] deg。同じ壁なら0付近、隣の壁なら90付近。"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def reject_front_conflict(ang_by_sensor):
    """sensor0 と sensor2 が別々の面を捉えているフレームで、誤った側を棄却する。

    判定:
      1. s0 と s2 の平面が CONFLICT_DEG 以上ずれていなければ何もしない。
      2. ずれている場合、s0/s2 のうち次のいずれかに当てはまる方を棄却する。
         - sensor1 と異なる面を捉えている (s1 との差が SAME_PLANE_DEG 以上)
         - sensor1 ではない方の隣 (s0→s5, s2→s6) と同じ面を捉えている
           (差が SAME_PLANE_DEG 未満)。横のセンサーと同じ壁を見ている
           ということは、正面の壁ではなく横の壁にフィットしている。
      3. どちらも棄却されなかった場合は、判断材料がないので両方棄却する。

    ang_by_sensor: {sensor番号: 方位角[deg]} (平面が取れたセンサーのみ)
    Returns: (棄却するセンサー番号の集合, 競合が起きたか)
    """
    if 0 not in ang_by_sensor or 2 not in ang_by_sensor:
        return set(), False
    if _angdiff(ang_by_sensor[0], ang_by_sensor[2]) < CONFLICT_DEG:
        return set(), False

    reject = set()
    for s in (0, 2):
        a = ang_by_sensor[s]
        # sensor1 と異なる面を見ている
        if 1 in ang_by_sensor and _angdiff(a, ang_by_sensor[1]) >= SAME_PLANE_DEG:
            reject.add(s)
            continue
        # 外側の隣 (s0→s5, s2→s6) と同じ面を見ている
        nb = OUTER_NEIGHBOR[s]
        if nb in ang_by_sensor and _angdiff(a, ang_by_sensor[nb]) < SAME_PLANE_DEG:
            reject.add(s)

    if not reject:          # 判断できなかったので両方捨てる
        reject = {0, 2}
    return reject, True


def smooth_circular(t, ang, conf, window_s=SMOOTH_WINDOW_S):
    """複素平面 (4倍角) のまま時間方向に移動平均する。集中度を重みに使う。"""
    F = len(t)
    out = np.full(F, np.nan)
    ok = np.isfinite(ang)
    if not ok.any():
        return out
    z = np.where(ok, np.exp(1j * K * np.radians(np.nan_to_num(ang))), 0.0)
    w = np.where(ok, np.nan_to_num(conf), 0.0)
    for f in range(F):
        m = ok & (np.abs(t - t[f]) <= window_s / 2)
        if not m.any() or w[m].sum() <= 0:
            continue
        zz = np.sum(w[m] * z[m]) / np.sum(w[m])
        out[f] = np.degrees(np.angle(zz)) / K
    return out


def unwrap_wall(ang):
    """[-45,45) に畳まれた角度列を連続になるように 90 度単位で展開する。

    機体が壁の境界 (±45度) をまたいで回った場合に、時系列として連続に
    見えるようにするための表示用の処理 (推定そのものには影響しない)。
    """
    out = np.array(ang, dtype=float)
    ok = np.flatnonzero(np.isfinite(out))
    if len(ok) < 2:
        return out
    acc = 0.0
    prev = out[ok[0]]
    for i in ok[1:]:
        d = out[i] + acc - prev
        while d > WALL_PERIOD_DEG / 2:
            acc -= WALL_PERIOD_DEG
            d -= WALL_PERIOD_DEG
        while d < -WALL_PERIOD_DEG / 2:
            acc += WALL_PERIOD_DEG
            d += WALL_PERIOD_DEG
        out[i] += acc
        prev = out[i]
    out[ok[0]] = ang[ok[0]]
    return out


def wall_angle(df, sensors, rng, arrs=None, idx=None, fresh=None):
    """全フレームで壁に対する傾きを推定する。

    Returns: dict(t, angle, angle_smooth, conf, n_used, n_planes,
                  per_sensor={s: (F,) 各面の方位角[-45,45)},
                  used={s: (F,) bool 融合に採用されたか})
    """
    if arrs is None:
        arrs = sensor_arrays(df, sensors)
    if idx is None:
        idx = np.arange(len(df))
        fresh = {s: np.ones(len(idx), dtype=bool) for s in sensors}

    F = len(idx)
    t = df["timestamp"].to_numpy(float)[idx] / 1000.0
    t = t - t[0]
    res = dict(
        t=t,
        angle=np.full(F, np.nan),
        conf=np.zeros(F),
        n_used=np.zeros(F, dtype=int),
        n_planes=np.zeros(F, dtype=int),
        per_sensor={s: np.full(F, np.nan) for s in sensors},
        used={s: np.zeros(F, dtype=bool) for s in sensors},
        planarity={s: np.full(F, np.nan) for s in sensors},
        # s0/s2 の面の食い違いで棄却されたフレーム
        front_dropped={s: np.zeros(F, dtype=bool) for s in sensors},
        conflict=np.zeros(F, dtype=bool),
    )

    for f, row in enumerate(idx):
        ang_by_sensor, w_by_sensor = {}, {}
        for s in sensors:
            if not fresh[s][f]:
                continue
            P, _ = points_body(arrs[s][row], s)
            pl, reason = fit_plane_ransac(P, rng)
            if reason != "ok":
                continue                      # 棄却は平面性スコアのみ
            n = pl["n"]
            if abs(n[2]) > NZ_MAX:
                continue                      # 天井/床の平面は方位角を持たない
            a = np.degrees(np.arctan2(n[1], n[0]))
            ang_by_sensor[s] = a
            w_by_sensor[s] = plane_weight(pl["planarity"])
            res["per_sensor"][s][f] = wrap_wall(a)
            res["planarity"][s][f] = pl["planarity"]

        res["n_planes"][f] = len(ang_by_sensor)

        # s0/s2 が別の面を見ているフレームの棄却
        dropped, conflict = reject_front_conflict(ang_by_sensor)
        res["conflict"][f] = conflict
        for s in dropped:
            res["front_dropped"][s][f] = True
            ang_by_sensor.pop(s, None)
            w_by_sensor.pop(s, None)

        srcs = sorted(ang_by_sensor)
        angs = [ang_by_sensor[s] for s in srcs]
        wts = [w_by_sensor[s] for s in srcs]

        if not angs:
            continue
        ang, conf, use = fuse_angles(angs, wts)
        res["angle"][f] = ang
        res["conf"][f] = conf
        res["n_used"][f] = int(use.sum())
        for s, u in zip(srcs, use):
            res["used"][s][f] = bool(u)

    res["angle_smooth"] = smooth_circular(t, res["angle"], res["conf"])
    return res


def plot(res, sensors, df, idx, out_path, hover=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tof_odometry_chamber import _use_jp_font, draw_hover_lines

    _use_jp_font()
    t = res["t"]
    fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    prop = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # (0) 各面の観測と融合結果
    for i, s in enumerate(sensors):
        a = res["per_sensor"][s]
        u = res["used"][s]
        c = prop[i % len(prop)]
        ax[0].scatter(t[u], a[u], s=26, color=c, edgecolors="k", linewidths=0.3,
                      zorder=3, label=f"sensor{s}")
        fd = res["front_dropped"][s] & np.isfinite(a)
        rej = np.isfinite(a) & ~u & ~fd
        if rej.any():
            ax[0].scatter(t[rej], a[rej], s=26, facecolors="none", edgecolors=c,
                          linewidths=1.0, zorder=2)
        if fd.any():
            ax[0].scatter(t[fd], a[fd], s=42, marker="x", color=c,
                          linewidths=1.4, zorder=2)
    ax[0].plot(t, res["angle"], lw=1.0, color="0.4", zorder=4, label="融合 (フレームごと)")
    ax[0].plot(t, res["angle_smooth"], lw=2.2, color="k", zorder=5,
               label=f"融合 + 時間平滑 ({SMOOTH_WINDOW_S:g}s)")
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].set_ylim(-46, 46)
    ax[0].set_ylabel("壁に対する傾き [deg]")
    ax[0].set_title("ToF のみで求めた壁に対する機体の傾き "
                    "(●=融合に採用, ○=外れ値として棄却, ×=s0/s2の面の食い違いで棄却)")

    # (1) 集中度と使用面数
    ax[1].plot(t, res["conf"], lw=1.4, color="tab:blue", label="集中度 |z| (面同士の整合)")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("集中度 |z|")
    b = ax[1].twinx()
    b.plot(t, res["n_planes"], lw=1.0, ls=":", color="tab:gray", label="平面が取れた面数")
    b.plot(t, res["n_used"], lw=1.2, color="tab:orange", label="融合に採用した面数")
    b.set_ylabel("面数")
    b.set_ylim(0, len(sensors) + 0.5)
    lines = ax[1].get_lines() + b.get_lines()
    ax[1].legend(lines, [l.get_label() for l in lines], fontsize=8, ncol=3)
    ax[1].set_title("推定の信頼度 (|z| が低い = 面同士が矛盾している)")

    # (2) 検証: EKF ヨーとの比較 (推定には使っていない)
    yaw_col = "stabilizer.yaw"
    ax[2].plot(t, unwrap_wall(res["angle_smooth"]), lw=2.0, color="k",
               label="ToF 壁角 (展開表示)")
    if yaw_col in df.columns:
        yaw = df[yaw_col].to_numpy(float)[idx]
        yaw = np.degrees(np.unwrap(np.radians(yaw)))
        ok = np.isfinite(res["angle_smooth"])
        if ok.any():
            f0 = int(np.flatnonzero(ok)[0])
            yaw_al = yaw - yaw[f0] + unwrap_wall(res["angle_smooth"])[f0]
            ax[2].plot(t, yaw_al, lw=1.4, ls="--", color="tab:red",
                       label="EKF yaw (初回で合わせただけ・推定には未使用)")
            d = wrap_wall(unwrap_wall(res["angle_smooth"]) - yaw_al)
            rmse = float(np.sqrt(np.nanmean(d ** 2)))
            ax[2].set_title(f"検証: EKF ヨーとの比較 (差の RMSE {rmse:.2f} deg)")
    ax[2].set_ylabel("angle [deg]")
    if not ax[2].get_title():
        ax[2].set_title("検証: EKF ヨーとの比較")

    # (3) 各面の平面性スコア
    for i, s in enumerate(sensors):
        ax[3].plot(t, res["planarity"][s], lw=1.0, color=prop[i % len(prop)],
                   marker=".", ms=3, label=f"sensor{s}")
    ax[3].axhline(PLANARITY_EPS, color="k", ls="--", lw=0.8, label="ε")
    ax[3].set_yscale("log")
    ax[3].set_ylabel(r"planarity $\lambda_3/\Sigma\lambda$")
    ax[3].set_xlabel("time [s]")
    ax[3].set_title("平面性スコア (これだけが棄却条件。ジャイロは一切使わない)")

    for i, a in enumerate(ax):
        a.grid(alpha=0.3)
        draw_hover_lines(a, hover, label=(i == 0))
        if i != 1:
            a.legend(fontsize=8, ncol=4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"壁角グラフを保存: {out_path}")


def main():
    global REJECT_DEG, SMOOTH_WINDOW_S, WEIGHT_MODE, CONFLICT_DEG, SAME_PLANE_DEG
    ap = argparse.ArgumentParser(
        description="ToF のみで壁に対する機体の傾きを求める (ジャイロ不使用)")
    ap.add_argument("csv", help="log_to_csv.py が出力した CSV")
    ap.add_argument("-s", "--sensors", type=int, nargs="+",
                    default=[0, 1, 2, 5, 6, 7],
                    help="使用するセンサー番号 (既定: 0 1 2 5 6 7 = 水平リング)")
    ap.add_argument("--eps", type=float, default=None,
                    help=f"平面性スコアの閾値 (既定: {PLANARITY_EPS})")
    ap.add_argument("--reject-deg", type=float, default=REJECT_DEG,
                    help=f"融合平均からの外れ値棄却の閾値[deg] (既定: {REJECT_DEG})")
    ap.add_argument("--conflict-deg", type=float, default=CONFLICT_DEG,
                    help=f"s0/s2 が別の面を見ているとみなす角度差[deg] (既定: {CONFLICT_DEG})")
    ap.add_argument("--same-plane-deg", type=float, default=SAME_PLANE_DEG,
                    help=f"2面を同じ面とみなす角度差[deg] (既定: {SAME_PLANE_DEG})")
    ap.add_argument("--no-front-reject", action="store_true",
                    help="s0/s2 の面の食い違いによる棄却を行わない")
    ap.add_argument("--weight", default=WEIGHT_MODE,
                    choices=["uniform", "inv_planarity", "inv_sqrt_planarity"],
                    help=f"面の重み付け (既定: {WEIGHT_MODE})")
    ap.add_argument("--smooth", type=float, default=SMOOTH_WINDOW_S,
                    help=f"時間平滑の窓[s] (既定: {SMOOTH_WINDOW_S})")
    ap.add_argument("--no-dedup", action="store_true",
                    help="ToF更新行での間引きをせず全行をフレームとして扱う")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    REJECT_DEG = args.reject_deg
    WEIGHT_MODE = args.weight
    CONFLICT_DEG = np.inf if args.no_front_reject else args.conflict_deg
    SAME_PLANE_DEG = args.same_plane_deg
    SMOOTH_WINDOW_S = args.smooth

    import pandas as pd
    path = args.csv
    if not os.path.isfile(path):
        path = os.path.join(CSV_DIR, os.path.basename(args.csv))
        if not os.path.isfile(path):
            sys.exit(f"ファイルが見つかりません: {args.csv}")
    df = pd.read_csv(path)
    if args.eps is not None:
        tof.PLANARITY_EPS = args.eps

    rng = np.random.default_rng(args.seed)
    arrs = sensor_arrays(df, args.sensors)
    if args.no_dedup:
        idx = np.arange(len(df))
        fresh = {s: np.ones(len(df), dtype=bool) for s in args.sensors}
    else:
        idx, fresh = tof_frames(arrs, args.sensors,
                                df["timestamp"].to_numpy(float) / 1000.0)

    res = wall_angle(df, args.sensors, rng, arrs, idx, fresh)
    dfi = df.iloc[idx].reset_index(drop=True)

    F = len(idx)
    valid = np.isfinite(res["angle"])
    print(f"入力: {path}  ({F} フレーム)")
    print(f"  推定できたフレーム : {int(valid.sum())}/{F} "
          f"({100.0 * valid.sum() / F:.1f}%)")
    print(f"  1フレームあたり使用面数 : 平均 {res['n_used'][valid].mean():.2f} "
          f"(平面が取れた面 {res['n_planes'][valid].mean():.2f})")
    print(f"  集中度 |z| : 平均 {res['conf'][valid].mean():.2f} "
          f"(中央値 {np.median(res['conf'][valid]):.2f})")
    print(f"  壁に対する傾き : 平均 {np.nanmean(res['angle_smooth']):+.2f} deg, "
          f"範囲 {np.nanmin(res['angle_smooth']):+.1f}〜{np.nanmax(res['angle_smooth']):+.1f} deg")
    # フレーム間の変化 (平滑前) = 単発測定のばらつきの目安
    d = wrap_wall(np.diff(res["angle"][valid]))
    if len(d):
        print(f"  連続フレーム間の変化 : 中央値 {np.median(np.abs(d)):.2f} deg, "
              f"1σ {d.std():.2f} deg")

    n_conf = int(res["conflict"].sum())
    print(f"  s0/s2 の面の食い違い : {n_conf}/{F} フレーム "
          f"({100.0 * n_conf / F:.1f}%)"
          + (f"  棄却 s0={int(res['front_dropped'][0].sum())} "
             f"s2={int(res['front_dropped'][2].sum())} "
             f"(両方棄却 {int((res['front_dropped'][0] & res['front_dropped'][2]).sum())})"
             if 0 in res["front_dropped"] and 2 in res["front_dropped"] else ""))

    # 面ごとの診断: 融合結果に対する残差の中央値 (取り付け角のズレが見える) と棄却率
    print("  面ごとの診断 (融合結果に対する残差の中央値 / 棄却率):")
    for s_ in args.sensors:
        a = res["per_sensor"][s_]
        m = np.isfinite(a) & np.isfinite(res["angle"])
        if not m.any():
            print(f"    sensor{s_}: 平面なし")
            continue
        resid = wrap_wall(a[m] - res["angle"][m])
        n_rej = int((m & ~res["used"][s_]).sum())
        print(f"    sensor{s_}: 残差 中央値 {np.median(resid):+6.2f} deg "
              f"(1σ {resid.std():5.2f}), 棄却 {n_rej}/{int(m.sum())} "
              f"({100.0 * n_rej / m.sum():.0f}%)")

    if "stabilizer.yaw" in dfi.columns:
        yaw = np.degrees(np.unwrap(np.radians(dfi["stabilizer.yaw"].to_numpy(float))))
        ok = np.isfinite(res["angle_smooth"])
        if ok.any():
            f0 = int(np.flatnonzero(ok)[0])
            u = unwrap_wall(res["angle_smooth"])
            diff = wrap_wall(u - (yaw - yaw[f0] + u[f0]))
            bias = np.nanmedian(diff)
            print(f"  EKFヨーとの差 (検証のみ) : RMSE {np.sqrt(np.nanmean(diff**2)):.2f} deg "
                  f"(定常オフセット {bias:+.2f} deg を除くと "
                  f"{np.sqrt(np.nanmean((diff - bias)**2)):.2f} deg), "
                  f"最大 {np.nanmax(np.abs(diff)):.2f} deg")

    from tof_odometry_chamber import hover_spans
    hover = hover_spans(dfi, res["t"])
    if hover:
        print("  ホバリング区間 : "
              + ", ".join(f"{a:.1f}〜{b:.1f}s" for a, b in hover))

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_wall_angle.png")
    plot(res, args.sensors, dfi, np.arange(F), out, hover=hover)

    if args.csv_out:
        cols = dict(t=res["t"], angle=res["angle"], angle_smooth=res["angle_smooth"],
                    conf=res["conf"], n_used=res["n_used"], n_planes=res["n_planes"])
        for s in args.sensors:
            cols[f"a_s{s}"] = res["per_sensor"][s]
            cols[f"used_s{s}"] = res["used"][s].astype(int)
            cols[f"front_dropped_s{s}"] = res["front_dropped"][s].astype(int)
        cols["conflict"] = res["conflict"].astype(int)
        pd.DataFrame(cols).to_csv(args.csv_out, index=False)
        print(f"推定結果CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
