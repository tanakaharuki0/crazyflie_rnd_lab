#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
チャンバー内自己位置推定 (tof_odometry.py の派生)。

内寸 550 x 550 x 500 mm の直方体チャンバー内をドローンが飛行した uSD ログから、
PCA+RANSAC による平面抽出 と EKF (stabilizer.*) の姿勢を融合して
「チャンバーのどこにいるか」を推定・可視化する。

tof_odometry.py からの変更点:

  ① 融合に使う姿勢を「生ジャイロの積分」ではなく EKF の出力
     (stabilizer.yaw / pitch / roll) にした。ゲート判定・ジャイロ代用・
     法線の世界座標への変換すべてでこの姿勢を使う。

  ② 平面までの垂直距離 d_i を壁までの距離として使い、チャンバー座標
     (x,y,z) を直接求める絶対位置推定を追加した。
     平面性スコアまたは法線の方位角 (ジャイロ整合ゲート) で棄却されて
     壁が取れなかった軸は、
        stateEstimate.x/y/z → 無ければ IMU (acc + EKF姿勢) の推測航法
     を補完データとして、直前の壁フィックスからの相対移動で埋める。

  ③ 3Dボックス・上面図・側面図・時系列で可視化する。

使い方:
    python3 tof_odometry_chamber.py log51.csv
    python3 tof_odometry_chamber.py log51.csv -s 0 1 2 5 6 7 --csv-out csv/log51_chamber.csv
"""
import argparse
import os
import sys

import numpy as np

import tof_odometry as tof
from tof_odometry import (CSV_DIR, OUT_DIR, GATE_DEG, PLANARITY_EPS,
                          rot_ypr, rot_to_ypr, sensor_arrays, tof_frames,
                          extract_planes, run_odometry)

# ---- チャンバー内寸 [mm] (x,y,z) ----
# 座標原点は「高さ0 (床) におけるチャンバーの中心」。
#   x, y ∈ [-X/2, +X/2] (中心が0)
#   z    ∈ [0, Z]       (床が0)
# Crazyflie は原点 (0,0,0) に置かれた状態から離陸する前提。
CHAMBER_SIZE = np.array([550.0, 550.0, 500.0])


def chamber_bounds():
    """チャンバー座標の下限・上限 [mm] を (lo, hi) で返す (床中心が原点)。"""
    X, Y, Z = CHAMBER_SIZE
    return (np.array([-X / 2.0, -Y / 2.0, 0.0]),
            np.array([X / 2.0, Y / 2.0, Z]))

# 法線が壁の軸からこの角度以上ずれていたら「どの壁か決められない」として捨てる
WALL_ANGLE_TOL_DEG = 35.0

# 下向き ToF (range.zrange) を床までの距離として z のフィックスに使うか
USE_ZRANGE = True
ZRANGE_MAX_MM = 2000.0

# ホバリング中を表す chamber.state の値 (グラフに開始/終了の縦線を入れる)
HOVER_STATE = 6
STATE_COL = "chamber.state"

G_MMSS = 9806.65      # 重力加速度 [mm/s^2]
IMU_BIAS_SEC = 0.5    # 先頭この秒数の平均を加速度バイアスとして除去
IMU_VEL_TAU_S = 1.0   # 推測航法の速度に掛ける減衰の時定数 [s] (0で減衰なし)


# ============================================================
# ① 姿勢: EKF (stabilizer.*) の出力を使う
# ============================================================
ATT_COLS = ("stabilizer.yaw", "stabilizer.pitch", "stabilizer.roll")


def ekf_attitude(df):
    """EKF の姿勢出力 (yaw,pitch,roll)[deg] を (F,3) で返す。列が無ければ None。

    yaw は ±180 をまたぐとゲート判定が壊れるのでアンラップしておく。
    tof_odometry.py の gyro_reference() の代わりにこれを att として使う。
    """
    if not all(c in df.columns for c in ATT_COLS):
        return None
    att = df[list(ATT_COLS)].to_numpy(dtype=float)
    att[:, 0] = np.degrees(np.unwrap(np.radians(att[:, 0])))
    return att


def attitude_delta(att):
    """姿勢からフレーム間回転量 [deg/frame] を作る (グラフの比較用)。"""
    F = len(att)
    d = np.zeros((F, 3))
    for f in range(1, F):
        d[f] = rot_to_ypr(rot_ypr(*att[f - 1]).T @ rot_ypr(*att[f]))
    return d


def hover_spans(df, t):
    """chamber.state == HOVER_STATE (ホバリング中) の区間を [(開始t, 終了t), ...] で返す。

    列が無ければ空リスト。連続するフレームを1区間にまとめる。
    """
    if STATE_COL not in df.columns:
        return []
    m = df[STATE_COL].to_numpy() == HOVER_STATE
    if not m.any():
        return []
    idx = np.flatnonzero(m)
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[brk + 1]])
    ends = np.concatenate([idx[brk], [idx[-1]]])
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


def draw_hover_lines(ax, spans, label=True):
    """ホバリングの開始・終了時刻に黒い縦線を引く。"""
    for i, (t0, t1) in enumerate(spans):
        ax.axvline(t0, color="k", lw=1.2, ls="-",
                   label="ホバリング開始/終了" if (label and i == 0) else None)
        ax.axvline(t1, color="k", lw=1.2, ls="-")


# ============================================================
# ② 壁までの距離からのチャンバー内絶対位置
# ============================================================
def yaw_offset_from_walls(res, sensors, att):
    """機体ヨー基準とチャンバー壁の向きのオフセット [deg] を推定する。

    壁は 90 度おきに 4 面あるので、世界座標での法線方位角 a を 4 倍した
    円環平均をとれば、どの壁を見ているかを知らなくてもオフセットが求まる
    (mod 90 度の平均)。結果は [-45,45) 度。
    """
    acc = 0.0 + 0.0j
    for f in range(len(att)):
        R = rot_ypr(*att[f])
        for s in sensors:
            n = res["plane_n"][s][f]
            if not np.isfinite(n).all():
                continue
            nw = R @ n
            if np.hypot(nw[0], nw[1]) < 0.5:      # ほぼ天井/床の法線は使わない
                continue
            a = np.arctan2(nw[1], nw[0])
            acc += np.exp(4j * a)
    if acc == 0:
        return 0.0
    return float(np.degrees(np.angle(acc)) / 4.0)


def wall_fixes(res, sensors, att, yaw_offset, df=None,
               tol_deg=WALL_ANGLE_TOL_DEG, use_gate=True):
    """各フレームで、抽出された平面をチャンバーの6面に割り当てて絶対位置を求める。

    平面距離 d_i は「ボディ原点 → 平面」の垂直距離なので、法線をチャンバー
    座標に回して最も近い軸に割り当てれば、その軸の座標が直接決まる:
        法線が +軸向き → coord = size - d      (正側の壁を見ている)
        法線が -軸向き → coord = d             (負側の壁を見ている)

    棄却の扱い:
      - 平面性スコアで落ちたフレームは plane_n が NaN なので自動的に除外。
      - 法線の方位角がジャイロ(EKF)整合ゲートで棄却されたフレーム
        (res["gated"][s][f]) も use_gate=True なら除外する。
      → その軸はフィックス無しとなり、後段で補完データに任される。

    Returns: dict(fix=(F,3) NaN埋め, n_fix=(F,3) 使った面の数,
                  wall=(F,3) 使った面 (+1/-1/0), src_z_range=(F,) bool)
    """
    F = len(att)
    sums = np.zeros((F, 3))
    cnts = np.zeros((F, 3))
    wall = np.zeros((F, 3))
    z_from_range = np.zeros(F, dtype=bool)
    cos_tol = np.cos(np.radians(tol_deg))

    Rz_off = rot_ypr(-yaw_offset, 0.0, 0.0)

    for f in range(F):
        R = Rz_off @ rot_ypr(*att[f])          # body → chamber
        for s in sensors:
            n = res["plane_n"][s][f]
            d = res["plane_d"][s][f]
            if not (np.isfinite(n).all() and np.isfinite(d)):
                continue
            if use_gate and res["gated"][s][f]:
                continue
            nw = R @ n
            ax = int(np.argmax(np.abs(nw)))
            c = abs(nw[ax])
            if c < cos_tol:
                continue                        # どの壁か決められない
            dist = d / c                        # 軸に垂直な距離に直す
            if not (0.0 <= dist <= CHAMBER_SIZE[ax] * 1.2):
                continue                        # チャンバー内寸と矛盾する測距
            # 原点は床中心なので、正側の壁は +hi[ax]、負側の壁は lo[ax] にある
            lo, hi = chamber_bounds()
            coord = hi[ax] - dist if nw[ax] > 0 else lo[ax] + dist
            sums[f, ax] += coord
            cnts[f, ax] += 1
            wall[f, ax] = 1 if nw[ax] > 0 else -1

    # 下向き ToF (range.zrange) を床までの距離として z に使う
    if USE_ZRANGE and df is not None and "range.zrange" in df.columns:
        zr = df["range.zrange"].to_numpy(dtype=float)
        for f in range(F):
            if not np.isfinite(zr[f]) or zr[f] <= 0 or zr[f] > ZRANGE_MAX_MM:
                continue
            R = rot_ypr(*att[f])
            tilt = float(np.clip((R @ np.array([0.0, 0.0, -1.0]))[2], -1.0, 0.0))
            if abs(tilt) < cos_tol:
                continue
            z = zr[f] * abs(tilt)               # 傾きぶんを補正した床(z=0)からの高さ
            if 0.0 <= z <= CHAMBER_SIZE[2] * 1.2:
                # 下向き専用の測距なので、水平リングのセンサーが傾いた拍子に
                # 天井/床に割り当てられた平面より信頼できる。平均せず上書きする。
                sums[f, 2] = z
                cnts[f, 2] = 1
                wall[f, 2] = -1
                z_from_range[f] = True

    fix = np.full((F, 3), np.nan)
    ok = cnts > 0
    fix[ok] = sums[ok] / cnts[ok]
    return dict(fix=fix, n_fix=cnts, wall=wall, z_from_range=z_from_range)


# ============================================================
# ② 補完データ: stateEstimate → 無ければ IMU 推測航法
# ============================================================
STATE_COLS = ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z")
ACC_COLS = ("acc.x", "acc.y", "acc.z")


def complement_track(df, t_s, att):
    """壁が取れないフレームを埋めるための補完軌跡 [mm] を作る。

    優先度: stateEstimate.x/y/z (EKF の位置) → 無ければ IMU 推測航法。

    IMU 推測航法は EKF 姿勢で加速度を世界座標に回し、重力を引いて 2 回積分
    する。バイアスは先頭 IMU_BIAS_SEC 秒 (静止想定) の平均で除去する。
    絶対値はすぐドリフトするが、ここでは「直前の壁フィックスからの相対移動」
    にしか使わないので、短い区間の埋め合わせとしては機能する。

    Returns: (track (F,3)[mm], 補完データの種類を表す文字列)
    """
    if all(c in df.columns for c in STATE_COLS):
        p = df[list(STATE_COLS)].to_numpy(dtype=float) * 1000.0
        return p, "stateEstimate"

    if not all(c in df.columns for c in ACC_COLS):
        return np.zeros((len(t_s), 3)), "なし"

    a_body = df[list(ACC_COLS)].to_numpy(dtype=float) * G_MMSS   # [g] → [mm/s^2]
    F = len(t_s)
    a_w = np.zeros((F, 3))
    for f in range(F):
        a_w[f] = rot_ypr(*att[f]) @ a_body[f]
    a_w[:, 2] -= G_MMSS

    n_bias = max(int(np.searchsorted(t_s - t_s[0], IMU_BIAS_SEC)), 1)
    a_w = a_w - np.nanmean(a_w[:n_bias], axis=0)
    a_w = np.nan_to_num(a_w)

    # 速度は残留バイアスで際限なく発散するので、時定数 IMU_VEL_TAU_S の
    # 漏れ積分 (リーキーインテグレータ) にする。壁フィックスの間の数秒を
    # 埋めるのが目的なので、それより長い時間スケールの成分は捨ててよい。
    dt = np.diff(t_s, prepend=t_s[0])[:, None]
    F = len(t_s)
    v = np.zeros((F, 3))
    p = np.zeros((F, 3))
    for f in range(1, F):
        decay = (np.exp(-dt[f] / IMU_VEL_TAU_S) if IMU_VEL_TAU_S > 0
                 else np.ones(1))
        v[f] = v[f - 1] * decay + a_w[f] * dt[f]
        p[f] = p[f - 1] + v[f] * dt[f]
    return p, "IMU推測航法"


def fuse_position(fixes, comp, start=None):
    """壁フィックスと補完軌跡を軸ごとに融合してチャンバー内座標を返す。

    フィックスがあるフレームはその絶対座標を採用し、そこを新しいアンカーに
    する。フィックスが無いフレームは「直前のアンカー + 補完軌跡の差分」で
    埋める (補完データの絶対オフセットとドリフトの蓄積を持ち込まない)。

    座標系は「高さ0のチャンバー中心が原点」で、離陸前の機体は (0,0,0) にいる。

    Returns: dict(pos=(F,3), src=(F,3) 0:補完 1:壁フィックス)
    """
    fix = fixes["fix"]
    F = len(fix)
    pos = np.zeros((F, 3))
    src = np.zeros((F, 3), dtype=int)

    # 機体はチャンバー床の中心 (0,0,0) に置かれた状態から離陸する前提なので、
    # 最初の壁フィックスが得られるまでは原点を初期アンカーとする。
    if start is None:
        start = np.zeros(3)
    anchor = np.array(start, dtype=float)
    anchor_comp = comp[0].copy()

    for f in range(F):
        for ax in range(3):
            if np.isfinite(fix[f, ax]):
                anchor[ax] = fix[f, ax]
                anchor_comp[ax] = comp[f, ax]
                pos[f, ax] = fix[f, ax]
                src[f, ax] = 1
            else:
                pos[f, ax] = anchor[ax] + (comp[f, ax] - anchor_comp[ax])
        # チャンバーの外には出ないので内寸でクリップする
        lo, hi = chamber_bounds()
        pos[f] = np.clip(pos[f], lo, hi)
    return dict(pos=pos, src=src)


# ============================================================
# ③ 可視化
# ============================================================
def _use_jp_font():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "IPAGothic", "TakaoGothic", "VL Gothic",
                 "Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_chamber(fuse, fixes, res, att, t, comp_name, out_path, sensors,
                 hover=()):
    """チャンバー内の推定位置を上面図 (x-y 水平面) と各軸の時系列で可視化する。

    座標原点は高さ0におけるチャンバー中心で、機体は (0,0,0) から離陸する。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_jp_font()
    pos = fuse["pos"]
    src = fuse["src"]
    lo, hi = chamber_bounds()

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0])

    # --- 上面図 (x-y 水平面) ---
    axt = fig.add_subplot(gs[0, 0:2])
    axt.add_patch(plt.Rectangle((lo[0], lo[1]), CHAMBER_SIZE[0], CHAMBER_SIZE[1],
                                fill=False, ec="0.5", lw=1.8))
    axt.axhline(0, color="0.8", lw=0.8)
    axt.axvline(0, color="0.8", lw=0.8)
    axt.plot(pos[:, 0], pos[:, 1], color="0.3", lw=0.7, alpha=0.6, zorder=2)
    sc = axt.scatter(pos[:, 0], pos[:, 1], c=t, cmap="viridis", s=22, zorder=3)
    axt.scatter(*pos[0, :2], color="tab:green", s=90, marker="o",
                edgecolors="k", zorder=4, label="開始 (原点)")
    axt.scatter(*pos[-1, :2], color="tab:red", s=90, marker="X",
                edgecolors="k", zorder=4, label="終了")
    # ホバリング区間の開始・終了地点を黒で示す (時系列グラフの縦線と対応)
    for i, (t0, t1) in enumerate(hover):
        for tt, mk in ((t0, "^"), (t1, "v")):
            f = int(np.argmin(np.abs(t - tt)))
            axt.scatter(pos[f, 0], pos[f, 1], s=70, marker=mk, color="k",
                        zorder=5,
                        label=("ホバリング開始/終了" if i == 0 and mk == "^" else None))
    axt.set_xlim(lo[0] - 40, hi[0] + 40)
    axt.set_ylim(lo[1] - 40, hi[1] + 40)
    axt.set_aspect("equal")
    axt.set_xlabel("x [mm]")
    axt.set_ylabel("y [mm]")
    axt.set_title(f"チャンバー内推定位置 上面図 (内寸 {CHAMBER_SIZE[0]:.0f}×"
                  f"{CHAMBER_SIZE[1]:.0f} mm, 原点=床中心)")
    axt.grid(alpha=0.3)
    axt.legend(fontsize=8, loc="upper right")
    fig.colorbar(sc, ax=axt, pad=0.02, shrink=0.85, label="time [s]")

    # --- 壁フィックスが得られた点だけを重ねた上面図 ---
    axf = fig.add_subplot(gs[0, 2])
    axf.add_patch(plt.Rectangle((lo[0], lo[1]), CHAMBER_SIZE[0], CHAMBER_SIZE[1],
                                fill=False, ec="0.5", lw=1.5))
    both = (src[:, 0] == 1) & (src[:, 1] == 1)
    only = ((src[:, 0] == 1) ^ (src[:, 1] == 1))
    none = (src[:, 0] == 0) & (src[:, 1] == 0)
    axf.scatter(pos[none, 0], pos[none, 1], s=12, color="0.6", marker="x",
                label=f"x,y とも補完 ({comp_name})")
    axf.scatter(pos[only, 0], pos[only, 1], s=18, color="tab:orange",
                edgecolors="k", linewidths=0.3, label="片軸のみ壁フィックス")
    axf.scatter(pos[both, 0], pos[both, 1], s=22, color="tab:blue",
                edgecolors="k", linewidths=0.3, label="x,y とも壁フィックス")
    axf.set_xlim(lo[0] - 40, hi[0] + 40)
    axf.set_ylim(lo[1] - 40, hi[1] + 40)
    axf.set_aspect("equal")
    axf.set_xlabel("x [mm]")
    axf.set_ylabel("y [mm]")
    axf.set_title("位置の出所 (壁フィックス / 補完)")
    axf.grid(alpha=0.3)
    axf.legend(fontsize=7, loc="upper right")

    # --- 各軸の時系列 (壁フィックス / 補完 を塗り分け) ---
    names = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax_i in range(3):
        a = fig.add_subplot(gs[1, ax_i])
        a.axhline(lo[ax_i], color="0.5", lw=1)
        a.axhline(hi[ax_i], color="0.5", lw=1)
        a.plot(t, pos[:, ax_i], lw=1.0, color="0.6", label="融合結果")
        m = src[:, ax_i] == 1
        a.scatter(t[m], pos[m, ax_i], s=14, color=colors[ax_i],
                  edgecolors="k", linewidths=0.3, zorder=4, label="壁フィックス")
        a.scatter(t[~m], pos[~m, ax_i], s=8, color="0.35", marker="x",
                  zorder=3, label=f"補完 ({comp_name})")
        a.set_xlabel("time [s]")
        a.set_ylabel(f"{names[ax_i]} [mm]")
        a.set_ylim(lo[ax_i] - 40, hi[ax_i] + 40)
        draw_hover_lines(a, hover)
        a.set_title(f"{names[ax_i]} 座標 "
                    f"(壁フィックス {int(m.sum())}/{len(t)} フレーム)")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"チャンバー位置グラフを保存: {out_path}")


def plot_wall_detection(fixes, res, sensors, att, yaw_offset, t, out_path,
                        hover=()):
    """どの壁をいつ見ていたか / 壁フィックス数と姿勢の内訳を可視化する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _use_jp_font()
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # (0) センサーごとの「見ている壁」
    labels = {(0, 1): "+x", (0, -1): "-x", (1, 1): "+y", (1, -1): "-y",
              (2, 1): "+z(天井)", (2, -1): "-z(床)"}
    ykeys = list(labels.keys())
    Rz_off = rot_ypr(-yaw_offset, 0.0, 0.0)
    prop = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cos_tol = np.cos(np.radians(WALL_ANGLE_TOL_DEG))
    for i, s in enumerate(sensors):
        xs, ys = [], []
        for f in range(len(t)):
            n = res["plane_n"][s][f]
            if not np.isfinite(n).all() or res["gated"][s][f]:
                continue
            nw = Rz_off @ rot_ypr(*att[f]) @ n
            axis = int(np.argmax(np.abs(nw)))
            if abs(nw[axis]) < cos_tol:
                continue
            key = (axis, 1 if nw[axis] > 0 else -1)
            xs.append(t[f])
            ys.append(ykeys.index(key) + 0.06 * (i - len(sensors) / 2))
        ax[0].scatter(xs, ys, s=16, color=prop[i % len(prop)], label=f"sensor{s}")
    ax[0].set_yticks(range(len(ykeys)))
    ax[0].set_yticklabels([labels[k] for k in ykeys])
    ax[0].set_title(f"各センサーが見ている壁 (ヨーオフセット {yaw_offset:+.1f} deg で補正)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=3)

    # (1) 軸ごとの壁フィックス数
    for i, nm in enumerate(["x", "y", "z"]):
        ax[1].plot(t, fixes["n_fix"][:, i], lw=1, label=f"{nm} 軸")
    ax[1].set_ylabel("壁フィックス数 / フレーム")
    ax[1].set_title("各軸で位置が確定した面の数 (0 = 補完データで埋めた区間)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    # (2) EKF 姿勢
    for i, nm in enumerate(["yaw", "pitch", "roll"]):
        ax[2].plot(t, att[:, i], lw=1, label=f"EKF {nm}")
    ax[2].set_ylabel("attitude [deg]")
    ax[2].set_xlabel("time [s]")
    ax[2].set_title("① 融合に使った姿勢 (EKF: stabilizer.yaw/pitch/roll)")
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=8)

    for a in ax:
        draw_hover_lines(a, hover, label=False)
    if hover:
        ax[0].plot([], [], color="k", lw=1.2, label="ホバリング開始/終了")
        ax[0].legend(fontsize=7, ncol=4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"壁検出グラフを保存: {out_path}")


# ============================================================
# main
# ============================================================
def main():
    global CHAMBER_SIZE, WALL_ANGLE_TOL_DEG
    ap = argparse.ArgumentParser(
        description="チャンバー内自己位置推定 (ToF平面 + EKF姿勢の融合)")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log51.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+",
                    default=[0, 1, 2, 5, 6, 7],
                    help="使用するセンサー番号 (既定: 0 1 2 5 6 7 = 水平リング)")
    ap.add_argument("--size", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help=f"チャンバー内寸 [mm] (既定: {CHAMBER_SIZE.tolist()})")
    ap.add_argument("--eps", type=float, default=None,
                    help=f"平面性スコアの閾値 (既定: {PLANARITY_EPS})")
    ap.add_argument("--gate-deg", type=float, default=GATE_DEG,
                    help="姿勢整合チェックの閾値[deg]。0で無効")
    ap.add_argument("--wall-tol-deg", type=float, default=WALL_ANGLE_TOL_DEG,
                    help="法線を壁に割り当てる許容角[deg]")
    ap.add_argument("--yaw-offset", type=float, default=None,
                    help="機体ヨー基準とチャンバーの角度差[deg] (既定: 自動推定)")
    ap.add_argument("--no-gate-reject", action="store_true",
                    help="方位角ゲートで棄却された法線も壁フィックスに使う")
    ap.add_argument("-o", "--output", default=None, help="出力PNGパス")
    ap.add_argument("--csv-out", default=None, help="推定位置のCSV出力パス")
    ap.add_argument("--seed", type=int, default=0, help="RANSACの乱数シード")
    args = ap.parse_args()

    import pandas as pd
    if args.size is not None:
        CHAMBER_SIZE = np.array(args.size, dtype=float)
    WALL_ANGLE_TOL_DEG = args.wall_tol_deg

    path = args.csv
    if not os.path.isfile(path):
        cand = os.path.join(CSV_DIR, os.path.basename(path))
        if not os.path.isfile(cand):
            sys.exit(f"ファイルが見つかりません: {args.csv}")
        path = cand
    df = pd.read_csv(path)

    if args.eps is not None:
        tof.PLANARITY_EPS = args.eps

    rng = np.random.default_rng(args.seed)
    arrs = sensor_arrays(df, args.sensors)
    idx, fresh = tof_frames(arrs, args.sensors,
                            df["timestamp"].to_numpy(float) / 1000.0)
    planes, plane_stats, _fail_f, _ok_f = extract_planes(
        df, args.sensors, rng, arrs, idx, fresh)

    df = df.iloc[idx].reset_index(drop=True)
    t_s = df["timestamp"].to_numpy(float) / 1000.0

    # ① 融合に使う姿勢は EKF (stabilizer.*)
    att = ekf_attitude(df)
    if att is None:
        sys.exit(f"EKF姿勢の列がありません: {ATT_COLS}")

    res = run_odometry(planes, args.sensors, df["timestamp"].to_numpy(),
                       att=att, gate_deg=args.gate_deg)

    # ② 壁からの絶対位置 + 補完
    yaw_offset = (args.yaw_offset if args.yaw_offset is not None
                  else yaw_offset_from_walls(res, args.sensors, att))
    fixes = wall_fixes(res, args.sensors, att, yaw_offset, df=df,
                       tol_deg=WALL_ANGLE_TOL_DEG,
                       use_gate=not args.no_gate_reject)
    comp, comp_name = complement_track(df, t_s, att)
    fuse = fuse_position(fixes, comp)

    # サマリ
    F = len(df)
    print(f"入力: {path}  ({F} フレーム)")
    print(f"  チャンバー内寸 [mm] : {CHAMBER_SIZE.tolist()} "
          f"(原点 = 高さ0のチャンバー中心, 離陸前は (0,0,0))")
    print(f"  姿勢ソース          : EKF (stabilizer.yaw/pitch/roll)")
    print(f"  補完データ          : {comp_name}")
    print(f"  ヨーオフセット      : {yaw_offset:+.2f} deg")
    for s in args.sensors:
        ok = np.isfinite(res["plane_d"][s]).sum()
        print(f"  sensor{s}: 平面抽出成功 {ok}/{F} ({100.0*ok/F:.1f}%), "
              f"方位角ゲート棄却 {int(res['gated'][s].sum())} フレーム")
    for i, nm in enumerate(["x", "y", "z"]):
        n_fix = int((fuse["src"][:, i] == 1).sum())
        print(f"  {nm}: 壁フィックス {n_fix}/{F} ({100.0*n_fix/F:.1f}%), "
              f"範囲 {fuse['pos'][:, i].min():.0f}〜{fuse['pos'][:, i].max():.0f} mm")
    print(f"  推定位置 開始 → 終了 [mm] : "
          f"{fuse['pos'][0].round(0)} → {fuse['pos'][-1].round(0)}")

    # ③ 可視化
    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_chamber.png")
    t_rel = t_s - t_s[0]
    hover = hover_spans(df, t_rel)
    if hover:
        print(f"  ホバリング区間 (chamber.state=={HOVER_STATE}) : "
              + ", ".join(f"{a:.1f}〜{b:.1f}s" for a, b in hover))
    plot_chamber(fuse, fixes, res, att, t_rel, comp_name, out, args.sensors,
                 hover=hover)
    plot_wall_detection(fixes, res, args.sensors, att, yaw_offset, t_rel,
                        os.path.join(OUT_DIR, f"{base}_chamber_walls.png"),
                        hover=hover)

    if args.csv_out:
        cols = dict(timestamp=df["timestamp"].to_numpy(), t=t_rel)
        for i, nm in enumerate(["x", "y", "z"]):
            cols[nm] = fuse["pos"][:, i]
            cols[f"{nm}_fix"] = fixes["fix"][:, i]
            cols[f"{nm}_src"] = fuse["src"][:, i]
            cols[f"{nm}_comp"] = comp[:, i]
        for i, nm in enumerate(["yaw", "pitch", "roll"]):
            cols[f"ekf_{nm}"] = att[:, i]
        if STATE_COL in df.columns:
            cols["chamber_state"] = df[STATE_COL].to_numpy()
        for s in args.sensors:
            cols[f"d_s{s}"] = res["plane_d"][s]
        pd.DataFrame(cols).to_csv(args.csv_out, index=False)
        print(f"推定位置CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
