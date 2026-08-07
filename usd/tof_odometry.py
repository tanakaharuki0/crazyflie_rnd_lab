#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超低密度ToF点群(VL53L8CX 4x4 = 16点/センサー)からの
移動量(Δt)・回転量(ΔR)の幾何学的抽出アルゴリズム。

"Drone Sparse ToF Odometry Research" の 3章 に基づく実装:

  3.1 PCA + RANSAC による平面抽出
      重心 p̄ = (1/N)Σ p_k, 共分散 C = (1/N)Σ (p_k-p̄)(p_k-p̄)^T
      C の固有値分解 λ1≥λ2≥λ3 → 最小固有値の固有ベクトル v3 = 法線 n_i
      平面性スコア λ3/(λ1+λ2+λ3) < ε なら平面とみなす
      センサーから平面までの垂直距離 d_i = n_i^T p̄

  3.2 法線ベクトル群からの回転量 ΔR の推定 (Kabsch / SVD)
      ΔR = argmin_{R∈SO(3)} Σ w_i ||n'_i - R n_i||^2

  3.3 Point-to-Plane 誤差モデルによる移動量 Δt の推定
      d'_i ≈ d_i - n_i^T Δt
      Δt = argmin_{Δt∈R^3} Σ w_i (d'_i - (d_i - n_i^T Δt))^2
      → 廊下等での退化(ランク落ち)は truncated SVD で観測可能な部分空間のみ解く

使い方:
    python3 tof_odometry.py log39.csv                 # sensor 0,1,2 を使用
    python3 tof_odometry.py log39.csv -s 0 1 2 -o out.png
    python3 tof_odometry.py log39.csv --csv-out odom39.csv
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "figs")

# ---- VL53L8CX の仕様 ----
FOV_DEG = 45.0          # 正方FOV (対角65度 ≒ 片辺45度)
GRID = 4                # 4x4 = 16 ゾーン
D_MIN_MM = 30.0         # 有効とみなす距離の下限
D_MAX_MM = 4000.0       # 有効とみなす距離の上限

# ---- 平面抽出パラメータ ----
RANSAC_ITERS = 60
RANSAC_THRESH_MM = 20.0   # 平面までの許容残差
MIN_INLIERS = 6           # 16点中この数以上のインライアが必要
PLANARITY_EPS = 0.006     # λ3/(λ1+λ2+λ3) の閾値 (tof_odometry.c と同値)
GATE_DEG = 1.0            # ジャイロ整合チェックの閾値[deg]。0で無効

# ---- センサー取り付け (docs/vl53l8cx-position.md, 位置[mm]/角度[deg]) ----
# 光軸はセンサー座標系の +x、+y が左、+z が上。yaw-pitch-roll (Z-Y-X) でボディ系へ。
SENSOR_MOUNT = {
    0:  dict(t=(56.6, -9.5, 0.0), ypr=(-22.5, 0.0, 0.0)),
    1:  dict(t=(59.2, 0.0, 0.0),   ypr=(0.0, 0.0, 0.0)),
    2:  dict(t=(56.6, 9.5, 0.0),  ypr=(22.5, 0.0, 0.0)),
    3:  dict(t=(56.6, 0.0, 9.5),   ypr=(0.0, 22.5, 0.0)),
    4:  dict(t=(56.6, 0.0, -9.5),  ypr=(0.0, -22.5, 0.0)),
    5:  dict(t=(0.0, -21.7, 0.0),  ypr=(-90.0, 0.0, 0.0)),
    6:  dict(t=(0.0, 21.7, 0.0),   ypr=(90.0, 0.0, 0.0)),
    7:  dict(t=(-59.2, 0.0, 0.0),  ypr=(180.0, 0.0, 0.0)),
    8:  dict(t=(-2.0, 0.0, 4.2),   ypr=(0.0, -90.0, 0.0)),
    9:  dict(t=(28.5, 0.0, -4.2),   ypr=(0.0, 90.0, 0.0)),
    10: dict(t=(-37.5, 0.0, -4.2),  ypr=(0.0, 90.0, 0.0)),
}


# ============================================================
# 幾何ユーティリティ
# ============================================================
def rot_ypr(yaw_deg, pitch_deg, roll_deg):
    """Z-Y-X (yaw→pitch→roll) 順の回転行列。"""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def rot_to_ypr(R):
    """回転行列 → (yaw, pitch, roll) [deg] (Z-Y-X)。"""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.cos(pitch) > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:  # ジンバルロック
        yaw = 0.0
        roll = np.arctan2(-R[1, 2], R[1, 1])
    return np.degrees([yaw, pitch, roll])


def zone_rays():
    """4x4 の各ゾーン中心方向の単位ベクトル (センサー座標系) を zone順(0..15)で返す。

    zone i → row = i//4, col = i%4。depth_map.py と同じく上下左右反転して
    物理的な向き (row↑ = 上, col↑ = 左) に直してから角度を割り当てる。

    注: 列の反転有無は法線の方位角の符号を反転させる。log39 で 4通りの
    反転規約を試したところ、sensor0/1/2 の法線の平均相互角は
      flip_col=1 で 42度、flip_col=0 で 78度 (行の反転はほぼ影響なし)
    となり、この規約が最も自己整合的だった (取り付け角の平均相互角は 60度)。
    """
    step = np.radians(FOV_DEG) / GRID
    rays = np.zeros((GRID * GRID, 3))
    for i in range(GRID * GRID):
        row, col = i // GRID, i % GRID
        r = (GRID - 1) - row   # 反転後の行 (大きいほど上)
        c = (GRID - 1) - col   # 反転後の列 (大きいほど左)
        az = (c - (GRID - 1) / 2.0) * step   # +y (左) 向きが正
        el = (r - (GRID - 1) / 2.0) * step   # +z (上) 向きが正
        rays[i] = [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    return rays


ZONE_RAYS = zone_rays()


def points_body(dist_mm, sensor):
    """1センサー16ゾーンの距離[mm] → ボディ座標系の点群 (有効点のみ)。"""
    mount = SENSOR_MOUNT[sensor]
    R = rot_ypr(*mount["ypr"])
    t = np.asarray(mount["t"], dtype=float)

    d = np.asarray(dist_mm, dtype=float)
    valid = np.isfinite(d) & (d > D_MIN_MM) & (d < D_MAX_MM)
    if valid.sum() == 0:
        return np.zeros((0, 3)), valid

    p_sensor = ZONE_RAYS[valid] * d[valid, None]   # 光線方向 × 距離
    return (R @ p_sensor.T).T + t, valid


# ============================================================
# 3.1 PCA + RANSAC 平面抽出
# ============================================================
def pca_plane(P):
    """点群 P (N,3) から PCA で平面を求める。

    Returns: (n, d, planarity, eigvals) — n は原点側を向く単位法線、d = n^T p̄ > 0。
    """
    pbar = P.mean(axis=0)
    Q = P - pbar
    C = (Q.T @ Q) / len(P)
    w, V = np.linalg.eigh(C)          # 昇順
    n = V[:, 0]                        # 最小固有値の固有ベクトル = 法線
    lam = w[::-1]                      # λ1 ≥ λ2 ≥ λ3
    planarity = lam[2] / max(lam.sum(), 1e-12)

    d = float(n @ pbar)
    if d < 0:                          # 法線の向きを「原点→平面」に統一
        n, d = -n, -d
    return n, d, float(planarity), lam


def fit_plane_ransac(P, rng):
    """RANSAC で外れ値を除いてから PCA 平面フィット。

    Returns: dict(n, d, planarity, n_inliers, rms) または None。
    """
    N = len(P)
    if N < MIN_INLIERS:
        return None

    best_inl = None
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(N, 3, replace=False)
        a, b, c = P[idx]
        nrm = np.cross(b - a, c - a)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm /= norm
        resid = np.abs((P - a) @ nrm)
        inl = resid < RANSAC_THRESH_MM
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
            if inl.sum() == N:
                break

    if best_inl is None or best_inl.sum() < MIN_INLIERS:
        return None

    # インライアで最終 PCA (2回リファイン)
    inl = best_inl
    for _ in range(2):
        n, d, planarity, _ = pca_plane(P[inl])
        resid = np.abs(P @ n - d)
        new_inl = resid < RANSAC_THRESH_MM
        if new_inl.sum() < MIN_INLIERS:
            break
        inl = new_inl

    n, d, planarity, _ = pca_plane(P[inl])
    if planarity > PLANARITY_EPS:
        return None

    rms = float(np.sqrt(np.mean((P[inl] @ n - d) ** 2)))
    return dict(n=n, d=d, planarity=planarity, n_inliers=int(inl.sum()), rms=rms)


# ============================================================
# 3.2 ΔR (Kabsch), 3.3 Δt (Point-to-Plane 最小二乗)
# ============================================================
def kabsch_normals(N_prev, N_curr, w):
    """ΔR = argmin_R Σ w_i ||n'_i - R n_i||^2 を SVD で解く。"""
    H = (N_prev * w[:, None]).T @ N_curr          # Σ w n n'^T
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ U.T))])
    R = V @ D @ U.T
    # 観測された法線方向の張る空間のランク (2未満なら回転は一意でない)
    rank = int((S > 1e-6 * max(S[0], 1e-12)).sum())
    return R, rank


def solve_translation(Ns, dd, w, rcond=0.05):
    """Σ w_i (Δd_i + n_i^T Δt)^2 の最小化。退化方向は truncated SVD で 0 とする。

    Ns: (M,3) 法線, dd: (M,) d'_i - d_i, w: (M,) 重み。
    Returns: (Δt, rank, 観測可能方向の基底)
    """
    sw = np.sqrt(w)
    A = Ns * sw[:, None]
    b = -dd * sw
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    keep = S > rcond * max(S[0], 1e-12)
    rank = int(keep.sum())
    if rank == 0:
        return np.zeros(3), 0, np.zeros((0, 3))
    dt = Vt[keep].T @ ((U[:, keep].T @ b) / S[keep])
    return dt, rank, Vt[keep]


# ============================================================
# メイン処理
# ============================================================
def sensor_arrays(df, sensors):
    """センサーごとの 16ゾーン距離 (N,16) を取り出す。"""
    arrs = {}
    for s in sensors:
        cols = [f"vl53l8cx_s{s}.d{i}" for i in range(16)]
        missing = [x for x in cols if x not in df.columns]
        if missing:
            sys.exit(f"sensor{s} の列がありません: {missing[:3]} ...")
        arrs[s] = df[cols].to_numpy(dtype=float)
    return arrs


# 1スイープとみなす時間幅 [s]。測距周期 (約200ms) より十分短くとる。
SWEEP_WINDOW_S = 0.10


def tof_frames(arrs, sensors, t):
    """
    ToF の1スイープを1フレームとして、その index と、各フレームでどのセンサーが
    更新されたかのマスクを返す。

    ログは 50Hz 前後だが ToF の測距は 5Hz なので、同じ距離データが約10行続く。
    全行をフレームとして扱うと「一つ前の計測」が自分自身になり Δ=0 だらけになる。
    さらに1スイープの SPI 読み出しは数行 (約40ms) に分散する (例: [0,1,2]→[5]→[6,7])
    ので、単に「どれかが更新された行」を拾うとフレームが分裂し、1フレームあたり
    1〜3個のセンサーしか揃わない。SWEEP_WINDOW_S 以内の更新は同一スイープとして
    まとめ、そのスイープの最終行 (全センサーの新しい値が入っている行) を採る。
    """
    n = len(next(iter(arrs.values())))
    changed = {s: np.zeros(n, dtype=bool) for s in sensors}
    for s in sensors:
        d = np.abs(np.diff(arrs[s], axis=0)).sum(axis=1) != 0
        changed[s][1:] = d
        changed[s][0] = True          # 先頭は初回取得とみなす

    any_changed = np.zeros(n, dtype=bool)
    for s in sensors:
        any_changed |= changed[s]
    rows = np.flatnonzero(any_changed)
    if len(rows) == 0:
        sys.exit("ToF の距離データが一度も更新されていません。")

    # 時間的に近い更新行を1スイープにまとめる
    brk = np.flatnonzero(np.diff(t[rows]) > SWEEP_WINDOW_S)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk, [len(rows) - 1]])

    idx = rows[ends]                                  # 各スイープの最終行
    fresh = {s: np.zeros(len(idx), dtype=bool) for s in sensors}
    for k, (a, b) in enumerate(zip(starts, ends)):
        for s in sensors:
            fresh[s][k] = changed[s][rows[a]:rows[b] + 1].any()
    return idx, fresh


def extract_planes(df, sensors, rng, arrs=None, idx=None, fresh=None):
    """全フレーム・全センサーの平面抽出。planes[f][s] = dict or None。

    idx/fresh を渡すと、そのセンサーの測距が更新されたフレームだけを計算する
    (更新のないフレームは None = 古いデータを使い回さない)。
    """
    if arrs is None:
        arrs = sensor_arrays(df, sensors)
    if idx is None:
        idx = np.arange(len(df))
        fresh = {s: np.ones(len(idx), dtype=bool) for s in sensors}

    planes = []
    for f, row in enumerate(idx):
        frame = {}
        for s in sensors:
            if not fresh[s][f]:
                frame[s] = None
                continue
            P, _ = points_body(arrs[s][row], s)
            frame[s] = fit_plane_ransac(P, rng) if len(P) >= MIN_INLIERS else None
        planes.append(frame)
    return planes


def run_odometry(planes, sensors, timestamps, att=None, gate_deg=GATE_DEG):
    """連続する「そのセンサー自身の計測」間で ΔR / Δt を推定し、累積を返す。

    att: (F,3) の (yaw,pitch,roll)[deg]。渡すとファームウェアと同じ
         ジャイロ整合チェックを行い、
           - 個々の法線がジャイロ予測と gate_deg 以上ずれた対応を捨てる
           - まとめた ΔR がジャイロと gate_deg 以上ずれたら ΔR ごと捨てて
             ジャイロの回転で代替する
    """
    F = len(planes)
    out = dict(
        t=np.asarray(timestamps, dtype=float) / 1000.0,   # [s]
        dR_ypr=np.full((F, 3), np.nan),
        dt=np.full((F, 3), np.nan),
        rank_R=np.zeros(F, dtype=int),
        rank_t=np.zeros(F, dtype=int),
        n_pairs=np.zeros(F, dtype=int),
        n_gated=np.zeros(F, dtype=int),
        rot_err=np.full(F, np.nan),
        rot_src=np.zeros(F, dtype=int),   # 0:なし 1:ToF 2:ジャイロ(乖離) 3:ジャイロ(対応なし)
        ypr=np.zeros((F, 3)),
        pos=np.zeros((F, 3)),
        plane_d={s: np.full(F, np.nan) for s in sensors},
        plane_n={s: np.full((F, 3), np.nan) for s in sensors},
        planarity={s: np.full(F, np.nan) for s in sensors},
    )

    for f in range(F):
        for s in sensors:
            pl = planes[f][s]
            if pl is not None:
                out["plane_d"][s][f] = pl["d"]
                out["plane_n"][s][f] = pl["n"]
                out["planarity"][s][f] = pl["planarity"]

    # 姿勢 (world_R_body) をフレームごとに用意
    R_att = None
    if att is not None:
        R_att = [rot_ypr(*att[f]) for f in range(F)]

    # センサーごとに「一つ前の自分の計測」を保持する
    prev = {s: None for s in sensors}      # (plane, R_att or None)
    R_cum = np.eye(3)
    p_cum = np.zeros(3)
    R_last = R_att[0] if R_att is not None else None

    for f in range(F):
        pairs, n_gated = [], 0
        for s in sensors:
            cur = planes[f][s]
            if cur is None:
                continue
            pv = prev[s]
            if pv is not None and pv[0] is not None:
                ok = True
                if gate_deg > 0 and R_att is not None and pv[1] is not None:
                    # 壁が固定なら n_curr = R_curr^T R_prev n_prev
                    n_pred = R_att[f].T @ pv[1] @ pv[0]["n"]
                    c = float(np.clip(n_pred @ cur["n"], -1.0, 1.0))
                    if np.degrees(np.arccos(c)) >= gate_deg:
                        ok = False
                        n_gated += 1
                if ok:
                    pairs.append((s, pv[0], cur))
            prev[s] = (cur, R_att[f] if R_att is not None else None)

        out["n_pairs"][f] = len(pairs)
        out["n_gated"][f] = n_gated

        # ジャイロによる機体回転 (前回このループを通ったとき → 今回)
        dR_gyro = None
        if R_att is not None and R_last is not None:
            dR_gyro = R_last.T @ R_att[f]      # body_prev_R_body_curr

        dRb, src = None, 0
        if pairs:
            N_prev = np.array([p[1]["n"] for p in pairs])
            N_curr = np.array([p[2]["n"] for p in pairs])
            w = np.array([1.0 / max(p[2]["planarity"], 1e-6) for p in pairs])
            w /= w.sum()

            dR, rank_R = kabsch_normals(N_prev, N_curr, w)
            out["rank_R"][f] = rank_R
            if rank_R < 2:        # 法線が1方向のみ → 回転は不定なので採用しない
                dR = np.eye(3)
            # Kabsch は法線を prev→curr に写す回転 (= body_curr_R_body_prev) を返す。
            # 機体自身の回転はその逆なので転置し、ジャイロと同じ向きに揃える。
            dRb = dR.T
            src = 1

            if dR_gyro is not None and gate_deg > 0:
                c = float(np.clip((np.trace(dRb.T @ dR_gyro) - 1.0) / 2.0, -1.0, 1.0))
                out["rot_err"][f] = np.degrees(np.arccos(c))
                if out["rot_err"][f] >= gate_deg:
                    dRb, src = dR_gyro, 2

            dd = np.array([p[2]["d"] - p[1]["d"] for p in pairs])
            dt, rank_t, _ = solve_translation(N_prev, dd, w)
            out["dt"][f] = dt
            out["rank_t"][f] = rank_t
            p_cum = p_cum + R_cum @ dt      # 姿勢を進める前の R_cum で回す
        elif dR_gyro is not None:
            dRb, src = dR_gyro, 3

        if dRb is not None:
            out["dR_ypr"][f] = rot_to_ypr(dRb)
            R_cum = R_cum @ dRb
        out["rot_src"][f] = src

        if R_att is not None:
            R_last = R_att[f]
        out["ypr"][f] = rot_to_ypr(R_cum)
        out["pos"][f] = p_cum

    return out


GYRO_COLS = ("gyro.x", "gyro.y", "gyro.z")   # roll/pitch/yaw の角速度 [deg/s]
GYRO_BIAS_SEC = 0.5                          # 先頭この秒数の平均をバイアスとして除去


def gyro_reference(df, t):
    """ジャイロ角速度から比較用の Δ回転[deg/frame] と累積回転[deg] を作る。

    gyro.x/y/z = roll/pitch/yaw 方向の角速度 [deg/s]。
    ToF 側と同じ (yaw, pitch, roll) の順に並べ替えて返す。列が無ければ None。
    """
    if not all(c in df.columns for c in GYRO_COLS):
        return None

    # [roll, pitch, yaw] rate → [yaw, pitch, roll] rate
    rate = df[list(GYRO_COLS)].to_numpy(dtype=float)[:, ::-1]

    # 静止しているはずの先頭区間の平均をバイアスとして差し引く
    n_bias = max(int(np.searchsorted(t - t[0], GYRO_BIAS_SEC)), 1)
    rate = rate - np.nanmean(rate[:n_bias], axis=0)

    dt_s = np.diff(t, prepend=t[0])                   # 各フレームの経過時間 [s]
    d_ypr = rate * dt_s[:, None]                      # [deg/frame]
    return dict(d_ypr=d_ypr, ypr=np.nancumsum(d_ypr, axis=0))


def plot(res, sensors, df, out_path, tmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 日本語ラベル用フォント (無ければ既定のまま)
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "IPAGothic", "TakaoGothic", "VL Gothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False

    t = res["t"]
    gyro = gyro_reference(df, t)
    t = t - t[0]
    fig, ax = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
    ypr_colors = ["tab:blue", "tab:orange", "tab:green"]

    # (0,0) 各センサーの平面距離 d_i
    for s in sensors:
        ax[0, 0].plot(t, res["plane_d"][s], lw=1, label=f"sensor{s}")
    ax[0, 0].set_ylabel("plane distance $d_i$ [mm]")
    ax[0, 0].set_title("3.1 PCA+RANSAC 平面抽出: センサー原点→平面の垂直距離")
    ax[0, 0].legend(fontsize=8)

    # (0,1) 平面性スコア
    for s in sensors:
        ax[0, 1].plot(t, res["planarity"][s], lw=1, label=f"sensor{s}")
    ax[0, 1].axhline(PLANARITY_EPS, color="k", ls="--", lw=0.8, label="ε")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel(r"planarity $\lambda_3/\Sigma\lambda$")
    ax[0, 1].set_title("平面性スコア (小さいほど良い平面)")
    ax[0, 1].legend(fontsize=8)

    # (1,0) フレーム間回転量
    for i, name in enumerate(["yaw", "pitch", "roll"]):
        ax[1, 0].plot(t, res["dR_ypr"][:, i], lw=1, color=ypr_colors[i],
                      label=f"ToF Δ{name}")
    if gyro is not None:
        for i, name in enumerate(["yaw", "pitch", "roll"]):
            ax[1, 0].plot(t, gyro["d_ypr"][:, i], lw=0.8, ls="--", alpha=0.6,
                          color=ypr_colors[i], label=f"gyro Δ{name}")
    ax[1, 0].set_ylabel("Δrotation [deg/frame]")
    ax[1, 0].set_title("3.2 Kabsch による回転量 ΔR (フレーム間)"
                       + (" / ジャイロ角速度の積" if gyro is not None else ""))
    ax[1, 0].legend(fontsize=7, ncol=2)

    # (1,1) フレーム間移動量
    for i, name in enumerate(["x", "y", "z"]):
        ax[1, 1].plot(t, res["dt"][:, i], lw=1, label=f"Δ{name}")
    ax[1, 1].set_ylabel("Δtranslation [mm/frame]")
    ax[1, 1].set_title("3.3 Point-to-Plane による移動量 Δt (フレーム間)")
    ax[1, 1].legend(fontsize=8)

    # (2,0) 累積姿勢
    for i, name in enumerate(["yaw", "pitch", "roll"]):
        ax[2, 0].plot(t, res["ypr"][:, i], lw=1.2, color=ypr_colors[i],
                      label=f"ToF {name}")
    if gyro is not None:
        for i, name in enumerate(["yaw", "pitch", "roll"]):
            ax[2, 0].plot(t, gyro["ypr"][:, i], lw=1.0, ls="--", alpha=0.75,
                          color=ypr_colors[i], label=f"gyro {name}")
    ax[2, 0].set_ylabel("cumulative rotation [deg]")
    ax[2, 0].set_title("累積回転量 (ΔR の積分)"
                       + (" と ジャイロ積分の比較" if gyro is not None else ""))
    ax[2, 0].legend(fontsize=7, ncol=2)

    # (2,1) 累積移動量 (ToF) vs stateEstimate
    for i, name in enumerate(["x", "y", "z"]):
        ax[2, 1].plot(t, res["pos"][:, i], lw=1.2, label=f"ToF {name}")
    for i, name in enumerate(["x", "y", "z"]):
        col = f"stateEstimate.{name}"
        if col in df.columns:
            ax[2, 1].plot(t, (df[col].to_numpy() - df[col].to_numpy()[0]) * 1000.0,
                          lw=1, ls="--", alpha=0.7, label=f"EKF {name}")
    ax[2, 1].set_ylabel("cumulative translation [mm]")
    ax[2, 1].set_title("累積移動量 (Δt の積分) と EKF の比較")
    ax[2, 1].legend(fontsize=7, ncol=2)

    # (3,0) 観測可能性 (ランク)
    ax[3, 0].plot(t, res["rank_t"], lw=1, label="rank(Δt)")
    ax[3, 0].plot(t, res["rank_R"], lw=1, label="rank(ΔR)")
    ax[3, 0].plot(t, res["n_pairs"], lw=1, ls=":", label="有効センサー数")
    ax[3, 0].set_ylabel("rank / count")
    ax[3, 0].set_xlabel("time [s]")
    ax[3, 0].set_title("幾何学的退化の診断 (rank<3 = その軸は不定)")
    ax[3, 0].legend(fontsize=8)

    # (3,1) 法線の向き (yaw成分) — 壁に対する相対角度
    for s in sensors:
        n = res["plane_n"][s]
        ax[3, 1].plot(t, np.degrees(np.arctan2(n[:, 1], n[:, 0])), lw=1,
                      label=f"sensor{s}")
    ax[3, 1].set_ylabel("normal azimuth [deg]")
    ax[3, 1].set_xlabel("time [s]")
    ax[3, 1].set_title("抽出された法線ベクトルの方位角 (ボディ座標系)")
    ax[3, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.grid(alpha=0.3)

    # 表示範囲を絞る (着地時の外れ値でスケールが潰れるのを防ぐ)。
    # 推定自体は全フレームで行い、表示だけを制限する。
    if tmax is not None:
        keep = t <= tmax
        for a in ax.ravel():
            a.set_xlim(t[0], tmax)
            a.relim(visible_only=True)
            a.autoscale(axis="y")
        for a, lines in ((a, a.get_lines()) for a in ax.ravel()):
            ys = np.concatenate([ln.get_ydata()[keep] for ln in lines
                                 if len(ln.get_ydata()) == len(t)] or [np.array([np.nan])])
            if np.isfinite(ys).any():
                lo, hi = np.nanmin(ys), np.nanmax(ys)
                pad = 0.05 * max(hi - lo, 1e-9)
                a.set_ylim(lo - pad, hi + pad)

    # 法線方位角は 10度刻みで目盛を振り、薄い灰色の点線を引く
    ax[3, 1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax[3, 1].grid(axis="y", which="major", color="0.6", ls=":", lw=0.8, alpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"グラフを保存: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="超低密度ToF点群からの移動量・回転量抽出")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log39.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+",
                    default=[0, 1, 2, 5, 6, 7],
                    help="使用するセンサー番号 (既定: 0 1 2 5 6 7 = 水平リング)")
    ap.add_argument("--eps", type=float, default=None,
                    help=f"平面性スコアの閾値 (既定: {PLANARITY_EPS})")
    ap.add_argument("--gate-deg", type=float, default=GATE_DEG,
                    help="ジャイロ整合チェックの閾値[deg]。0で無効")
    ap.add_argument("--no-dedup", action="store_true",
                    help="ToF更新行での間引きをせず、CSVの全行をフレームとして扱う")
    ap.add_argument("-o", "--output", default=None, help="出力PNGパス")
    ap.add_argument("--csv-out", default=None, help="推定結果のCSV出力パス")
    ap.add_argument("--seed", type=int, default=0, help="RANSACの乱数シード")
    ap.add_argument("--tmax", type=float, default=None,
                    help="グラフの表示上限時刻 [s]。推定は全フレームで行い表示だけ制限する")
    args = ap.parse_args()

    import pandas as pd
    path = args.csv
    if not os.path.isfile(path):
        cand = os.path.join(CSV_DIR, os.path.basename(path))
        if not os.path.isfile(cand):
            sys.exit(f"ファイルが見つかりません: {args.csv}")
        path = cand
    df = pd.read_csv(path)

    if args.eps is not None:
        globals()["PLANARITY_EPS"] = args.eps

    rng = np.random.default_rng(args.seed)
    arrs = sensor_arrays(df, args.sensors)
    if args.no_dedup:
        idx = np.arange(len(df))
        fresh = {s: np.ones(len(df), dtype=bool) for s in args.sensors}
    else:
        idx, fresh = tof_frames(arrs, args.sensors,
                                df["timestamp"].to_numpy(float) / 1000.0)

    planes = extract_planes(df, args.sensors, rng, arrs, idx, fresh)

    # ジャイロ整合チェック用の姿勢 (yaw, pitch, roll)[deg]
    att = None
    acols = ("stabilizer.yaw", "stabilizer.pitch", "stabilizer.roll")
    if args.gate_deg > 0 and all(c in df.columns for c in acols):
        att = df[list(acols)].to_numpy(dtype=float)[idx]
    elif args.gate_deg > 0:
        print("警告: stabilizer.* が無いのでジャイロ整合チェックは無効です",
              file=sys.stderr)

    df = df.iloc[idx].reset_index(drop=True)
    res = run_odometry(planes, args.sensors, df["timestamp"].to_numpy(),
                       att=att, gate_deg=args.gate_deg)

    # サマリ
    print(f"入力: {path}  ({len(df)} フレーム)")
    for s in args.sensors:
        ok = np.isfinite(res["plane_d"][s]).sum()
        print(f"  sensor{s}: 平面抽出成功 {ok}/{len(df)} フレーム "
              f"({100.0 * ok / len(df):.1f}%), 平均距離 "
              f"{np.nanmean(res['plane_d'][s]):.1f} mm")
    print(f"  ΔR/Δt 推定フレーム数 : {(res['n_pairs'] > 0).sum()}")
    print(f"  Δt のランク 3/2/1/0  : "
          f"{(res['rank_t']==3).sum()}/{(res['rank_t']==2).sum()}/"
          f"{(res['rank_t']==1).sum()}/{(res['rank_t']==0).sum()}")
    src = res["rot_src"]
    print(f"  ΔR の出所 ToF/ジャイロ(乖離)/ジャイロ(対応なし)/なし : "
          f"{(src==1).sum()}/{(src==2).sum()}/{(src==3).sum()}/{(src==0).sum()}")
    print(f"  ゲートで捨てた対応 : {res['n_gated'].sum()} 件")
    print(f"  累積回転 (yaw,pitch,roll) [deg] : {res['ypr'][-1].round(2)}")
    print(f"  累積移動 (x,y,z) [mm]           : {res['pos'][-1].round(1)}")

    gyro = gyro_reference(df, res["t"])
    if gyro is not None:
        print(f"  ジャイロ積分 (yaw,pitch,roll)    : {gyro['ypr'][-1].round(2)}")
        rmse = np.sqrt(np.nanmean((res["ypr"] - gyro["ypr"]) ** 2, axis=0))
        print(f"  ToF vs ジャイロ RMSE [deg]      : {rmse.round(2)}")

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_tof_odometry.png")
    plot(res, args.sensors, df, out, tmax=args.tmax)

    if args.csv_out:
        cols = dict(timestamp=df["timestamp"].to_numpy())
        for i, nm in enumerate(["dyaw", "dpitch", "droll"]):
            cols[nm] = res["dR_ypr"][:, i]
        for i, nm in enumerate(["dx", "dy", "dz"]):
            cols[nm] = res["dt"][:, i]
        for i, nm in enumerate(["yaw", "pitch", "roll"]):
            cols[nm] = res["ypr"][:, i]
        for i, nm in enumerate(["x", "y", "z"]):
            cols[nm] = res["pos"][:, i]
        for s in args.sensors:
            cols[f"d_s{s}"] = res["plane_d"][s]
        cols["rank_t"] = res["rank_t"]
        if gyro is not None:
            for i, nm in enumerate(["yaw", "pitch", "roll"]):
                cols[f"gyro_{nm}"] = gyro["ypr"][:, i]
        pd.DataFrame(cols).to_csv(args.csv_out, index=False)
        print(f"推定結果CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
