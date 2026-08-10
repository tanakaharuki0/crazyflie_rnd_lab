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
MIN_INLIERS = 8           # 16点中この数以上のインライアが必要
PLANARITY_EPS = 0.006     # λ3/(λ1+λ2+λ3) の閾値 (tof_odometry.c と同値)
GATE_DEG = 2.0            # ジャイロ整合チェックの閾値[deg]。0で無効
LARGE_ERR_DEG = 2.0      # ジャイロとのズレがこれ以上の点はグラフ上に点でマーキングする[deg]

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

    Returns: (dict(n, d, planarity, n_inliers, rms) または None, reason)
    reason: "ok" / "too_few_points" / "ransac_fail" / "planarity_fail"
    """
    N = len(P)
    if N < MIN_INLIERS:
        return None, "too_few_points"

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
        return None, "ransac_fail"

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
        return None, "planarity_fail"

    rms = float(np.sqrt(np.mean((P[inl] @ n - d) ** 2)))
    return dict(n=n, d=d, planarity=planarity, n_inliers=int(inl.sum()), rms=rms), "ok"


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


def solve_yaw_only(pairs, att_f):
    """法線が1方向しかない (rank<2) ときに、生ジャイロの pitch/roll を既知
    として yaw だけを厳密に解く。

    n_pred(curr) = R_prev^T @ R_curr @ n_prev (実測との符号一致を確認済みの式)
    を変形すると R_prev @ n_curr = R_curr @ n_prev となる。
    Rz(psi_p)Ry(theta_p)Rx(phi_p)@n_curr = Rz(psi_c)Ry(theta_c)Rx(phi_c)@n_prev
    の両辺に Rz(-psi_p) を左からかけると

        Ry(theta_p)Rx(phi_p)@n_curr = Rz(psi_c-psi_p) @ Ry(theta_c)Rx(phi_c)@n_prev

    となり、Rz は (x,y) 成分だけを回すので Δψ=psi_c-psi_p を閉形式で解ける
    (pitch/roll は prev・curr 各時刻ごとに個別の値を使ってよく、途中で
    pitch/roll が変化していても幾何学的に厳密に成り立つ)。
    左辺は「前回の pitch/roll」を「今回の法線」に、右辺は「今回の
    pitch/roll」を「前回の法線」に適用する組み合わせになる点に注意
    (ランダム回転での数値検証でこの組み合わせのみ正しいΔψを再現することを確認済み)。

    pairs: run_odometry() の pairs (各要素 (s, prev_plane, cur_plane, R_att_prev))
    att_f: 現在フレームの (yaw,pitch,roll)[deg] (生ジャイロ積分)

    Returns: 現在フレームの姿勢行列 (Rz(psi_curr_est)@Ry(pitch)@Rx(roll)、
    psi_curr_est は各ペアからの推定の重み付き円環平均) または None。
    """
    psi_est, weights = [], []
    for _s, prev_plane, cur_plane, R_att_prev in pairs:
        psi_p, theta_p, phi_p = rot_to_ypr(R_att_prev)

        # 前回の pitch/roll を今回の法線に、今回の pitch/roll を前回の法線に適用
        m_A = rot_ypr(0.0, theta_p, phi_p) @ cur_plane["n"]
        m_B = rot_ypr(0.0, att_f[1], att_f[2]) @ prev_plane["n"]

        # 法線がほぼ真上/真下 (水平成分がほぼ0) だと yaw を観測できない
        if np.hypot(m_A[0], m_A[1]) < 1e-3 or np.hypot(m_B[0], m_B[1]) < 1e-3:
            continue

        dpsi = np.degrees(np.arctan2(m_A[1], m_A[0]) - np.arctan2(m_B[1], m_B[0]))
        psi_est.append(psi_p + dpsi)
        weights.append(1.0 / max(cur_plane["planarity"], 1e-6))

    if not psi_est:
        return None

    psi_est = np.radians(np.asarray(psi_est))
    weights = np.asarray(weights)
    weights = weights / weights.sum()
    sin_mean = np.sum(weights * np.sin(psi_est))
    cos_mean = np.sum(weights * np.cos(psi_est))
    psi_curr = np.degrees(np.arctan2(sin_mean, cos_mean))

    return rot_ypr(psi_curr, att_f[1], att_f[2])


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

    Returns: (planes, stats, planarity_fail_frames, planarity_ok_frames)。
    stats[s] は fit_plane_ransac() を実際に呼んだ回数の内訳
    (too_few_points/ransac_fail/planarity_fail/ok)。
    planarity_fail_frames / planarity_ok_frames は、平面性スコアで棄却/採用に
    なったフレーム番号 f のリスト (センサーごとに複数回入りうる)。
    """
    if arrs is None:
        arrs = sensor_arrays(df, sensors)
    if idx is None:
        idx = np.arange(len(df))
        fresh = {s: np.ones(len(idx), dtype=bool) for s in sensors}

    stats = {s: dict(too_few_points=0, ransac_fail=0, planarity_fail=0, ok=0) for s in sensors}
    planarity_fail_frames = []
    planarity_ok_frames = []
    planes = []
    for f, row in enumerate(idx):
        frame = {}
        for s in sensors:
            if not fresh[s][f]:
                frame[s] = None
                continue
            P, _ = points_body(arrs[s][row], s)
            pl, reason = fit_plane_ransac(P, rng)
            frame[s] = pl
            stats[s][reason] += 1
            if reason == "planarity_fail":
                planarity_fail_frames.append(f)
            elif reason == "ok":
                planarity_ok_frames.append(f)
        planes.append(frame)
    return planes, stats, planarity_fail_frames, planarity_ok_frames


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
        # gyro代用で上書きされる前の、Kabsch が実際に計算した生のΔR[deg]
        # (pairs が非空だったフレームは棄却の有無に関わらず必ず入る)
        kabsch_dR_ypr=np.full((F, 3), np.nan),
        # kabsch_dR_ypr が実際に表している「前回 Kabsch 成立時」からのフレーム数
        kabsch_ref_gap=np.full(F, np.nan),
        dt=np.full((F, 3), np.nan),
        rank_R=np.zeros(F, dtype=int),
        rank_t=np.zeros(F, dtype=int),
        n_pairs=np.zeros(F, dtype=int),
        n_gated=np.zeros(F, dtype=int),
        rot_err=np.full(F, np.nan),
        rot_src=np.zeros(F, dtype=int),   # 0:なし 1:ToF 2:ジャイロ(乖離) 3:ジャイロ(対応なし) 4:ToF(yawのみ)
        ypr=np.zeros((F, 3)),
        pos=np.zeros((F, 3)),
        plane_d={s: np.full(F, np.nan) for s in sensors},
        plane_n={s: np.full((F, 3), np.nan) for s in sensors},
        planarity={s: np.full(F, np.nan) for s in sensors},
        # そのセンサーの「前回の自分の計測」との法線変化がジャイロ整合チェックで
        # 棄却された (gate_deg 以上ずれた) フレームかどうか
        gated={s: np.zeros(F, dtype=bool) for s in sensors},
        # そのセンサー・フレームでの法線ペアのジャイロ予測とのズレ角[deg] (計算されなければ NaN)
        pair_err={s: np.full(F, np.nan) for s in sensors},
        # ① 法線ペアごとのジャイロ予測とのズレ角[deg] (棄却されたものも含め全て、集計用)
        pair_gate_err=[],
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

    # センサーごとに直近2回分の自分の計測を保持する (prev1=直前, prev2=前々回)。
    # 「前回」の更新は棄却の有無に関わらず常に行う。直前のデータとの比較が
    # 棄却された場合は、前々回のデータとも比較してみて、それが gate_deg 未満
    # なら棄却せず採用する (1回の飛び値だけを理由に切り捨てず、もう1段階だけ
    # 遡って救済する)。
    prev1 = {s: None for s in sensors}     # (plane, R_att or None)
    prev2 = {s: None for s in sensors}
    R_cum = np.eye(3)
    p_cum = np.zeros(3)
    R_last = R_att[0] if R_att is not None else None
    R_kabsch_prev1 = R_att[0] if R_att is not None else None   # 前回 Kabsch が成立した時の姿勢
    R_kabsch_prev2 = R_att[0] if R_att is not None else None   # 前々回 Kabsch が成立した時の姿勢
    R_cum_at_kabsch_prev1 = np.eye(3)   # 前回 Kabsch 成立時点での R_cum のスナップショット
    kabsch_prev1_frame = 0                                      # 前回 Kabsch が成立したフレーム番号

    for f in range(F):
        pairs, n_gated = [], 0
        for s in sensors:
            cur = planes[f][s]
            if cur is None:
                continue
            pv1 = prev1[s]
            ok, chosen = True, None
            if pv1 is not None and pv1[0] is not None:
                chosen = pv1
                if gate_deg > 0 and R_att is not None and pv1[1] is not None:
                    # 壁が固定なら n_curr = R_prev^T R_curr n_prev
                    # (R_att は world_R_body。ズレ角の検証で OLDER.T@NEWER の順が
                    # 実測と符号一致することを確認済み。逆順だと符号が反転する)
                    n_pred1 = pv1[1].T @ R_att[f] @ pv1[0]["n"]
                    c1 = float(np.clip(n_pred1 @ cur["n"], -1.0, 1.0))
                    err1 = np.degrees(np.arccos(c1))

                    pv2 = prev2[s]
                    err2 = None
                    if pv2 is not None and pv2[0] is not None and pv2[1] is not None:
                        n_pred2 = pv2[1].T @ R_att[f] @ pv2[0]["n"]
                        c2 = float(np.clip(n_pred2 @ cur["n"], -1.0, 1.0))
                        err2 = np.degrees(np.arccos(c2))

                    # ヒストグラム集計用には、前回・前々回どちらとの比較が
                    # 小さいか (=より整合していた方) を採用する
                    pair_err_deg = err1 if err2 is None else min(err1, err2)
                    out["pair_gate_err"].append(pair_err_deg)
                    out["pair_err"][s][f] = pair_err_deg

                    if err1 < gate_deg:
                        ok, chosen = True, pv1
                    elif err2 is not None and err2 < gate_deg:
                        ok, chosen = True, pv2
                    else:
                        ok, chosen = False, None
                        n_gated += 1
                        out["gated"][s][f] = True
                if ok and chosen is not None:
                    pairs.append((s, chosen[0], cur, chosen[1]))
            prev2[s] = prev1[s]
            prev1[s] = (cur, R_att[f] if R_att is not None else None)

        out["n_pairs"][f] = len(pairs)
        out["n_gated"][f] = n_gated

        # ジャイロによる機体回転 (前回このループを通ったとき → 今回、フォールバック代用用)
        dR_gyro = None
        if R_att is not None and R_last is not None:
            dR_gyro = R_last.T @ R_att[f]      # body_prev_R_body_curr

        dRb, src, since_ref = None, 0, False
        if pairs:
            N_prev = np.array([p[1]["n"] for p in pairs])
            N_curr = np.array([p[2]["n"] for p in pairs])
            w = np.array([1.0 / max(p[2]["planarity"], 1e-6) for p in pairs])
            w /= w.sum()

            dR, rank_R = kabsch_normals(N_prev, N_curr, w)
            out["rank_R"][f] = rank_R
            yaw_only = False
            if rank_R < 2:
                # 法線が1方向のみ → 3自由度の回転は不定だが、生ジャイロの
                # pitch/roll を既知として yaw だけは厳密に解ける (solve_yaw_only)
                R_curr_hybrid = solve_yaw_only(pairs, att[f]) if att is not None else None
                if R_curr_hybrid is not None:
                    ref = R_kabsch_prev1 if R_kabsch_prev1 is not None else R_att[f]
                    dRb = ref.T @ R_curr_hybrid
                    yaw_only = True
                else:
                    dRb = np.eye(3)
            else:
                # Kabsch は法線を prev→curr に写す回転 (= body_curr_R_body_prev) を返す。
                # 機体自身の回転はその逆なので転置し、ジャイロと同じ向きに揃える。
                dRb = dR.T
            src = 4 if yaw_only else 1
            out["kabsch_dR_ypr"][f] = rot_to_ypr(dRb)   # 棄却前の生の Kabsch(または yaw のみ)ΔR

            # dRb と比較するジャイロ回転は「前回 Kabsch が成立したフレーム」を起点
            # にする (① の pv1 と同じ考え方。R_last=1ループ前だと、pairs が
            # 疎にしか成立しないセンサー構成では dRb が表す実時間幅とズレる)。
            # 前回起点での比較が棄却された場合は、① と同様に前々回 Kabsch 起点
            # とも比較し、gate_deg 未満なら棄却せず採用する。
            # 棄却して代用する値は R_cum の1フレーム分の更新に使うので、比較に
            # だけ dR_gyro_kabsch* を使い、代用そのものは他フレームと同じ1フレ
            # ーム分の dR_gyro を使う (でないと途中フレームで既に積算済みの回転
            # を二重に積むことになる)。
            if R_att is not None and gate_deg > 0:
                accepted = R_kabsch_prev1 is None
                if R_kabsch_prev1 is not None:
                    dR_gyro_kabsch1 = R_kabsch_prev1.T @ R_att[f]
                    c1 = float(np.clip((np.trace(dRb.T @ dR_gyro_kabsch1) - 1.0) / 2.0, -1.0, 1.0))
                    err1 = np.degrees(np.arccos(c1))

                    err2 = None
                    if R_kabsch_prev2 is not None:
                        dR_gyro_kabsch2 = R_kabsch_prev2.T @ R_att[f]
                        c2 = float(np.clip((np.trace(dRb.T @ dR_gyro_kabsch2) - 1.0) / 2.0, -1.0, 1.0))
                        err2 = np.degrees(np.arccos(c2))

                    # ヒストグラム集計用には、前回・前々回どちらとの比較が
                    # 小さいか (=より整合していた方) を採用する
                    out["rot_err"][f] = err1 if err2 is None else min(err1, err2)

                    if err1 < gate_deg:
                        accepted = True
                    elif err2 is not None and err2 < gate_deg:
                        accepted = True
                if not accepted:
                    dRb, src = dR_gyro, 2

            # src が 1 か 4 のままなら dRb は「前回 Kabsch 成立時からの累積回転」を
            # 表す (途中フレームの単発ジャイロ更新とは意味が違う)。R_cum に単純に
            # 掛け算すると、間のフレームで既にジャイロ代用として積算した分と
            # 二重に積むことになるため、前回 Kabsch 成立時点の R_cum のスナップ
            # ショットを起点として置き換える。
            since_ref = src in (1, 4)

            dd = np.array([p[2]["d"] - p[1]["d"] for p in pairs])
            dt, rank_t, _ = solve_translation(N_prev, dd, w)
            out["dt"][f] = dt
            out["rank_t"][f] = rank_t
            p_cum = p_cum + R_cum @ dt      # 姿勢を進める前の R_cum で回す

            if R_att is not None:
                # 今回 Kabsch が成立したので次回・次々回の起点を更新
                R_kabsch_prev2 = R_kabsch_prev1
                R_kabsch_prev1 = R_att[f]
        elif dR_gyro is not None:
            dRb, src = dR_gyro, 3

        if dRb is not None:
            out["dR_ypr"][f] = rot_to_ypr(dRb)
            if since_ref:
                R_cum = R_cum_at_kabsch_prev1 @ dRb
            else:
                R_cum = R_cum @ dRb
        out["rot_src"][f] = src

        if pairs:
            # 次に Kabsch が成立したときの起点として、更新後の R_cum を保存
            R_cum_at_kabsch_prev1 = R_cum

        if R_att is not None:
            R_last = R_att[f]
        out["ypr"][f] = rot_to_ypr(R_cum)
        out["pos"][f] = p_cum

    out["pair_gate_err"] = np.asarray(out["pair_gate_err"], dtype=float)
    return out


GYRO_COLS = ("gyro.x", "gyro.y", "gyro.z")   # roll/pitch/yaw の角速度 [deg/s] (機体角速度 p,q,r)
GYRO_BIAS_SEC = 0.5                          # 先頭この秒数の平均をバイアスとして除去


def rodrigues(v):
    """回転ベクトル v [rad] (軸=方向, ノルム=回転角) を 3x3 回転行列に変換する。"""
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def gyro_reference(df, t):
    """生ジャイロ角速度 (gyro.x/y/z) から Δ回転[deg/frame] と累積回転[deg] を作る。

    gyro.x/y/z はボディ座標系 roll/pitch/yaw 軸まわりの機体角速度 (p,q,r) その
    ものなので、各ステップの回転ベクトル v=(p,q,r)*dt をロドリゲスの公式で
    回転行列に変換し、姿勢を行列の逐次合成 R[f] = R[f-1] @ Exp(v) で積分する
    (yaw,pitch,roll を別々に単純積分する方法は機体角速度とオイラー角速度を
    混同しており、姿勢が傾くほど軸間カップリングの誤差が乗る)。
    ステップ角は通常数度以下で、ステップ数もToFフレーム数程度 (数百) なので
    増える計算量は無視できる。

    stabilizer.yaw/pitch/roll (EKF等の姿勢推定値) は無補正のまま使うとドリフト
    が大きく信頼できないことがわかったため、run_odometry() の「ジャイロ代用」
    (rot_src 2/3) にも、この生ジャイロ積分を att として渡して使う。
    つまりこの関数の出力が「代用元」と「比較対象」を兼ねる唯一のジャイロ信号
    になる。列が無ければ None。
    """
    if not all(c in df.columns for c in GYRO_COLS):
        return None

    rate_body = df[list(GYRO_COLS)].to_numpy(dtype=float)   # (p,q,r) = (roll,pitch,yaw)軸 [deg/s]

    # 静止しているはずの先頭区間の平均をバイアスとして差し引く
    n_bias = max(int(np.searchsorted(t - t[0], GYRO_BIAS_SEC)), 1)
    rate_body = rate_body - np.nanmean(rate_body[:n_bias], axis=0)

    dt_s = np.diff(t, prepend=t[0])                        # 各フレームの経過時間 [s]
    v = np.radians(rate_body) * dt_s[:, None]              # 各ステップの回転ベクトル [rad]

    F = len(t)
    R = np.eye(3)
    ypr = np.zeros((F, 3))
    d_ypr = np.zeros((F, 3))
    for f in range(F):
        R_step = rodrigues(v[f])
        d_ypr[f] = rot_to_ypr(R_step)      # [deg/frame] (表示用)
        R = R @ R_step
        ypr[f] = rot_to_ypr(R)             # 累積姿勢 [deg]
    return dict(d_ypr=d_ypr, ypr=ypr)


def plot(res, sensors, df, out_path, tmax=None, gate_deg=GATE_DEG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.font_manager as fm

    # 日本語ラベル用フォント (無ければ既定のまま)
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "IPAGothic", "TakaoGothic", "VL Gothic",
                 "Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False

    t = res["t"]
    gyro = gyro_reference(df, t)
    t = t - t[0]
    fig, ax = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
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
    # yaw: Kabsch が計算した生の値 (棄却の有無に関わらず pairs があれば全て)
    # を点で全部プロットし、ジャイロとのズレ (rot_err) を色で表す。
    # ズレ = gate_deg で黄色、それより小さいと緑→青、大きいほど赤に近づく。
    err_cmap = mcolors.LinearSegmentedColormap.from_list(
        "err_cmap", ["blue", "green", "yellow", "orange", "red"])
    if gate_deg > 0:
        err_norm = mcolors.TwoSlopeNorm(vcenter=gate_deg, vmin=0.0, vmax=2.0 * gate_deg)
    else:
        err_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    yaw_kabsch = res["kabsch_dR_ypr"][:, 0]
    has_kabsch = ~np.isnan(yaw_kabsch) & ~np.isnan(res["rot_err"])
    if has_kabsch.any():
        sc = ax[1, 0].scatter(t[has_kabsch], yaw_kabsch[has_kabsch], s=15,
                               c=res["rot_err"][has_kabsch], cmap=err_cmap,
                               norm=err_norm, edgecolors="k", linewidths=0.3,
                               zorder=5, label="Kabsch Δyaw (生値)")
        fig.colorbar(sc, ax=ax[1, 0], pad=0.01, label="ジャイロとのズレ [deg]")

    # gyroではなく実際に Kabsch (ToF) の結果が採用されたフレーム (rot_src==1) は
    # pitch/roll では従来通り青色の点でマーキングする
    kabsch_ok = res["rot_src"] == 1
    for i, name in zip((1, 2), ("pitch", "roll")):
        ax[1, 0].plot(t, res["dR_ypr"][:, i], lw=1, color=ypr_colors[i],
                      label=f"ToF Δ{name}")
        if kabsch_ok.any():
            ax[1, 0].scatter(t[kabsch_ok], res["dR_ypr"][kabsch_ok, i], s=15,
                              color="tab:blue", edgecolors="k", linewidths=0.3,
                              zorder=5)
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
    # yaw は法線から推定できた区間 (rot_src==1 または rank不足でも yaw だけ
    # 推定できた rot_src==4) は青、ジャイロで代替した区間は赤で塗り分ける
    tof_src = (res["rot_src"] == 1) | (res["rot_src"] == 4)
    yaw = res["ypr"][:, 0]
    yaw_tof = np.where(tof_src, yaw, np.nan)
    yaw_gyro_sub = np.where(~tof_src, yaw, np.nan)
    ax[2, 0].plot(t, yaw_tof, lw=1.2, color=ypr_colors[0], label="ToF yaw")
    ax[2, 0].plot(t, yaw_gyro_sub, lw=1.2, color="red", label="ToF yaw (gyro代替)")
    for i, name in zip((1, 2), ("pitch", "roll")):
        ax[2, 0].plot(t, res["ypr"][:, i], lw=1.2, color=ypr_colors[i],
                      label=f"ToF {name}")
    if gyro is not None:
        for i, name in enumerate(["yaw", "pitch", "roll"]):
            ax[2, 0].plot(t, gyro["ypr"][:, i], lw=1.0, ls="--", alpha=0.75,
                          color=ypr_colors[i], label=f"gyro {name}")
    ax[2, 0].axhline(180, color="k", lw=0.5)
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
    # 平面抽出/平面性スコアで棄却された区間 (前後どちらかが NaN) は元の実装通り
    # 描画しない (隣接フレーム同士でなければ線をつながない)。
    # ジャイロ整合チェックで棄却された遷移 (res["gated"][s]) は点線にする。
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, s in enumerate(sensors):
        n = res["plane_n"][s]
        az = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
        color = prop_colors[i % len(prop_colors)]
        ax[3, 1].plot([], [], lw=1, color=color, label=f"sensor{s}")
        ok = ~np.isnan(az)
        for f in range(1, len(az)):
            if not (ok[f - 1] and ok[f]):
                continue
            ls = "--" if res["gated"][s][f] else "-"
            ax[3, 1].plot(t[f - 1:f + 1], az[f - 1:f + 1], lw=1, ls=ls, color=color)
        # ジャイロとのズレが LARGE_ERR_DEG 以上のフレームのみ点でマーキングする
        # (採用された(棄却されなかった)点は線のみで表し、マーカーは描かない)
        big = res["pair_err"][s] >= LARGE_ERR_DEG
        if big.any():
            ax[3, 1].scatter(t[big], az[big], s=25, color=color,
                              edgecolors="k", linewidths=0.5, zorder=5)
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


HIST_LO, HIST_HI, HIST_WIDTH = 0.0, 10.0, 0.2   # ズレ角ヒストグラムのビン設定 [deg]


def _hist_with_overflow(data, lo=HIST_LO, hi=HIST_HI, width=HIST_WIDTH):
    """[lo,hi) を width 刻みでヒストグラム化し、hi 以上は末尾のビンにまとめる
    (ズレ角は arccos 由来で常に0以上なので下側のはみ出しビンは作らない)。

    Returns: (bin中心の配列, 度数の配列, はみ出しビンかどうかのbool配列)
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    edges = np.arange(lo, hi + width / 2, width)
    counts, _ = np.histogram(data, bins=edges)
    n_over = int(np.sum(data >= hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    centers_full = np.concatenate([centers, [hi + width]])
    counts_full = np.concatenate([counts, [n_over]])
    is_overflow = np.concatenate([np.zeros(len(centers), dtype=bool), [True]])
    return centers_full, counts_full, is_overflow


def plot_gate_histograms(res, out_path):
    """① 法線ペアのジャイロ不整合ズレ、② KabschのΔRのジャイロ乖離のヒストグラム。

    横軸: 生じたズレ [deg] (0〜10を0.2度刻み、10度以上は末尾のビンにまとめる)
    縦軸: そのズレ幅で観測された回数
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))

    specs = [
        (ax[0], res["pair_gate_err"], "tab:blue",
         "① 法線ペアのジャイロ不整合ズレの分布", "normal-pair angle error vs gyro [deg]"),
        (ax[1], res["rot_err"], "tab:orange",
         "② KabschのΔRのジャイロ乖離の分布", "Kabsch ΔR angle error vs gyro [deg]"),
    ]
    for a, data, color, title, xlabel in specs:
        centers, counts, is_overflow = _hist_with_overflow(data)
        colors = np.where(is_overflow, "0.6", color)
        a.bar(centers, counts, width=HIST_WIDTH * 0.9, color=colors)
        a.axvline(GATE_DEG, color="k", ls="--", lw=0.8, label=f"gate_deg={GATE_DEG:g}")
        a.set_xlabel(xlabel)
        a.set_ylabel("count")
        a.set_title(title)
        a.set_xticks(list(np.arange(HIST_LO, HIST_HI - 1e-9, 1.0)) + [HIST_HI + HIST_WIDTH])
        a.set_xticklabels([f"{v:g}" for v in np.arange(HIST_LO, HIST_HI - 1e-9, 1.0)]
                           + [f">={HIST_HI:g}"])
        a.grid(alpha=0.3)
        a.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"ズレ角ヒストグラムを保存: {out_path}")


TIME_HIST_LO, TIME_HIST_HI, TIME_HIST_WIDTH = 0.0, 60.0, 2.0   # 棄却時刻ヒストグラムのビン設定 [s]


def plot_rejection_time_histograms(res, planarity_fail_t, planarity_ok_t, out_path):
    """3種類の棄却/採用が発生した時刻のヒストグラム (上段:棄却, 下段:採用)。

    - 平面性スコア (fit_plane_ransac が planarity_fail/ok を返した時刻)
    - ① 法線ペアのジャイロ乖離 (res["gated"] / 比較したが棄却されなかったもの)
    - ② KabschのΔRのジャイロ乖離 (res["rot_src"]==2 / ==1)

    横軸: 時刻[s] (0〜60sを2s刻み)、縦軸: その時間帯での回数
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = res["t"] - res["t"][0]

    pca_gate_t = np.concatenate([t[res["gated"][s]] for s in res["gated"]]) \
        if res["gated"] else np.array([])
    pca_ok_t = np.concatenate([
        t[~np.isnan(res["pair_err"][s]) & ~res["gated"][s]] for s in res["gated"]
    ]) if res["gated"] else np.array([])
    kabsch_gate_t = t[res["rot_src"] == 2]
    kabsch_ok_t = t[(res["rot_src"] == 1) | (res["rot_src"] == 4)]

    edges = np.arange(TIME_HIST_LO, TIME_HIST_HI + TIME_HIST_WIDTH / 2, TIME_HIST_WIDTH)

    fig, ax = plt.subplots(2, 3, figsize=(14, 7.0))
    specs = [
        (ax[0, 0], np.asarray(planarity_fail_t, dtype=float), "tab:purple",
         "平面性スコアによる棄却"),
        (ax[0, 1], np.asarray(pca_gate_t, dtype=float), "tab:blue",
         "① 法線ペアのジャイロ乖離による棄却"),
        (ax[0, 2], np.asarray(kabsch_gate_t, dtype=float), "tab:orange",
         "② KabschのΔRのジャイロ乖離による棄却"),
        (ax[1, 0], np.asarray(planarity_ok_t, dtype=float), "tab:purple",
         "平面性スコアで採用 (棄却されなかった)"),
        (ax[1, 1], np.asarray(pca_ok_t, dtype=float), "tab:blue",
         "① 法線ペアで採用 (棄却されなかった)"),
        (ax[1, 2], np.asarray(kabsch_ok_t, dtype=float), "tab:orange",
         "② KabschのΔRで採用 (棄却されなかった)"),
    ]
    for a, times, color, title in specs:
        a.hist(times, bins=edges, color=color, edgecolor="k", linewidth=0.3)
        a.set_xlabel("time [s]")
        a.set_ylabel("count")
        a.set_title(title)
        a.set_xlim(TIME_HIST_LO, TIME_HIST_HI)
        a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
        a.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"棄却時刻ヒストグラムを保存: {out_path}")


def plot_normal_azimuth_per_sensor(res, sensors, out_dir, base, gate_deg=GATE_DEG, att=None):
    """センサーごとに法線ベクトルの方位角グラフを別々のPNGとして保存する。

    3.2 KabschのΔR (yaw) のグラフと同じ配色ルールで、各点をジャイロとの
    ズレ (pair_err) で色分けする: ズレ=gate_deg で黄色、それより小さいと
    緑→青、大きいほど赤に近づく。

    att (F,3) の (yaw,pitch,roll)[deg] を渡すと、そのセンサーの最初の
    有効な計測を基準に、以降はジャイロの回転だけで法線を回して予測した
    方位角を比較用の点線として重ねる (実測点としてはプロットしない)。

    Returns: 保存したPNGパスのリスト。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    t = res["t"] - res["t"][0]

    R_att = None
    if att is not None:
        R_att = [rot_ypr(*att[f]) for f in range(len(att))]

    err_cmap = mcolors.LinearSegmentedColormap.from_list(
        "err_cmap", ["blue", "green", "yellow", "orange", "red"])
    if gate_deg > 0:
        err_norm = mcolors.TwoSlopeNorm(vcenter=gate_deg, vmin=0.0, vmax=10.0)
    else:
        err_norm = mcolors.Normalize(vmin=0.0, vmax=10.0)

    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for s in sensors:
        n = res["plane_n"][s]
        az = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
        pair_err = res["pair_err"][s]
        ok = ~np.isnan(az)

        fig, ax = plt.subplots(figsize=(9, 4))

        # 隣接する有効フレーム同士だけを薄い線でつなぐ (平面性棄却区間はまたがない)
        for f in range(1, len(az)):
            if ok[f - 1] and ok[f]:
                ax.plot(t[f - 1:f + 1], az[f - 1:f + 1], lw=0.8, color="0.75", zorder=1)

        # 比較用のジャイロ予測 (最初の有効な計測を基準に、ジャイロの回転だけで
        # 法線を回して予測した方位角を点線で重ねる。実測点としては打たない)
        if R_att is not None and ok.any():
            f0 = int(np.flatnonzero(ok)[0])
            n0 = n[f0]
            R0 = R_att[f0]
            gyro_az = np.full(len(az), np.nan)
            for f in range(f0, len(az)):
                n_pred = R0.T @ R_att[f] @ n0
                gyro_az[f] = np.degrees(np.arctan2(n_pred[1], n_pred[0]))
            ax.plot(t[f0:], gyro_az[f0:], lw=1, ls="--", color="k", alpha=0.7,
                    zorder=2, label="gyro予測 (初回計測基準)")

        has_err = ok & ~np.isnan(pair_err)
        if has_err.any():
            sc = ax.scatter(t[has_err], az[has_err], s=18, c=pair_err[has_err],
                             cmap=err_cmap, norm=err_norm, edgecolors="k",
                             linewidths=0.3, zorder=3, label="ジャイロと比較あり")
            fig.colorbar(sc, ax=ax, pad=0.01, label="ジャイロとのズレ [deg]")

        # 直前の自分の計測が無く比較できなかった点はグレーで表示
        no_err = ok & np.isnan(pair_err)
        if no_err.any():
            ax.scatter(t[no_err], az[no_err], s=18, color="0.5",
                       edgecolors="k", linewidths=0.3, zorder=2, label="比較対象なし")

        ax.set_xlabel("time [s]")
        ax.set_ylabel("normal azimuth [deg]")
        ax.set_title(f"sensor{s}: 抽出された法線ベクトルの方位角 (ボディ座標系)")
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{base}_normal_azimuth_sensor{s}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        out_paths.append(out_path)
        print(f"法線方位角グラフ(sensor{s})を保存: {out_path}")
    return out_paths


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

    planes, plane_stats, planarity_fail_frames, planarity_ok_frames = extract_planes(
        df, args.sensors, rng, arrs, idx, fresh)

    df = df.iloc[idx].reset_index(drop=True)
    t_s = df["timestamp"].to_numpy(float) / 1000.0

    # ジャイロ整合チェック/代用用の姿勢 (yaw, pitch, roll)[deg] は生ジャイロ積分。
    # 代用元 (att) と比較・RMSE計算に使う参照を同じ信号にするため gyro を使い回す。
    gyro = gyro_reference(df, t_s)
    att = None
    if args.gate_deg > 0 and gyro is not None:
        att = gyro["ypr"]
    elif args.gate_deg > 0:
        print("警告: gyro.x/y/z が無いのでジャイロ整合チェックは無効です",
              file=sys.stderr)

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
    print(f"  ΔR の出所 ToF/ジャイロ(乖離)/ジャイロ(対応なし)/なし/ToF(yawのみ) : "
          f"{(src==1).sum()}/{(src==2).sum()}/{(src==3).sum()}/{(src==0).sum()}/{(src==4).sum()}")
    print(f"  ゲートで捨てた対応 : {res['n_gated'].sum()} 件")

    # 平面性スコアによる棄却率 (fit_plane_ransac が planarity > PLANARITY_EPS で
    # None を返した回数 / 実際に fit_plane_ransac を呼んだ回数)
    tot = {k: sum(plane_stats[s][k] for s in args.sensors)
           for k in ("too_few_points", "ransac_fail", "planarity_fail", "ok")}
    n_attempts = sum(tot.values())
    if n_attempts > 0:
        print(f"  平面性スコアによる棄却率 : {tot['planarity_fail']}/{n_attempts} "
              f"({100.0 * tot['planarity_fail'] / n_attempts:.1f}%) "
              f"[点不足 {tot['too_few_points']}, RANSAC失敗 {tot['ransac_fail']}]")

    # ジャイロ整合チェックによる法線ペアの棄却率 (①: n_pred vs cur["n"])
    n_gated_total = int(res["n_gated"].sum())
    n_pair_total = int(res["n_pairs"].sum())
    n_pair_attempts = n_gated_total + n_pair_total
    if n_pair_attempts > 0:
        print(f"  法線ペアのジャイロ不整合棄却率 : {n_gated_total}/{n_pair_attempts} "
              f"({100.0 * n_gated_total / n_pair_attempts:.1f}%)")

    # Kabsch の ΔR がジャイロと gate_deg 以上ずれて棄却され、代わりにジャイロの
    # 回転を採用した割合 (rot_src: 1=ToF採用, 4=ToF(yawのみ)採用, 2=ジャイロ代用(乖離))
    n_tof = int((src == 1).sum()) + int((src == 4).sum())
    n_kabsch_rejected = int((src == 2).sum())
    n_kabsch_attempts = n_tof + n_kabsch_rejected
    if n_kabsch_attempts > 0:
        print(f"  KabschのΔRがジャイロ乖離で棄却された割合 : "
              f"{n_kabsch_rejected}/{n_kabsch_attempts} "
              f"({100.0 * n_kabsch_rejected / n_kabsch_attempts:.1f}%)")

    print(f"  累積回転 (yaw,pitch,roll) [deg] : {res['ypr'][-1].round(2)}")
    print(f"  累積移動 (x,y,z) [mm]           : {res['pos'][-1].round(1)}")

    if gyro is not None:
        print(f"  ジャイロ積分 (yaw,pitch,roll)    : {gyro['ypr'][-1].round(2)}")
        rmse = np.sqrt(np.nanmean((res["ypr"] - gyro["ypr"]) ** 2, axis=0))
        print(f"  ToF vs ジャイロ RMSE [deg]      : {rmse.round(2)}")

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_tof_odometry.png")
    plot(res, args.sensors, df, out, tmax=args.tmax, gate_deg=args.gate_deg)

    hist_out = os.path.join(OUT_DIR, f"{base}_gate_hist.png")
    plot_gate_histograms(res, hist_out)

    planarity_fail_t = t_s[planarity_fail_frames] - t_s[0]
    planarity_ok_t = t_s[planarity_ok_frames] - t_s[0]
    time_hist_out = os.path.join(OUT_DIR, f"{base}_rejection_time_hist.png")
    plot_rejection_time_histograms(res, planarity_fail_t, planarity_ok_t, time_hist_out)

    plot_normal_azimuth_per_sensor(res, args.sensors, OUT_DIR, base,
                                    gate_deg=args.gate_deg, att=att)

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
