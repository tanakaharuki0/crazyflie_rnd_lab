#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crazyflie firmware (crazyflie-firmware-vl53l8cx) の
  src/modules/src/kalman_core/kalman_core.c
  src/modules/src/kalman_core/mm_flow.c
  src/modules/src/kalman_core/mm_yaw_error.c
を Python に移植した、姿勢誤差(error-state)型のEKF。

状態ベクトル S (9次元, kalman_core.h の KC_STATE_* と同じ並び):
    X, Y, Z       : ワールド座標系での位置 [m]
    PX, PY, PZ    : ボディ座標系での速度 [m/s]
    D0, D1, D2    : 姿勢誤差 (小角近似, ロール軸・ピッチ軸・ヨー軸まわり) [rad]
      (finalize() で毎回クォータニオン q に畳み込まれ、D0..D2 は0に戻る)

姿勢そのものはクォータニオン q=[w,x,y,z] で保持し (特異点を避けるため)、
predict()のたびにジャイロでq自体を直接回転させる (kalman_core.cと同じ設計)。

単位はすべて firmware内部と同じ SI 単位:
    加速度 [m/s^2] (ログのacc.*はg単位なので GRAVITY_MAGNITUDE 倍が必要)
    角速度 [rad/s] (ログのgyro.*はdeg/s単位なので DEG_TO_RAD 倍が必要)
"""
import numpy as np

GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
EPS = 1e-9

MAX_COVARIANCE = 100.0
MIN_COVARIANCE = 1e-6

# 状態インデックス (kalman_core.h の kalmanCoreStateIdx_t と同じ)
KC_STATE_X, KC_STATE_Y, KC_STATE_Z = 0, 1, 2
KC_STATE_PX, KC_STATE_PY, KC_STATE_PZ = 3, 4, 5
KC_STATE_D0, KC_STATE_D1, KC_STATE_D2 = 6, 7, 8
KC_STATE_DIM = 9


class KalmanCoreParams:
    """kalman_core_params_defaults.h の KALMAN_CORE_DEFAULT_PARAMS_INIT を移植。

    procNoiseAcc_z のデフォルトは CONFIG_ESTIMATOR_KALMAN_GENERAL_PURPOSE の
    有無で 0.5/1.0 が切り替わる。ToFリングのような非標準デッキ構成は
    general-purpose 側でビルドされる想定のため、そちらのデフォルト(0.5)を採用する
    (quadIsFlyingが常にFalseになるのも同じconfigの挙動: 下記KalmanCore参照)。
    """
    def __init__(self):
        self.stdDevInitialPosition_xy = 100.0
        self.stdDevInitialPosition_z = 1.0
        self.stdDevInitialVelocity = 0.01
        self.stdDevInitialAttitude_rollpitch = 0.01
        self.stdDevInitialAttitude_yaw = 0.01

        self.procNoiseAcc_xy = 0.5
        self.procNoiseAcc_z = 0.5   # general-purpose config のデフォルト
        self.procNoiseVel = 0.0
        self.procNoisePos = 0.0
        self.procNoiseAtt = 0.0
        self.measNoiseBaro = 2.0            # [m]
        self.measNoiseGyro_rollpitch = 0.1  # [rad/s]
        self.measNoiseGyro_yaw = 0.1        # [rad/s]

        self.initialX = 0.0
        self.initialY = 0.0
        self.initialZ = 0.0
        self.initialYaw = 0.0   # [rad]

        self.attitudeReversion = 0.001

        self.dragB_x = 0.0
        self.dragB_y = 0.0
        self.dragB_z = 0.0

        self.cop_x = 0.0
        self.cop_y = 0.0
        self.cop_z = 0.0


def _quat_mult(a, b):
    """クォータニオン積 a*b ([w,x,y,z]順)。kalman_core.cのpredictDt内のtmpq計算と同じ。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        ax*bw + aw*bx + az*by - ay*bz,
        ay*bw - az*bx + aw*by + ax*bz,
        az*bw + ay*bx - ax*by + aw*bz,
    ])


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [w*w + x*x - y*y - z*z, 2*x*y - 2*w*z,         2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         w*w - x*x + y*y - z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,         w*w - x*x - y*y + z*z],
    ])


def quat_from_rotmat(R):
    """回転行列 → クォータニオン[w,x,y,z] (オイラー角を経由しない直接変換, Shepperd法)。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + EPS)


def _small_angle_rot_cov_matrix(d0, d1, d2):
    """attitude-error方向の共分散を回転させる9x9行列 (kalman_core.c predictDt/finalize と同一式)。"""
    A = np.eye(KC_STATE_DIM)
    A[KC_STATE_D0, KC_STATE_D0] = 1 - d1*d1/2 - d2*d2/2
    A[KC_STATE_D0, KC_STATE_D1] = d2 + d0*d1/2
    A[KC_STATE_D0, KC_STATE_D2] = -d1 + d0*d2/2

    A[KC_STATE_D1, KC_STATE_D0] = -d2 + d0*d1/2
    A[KC_STATE_D1, KC_STATE_D1] = 1 - d0*d0/2 - d2*d2/2
    A[KC_STATE_D1, KC_STATE_D2] = d0 + d1*d2/2

    A[KC_STATE_D2, KC_STATE_D0] = d1 + d0*d2/2
    A[KC_STATE_D2, KC_STATE_D1] = -d0 + d1*d2/2
    A[KC_STATE_D2, KC_STATE_D2] = 1 - d0*d0/2 - d1*d1/2
    return A


class KalmanCore:
    """kalman_core.c の kalmanCoreData_t + 各関数群のPython移植 (疎結合フュージョン用)。"""

    def __init__(self, params=None, now_s=0.0):
        self.params = params or KalmanCoreParams()
        p = self.params

        self.S = np.zeros(KC_STATE_DIM)
        self.S[KC_STATE_X] = p.initialX
        self.S[KC_STATE_Y] = p.initialY
        self.S[KC_STATE_Z] = p.initialZ

        self.initial_q = np.array([np.cos(p.initialYaw/2), 0.0, 0.0, np.sin(p.initialYaw/2)])
        self.q = self.initial_q.copy()
        self.R = np.eye(3)

        self.P = np.zeros((KC_STATE_DIM, KC_STATE_DIM))
        self.P[KC_STATE_X, KC_STATE_X] = p.stdDevInitialPosition_xy ** 2
        self.P[KC_STATE_Y, KC_STATE_Y] = p.stdDevInitialPosition_xy ** 2
        self.P[KC_STATE_Z, KC_STATE_Z] = p.stdDevInitialPosition_z ** 2
        self.P[KC_STATE_PX, KC_STATE_PX] = p.stdDevInitialVelocity ** 2
        self.P[KC_STATE_PY, KC_STATE_PY] = p.stdDevInitialVelocity ** 2
        self.P[KC_STATE_PZ, KC_STATE_PZ] = p.stdDevInitialVelocity ** 2
        self.P[KC_STATE_D0, KC_STATE_D0] = p.stdDevInitialAttitude_rollpitch ** 2
        self.P[KC_STATE_D1, KC_STATE_D1] = p.stdDevInitialAttitude_rollpitch ** 2
        self.P[KC_STATE_D2, KC_STATE_D2] = p.stdDevInitialAttitude_yaw ** 2

        self.is_updated = False
        self.last_prediction_s = now_s
        self.last_process_noise_s = now_s

    # ------------------------------------------------------------------
    # predict (kalmanCorePredict / predictDt)
    # ------------------------------------------------------------------
    def predict(self, acc_ms2, gyro_rads, now_s, quad_is_flying=False):
        dt = now_s - self.last_prediction_s
        if dt > 0:
            self._predict_dt(acc_ms2, gyro_rads, dt, quad_is_flying)
        self.last_prediction_s = now_s

    def _predict_dt(self, acc, gyro, dt, quad_is_flying):
        p = self.params
        R = self.R
        S = self.S
        dt2 = dt * dt

        A = np.eye(KC_STATE_DIM)
        # 位置 ← ボディ速度
        A[KC_STATE_X, KC_STATE_PX] = R[0, 0]*dt; A[KC_STATE_Y, KC_STATE_PX] = R[1, 0]*dt; A[KC_STATE_Z, KC_STATE_PX] = R[2, 0]*dt
        A[KC_STATE_X, KC_STATE_PY] = R[0, 1]*dt; A[KC_STATE_Y, KC_STATE_PY] = R[1, 1]*dt; A[KC_STATE_Z, KC_STATE_PY] = R[2, 1]*dt
        A[KC_STATE_X, KC_STATE_PZ] = R[0, 2]*dt; A[KC_STATE_Y, KC_STATE_PZ] = R[1, 2]*dt; A[KC_STATE_Z, KC_STATE_PZ] = R[2, 2]*dt

        # 位置 ← 姿勢誤差
        A[KC_STATE_X, KC_STATE_D0] = (S[KC_STATE_PY]*R[0, 2] - S[KC_STATE_PZ]*R[0, 1])*dt
        A[KC_STATE_Y, KC_STATE_D0] = (S[KC_STATE_PY]*R[1, 2] - S[KC_STATE_PZ]*R[1, 1])*dt
        A[KC_STATE_Z, KC_STATE_D0] = (S[KC_STATE_PY]*R[2, 2] - S[KC_STATE_PZ]*R[2, 1])*dt
        A[KC_STATE_X, KC_STATE_D1] = (-S[KC_STATE_PX]*R[0, 2] + S[KC_STATE_PZ]*R[0, 0])*dt
        A[KC_STATE_Y, KC_STATE_D1] = (-S[KC_STATE_PX]*R[1, 2] + S[KC_STATE_PZ]*R[1, 0])*dt
        A[KC_STATE_Z, KC_STATE_D1] = (-S[KC_STATE_PX]*R[2, 2] + S[KC_STATE_PZ]*R[2, 0])*dt
        A[KC_STATE_X, KC_STATE_D2] = (S[KC_STATE_PX]*R[0, 1] - S[KC_STATE_PY]*R[0, 0])*dt
        A[KC_STATE_Y, KC_STATE_D2] = (S[KC_STATE_PX]*R[1, 1] - S[KC_STATE_PY]*R[1, 0])*dt
        A[KC_STATE_Z, KC_STATE_D2] = (S[KC_STATE_PX]*R[2, 1] - S[KC_STATE_PY]*R[2, 0])*dt

        # ボディ速度 ← ボディ速度 (ジャイロの角速度カップリング, 飛行中はドラッグも)
        gx, gy, gz = gyro
        drag = (p.dragB_x, p.dragB_y, p.dragB_z) if quad_is_flying else (0.0, 0.0, 0.0)
        A[KC_STATE_PX, KC_STATE_PX] = 1 - dt*drag[0]
        A[KC_STATE_PY, KC_STATE_PX] = -gz*dt
        A[KC_STATE_PZ, KC_STATE_PX] = gy*dt
        A[KC_STATE_PX, KC_STATE_PY] = gz*dt
        A[KC_STATE_PY, KC_STATE_PY] = 1 - dt*drag[1]
        A[KC_STATE_PZ, KC_STATE_PY] = -gx*dt
        A[KC_STATE_PX, KC_STATE_PZ] = -gy*dt
        A[KC_STATE_PY, KC_STATE_PZ] = gx*dt
        A[KC_STATE_PZ, KC_STATE_PZ] = 1 - dt*drag[2]

        # ボディ速度 ← 姿勢誤差 (重力の投影)
        A[KC_STATE_PX, KC_STATE_D0] = 0.0
        A[KC_STATE_PY, KC_STATE_D0] = -GRAVITY_MAGNITUDE*R[2, 2]*dt
        A[KC_STATE_PZ, KC_STATE_D0] = GRAVITY_MAGNITUDE*R[2, 1]*dt
        A[KC_STATE_PX, KC_STATE_D1] = GRAVITY_MAGNITUDE*R[2, 2]*dt
        A[KC_STATE_PY, KC_STATE_D1] = 0.0
        A[KC_STATE_PZ, KC_STATE_D1] = -GRAVITY_MAGNITUDE*R[2, 0]*dt
        A[KC_STATE_PX, KC_STATE_D2] = -GRAVITY_MAGNITUDE*R[2, 1]*dt
        A[KC_STATE_PY, KC_STATE_D2] = GRAVITY_MAGNITUDE*R[2, 0]*dt
        A[KC_STATE_PZ, KC_STATE_D2] = 0.0

        # 姿勢誤差 ← 姿勢誤差 (2次近似, Covariance Correction Step for Kalman Filtering with an Attitude)
        d0, d1, d2 = gx*dt/2, gy*dt/2, gz*dt/2
        A[KC_STATE_D0:KC_STATE_D2+1, KC_STATE_D0:KC_STATE_D2+1] = \
            _small_angle_rot_cov_matrix(d0, d1, d2)[KC_STATE_D0:KC_STATE_D2+1, KC_STATE_D0:KC_STATE_D2+1]

        # ====== 共分散の伝播 ======
        self.P = A @ self.P @ A.T

        # ====== 状態の伝播 ======
        ax, ay, az = acc
        if quad_is_flying:
            zacc = az
            dx, dy = S[KC_STATE_PX]*dt, S[KC_STATE_PY]*dt
            dz = S[KC_STATE_PZ]*dt + zacc*dt2/2.0
        else:
            dx = S[KC_STATE_PX]*dt + ax*dt2/2.0
            dy = S[KC_STATE_PY]*dt + ay*dt2/2.0
            dz = S[KC_STATE_PZ]*dt + az*dt2/2.0

        S[KC_STATE_X] += R[0, 0]*dx + R[0, 1]*dy + R[0, 2]*dz
        S[KC_STATE_Y] += R[1, 0]*dx + R[1, 1]*dy + R[1, 2]*dz
        S[KC_STATE_Z] += R[2, 0]*dx + R[2, 1]*dy + R[2, 2]*dz - GRAVITY_MAGNITUDE*dt2/2.0

        pxo, pyo, pzo = S[KC_STATE_PX], S[KC_STATE_PY], S[KC_STATE_PZ]
        if quad_is_flying:
            vcop_x = pxo + gy*p.cop_z - gz*p.cop_y
            vcop_y = pyo + gz*p.cop_x - gx*p.cop_z
            vcop_z = pzo + gx*p.cop_y - gy*p.cop_x
            S[KC_STATE_PX] += dt*(gz*pyo - gy*pzo - GRAVITY_MAGNITUDE*R[2, 0] - p.dragB_x*vcop_x)
            S[KC_STATE_PY] += dt*(-gz*pxo + gx*pzo - GRAVITY_MAGNITUDE*R[2, 1] - p.dragB_y*vcop_y)
            S[KC_STATE_PZ] += dt*(zacc + gy*pxo - gx*pyo - GRAVITY_MAGNITUDE*R[2, 2] - p.dragB_z*vcop_z)
        else:
            S[KC_STATE_PX] += dt*(ax + gz*pyo - gy*pzo - GRAVITY_MAGNITUDE*R[2, 0])
            S[KC_STATE_PY] += dt*(ay - gz*pxo + gx*pzo - GRAVITY_MAGNITUDE*R[2, 1])
            S[KC_STATE_PZ] += dt*(az + gy*pxo - gx*pyo - GRAVITY_MAGNITUDE*R[2, 2])

        # ====== 姿勢 (クォータニオン) の伝播 ======
        dtwx, dtwy, dtwz = gx*dt, gy*dt, gz*dt
        angle = np.sqrt(dtwx**2 + dtwy**2 + dtwz**2) + EPS
        ca, sa = np.cos(angle/2.0), np.sin(angle/2.0)
        dq = np.array([ca, sa*dtwx/angle, sa*dtwy/angle, sa*dtwz/angle])
        q_new = _quat_mult(dq, self.q)

        if not quad_is_flying:
            keep = 1.0 - p.attitudeReversion
            q_new = keep*q_new + p.attitudeReversion*self.initial_q

        self.q = q_new / (np.linalg.norm(q_new) + EPS)
        self.is_updated = True

    # ------------------------------------------------------------------
    def add_process_noise(self, now_s):
        p = self.params
        dt = now_s - self.last_process_noise_s
        if dt <= 0:
            return
        dt2 = dt*dt
        self.P[KC_STATE_X, KC_STATE_X] += (p.procNoiseAcc_xy*dt2 + p.procNoiseVel*dt + p.procNoisePos) ** 2
        self.P[KC_STATE_Y, KC_STATE_Y] += (p.procNoiseAcc_xy*dt2 + p.procNoiseVel*dt + p.procNoisePos) ** 2
        self.P[KC_STATE_Z, KC_STATE_Z] += (p.procNoiseAcc_z*dt2 + p.procNoiseVel*dt + p.procNoisePos) ** 2
        self.P[KC_STATE_PX, KC_STATE_PX] += (p.procNoiseAcc_xy*dt + p.procNoiseVel) ** 2
        self.P[KC_STATE_PY, KC_STATE_PY] += (p.procNoiseAcc_xy*dt + p.procNoiseVel) ** 2
        self.P[KC_STATE_PZ, KC_STATE_PZ] += (p.procNoiseAcc_z*dt + p.procNoiseVel) ** 2
        self.P[KC_STATE_D0, KC_STATE_D0] += (p.measNoiseGyro_rollpitch*dt + p.procNoiseAtt) ** 2
        self.P[KC_STATE_D1, KC_STATE_D1] += (p.measNoiseGyro_rollpitch*dt + p.procNoiseAtt) ** 2
        self.P[KC_STATE_D2, KC_STATE_D2] += (p.measNoiseGyro_yaw*dt + p.procNoiseAtt) ** 2
        self._clip_covariance()
        self.last_process_noise_s = now_s

    # ------------------------------------------------------------------
    def finalize(self):
        """姿勢誤差 D0..D2 をクォータニオンへ畳み込み、Rを更新する (kalmanCoreFinalize)。"""
        if not self.is_updated:
            return False

        v0, v1, v2 = self.S[KC_STATE_D0], self.S[KC_STATE_D1], self.S[KC_STATE_D2]
        if (abs(v0) > 0.1e-3 or abs(v1) > 0.1e-3 or abs(v2) > 0.1e-3) and \
           (abs(v0) < 10 and abs(v1) < 10 and abs(v2) < 10):
            angle = np.sqrt(v0**2 + v1**2 + v2**2) + EPS
            ca, sa = np.cos(angle/2.0), np.sin(angle/2.0)
            dq = np.array([ca, sa*v0/angle, sa*v1/angle, sa*v2/angle])
            q_new = _quat_mult(dq, self.q)
            self.q = q_new / (np.linalg.norm(q_new) + EPS)

            A = _small_angle_rot_cov_matrix(v0/2, v1/2, v2/2)
            self.P = A @ self.P @ A.T

        self.R = _quat_to_R(self.q)
        self.S[KC_STATE_D0] = self.S[KC_STATE_D1] = self.S[KC_STATE_D2] = 0.0
        self._clip_covariance()
        self.is_updated = False
        return True

    # ------------------------------------------------------------------
    def _clip_covariance(self):
        P = self.P
        P[:] = 0.5*P + 0.5*P.T
        diag = np.diag(P).copy()
        diag[diag < MIN_COVARIANCE] = MIN_COVARIANCE
        np.fill_diagonal(P, diag)
        P[P > MAX_COVARIANCE] = MAX_COVARIANCE
        P[P < -MAX_COVARIANCE] = -MAX_COVARIANCE

    def scalar_update(self, H, error, std_meas_noise):
        """kalmanCoreScalarUpdate の移植 (Joseph形式の共分散更新)。

        H: (KC_STATE_DIM,) の観測行列, error: innovation (measured - predicted),
        std_meas_noise: 観測ノイズの標準偏差。
        """
        H = np.asarray(H, dtype=float)
        Pm = self.P
        PHT = Pm @ H
        R = std_meas_noise ** 2
        HPHR = R + float(H @ PHT)

        K = PHT / HPHR
        self.S = self.S + K * error

        I = np.eye(KC_STATE_DIM)
        KH = np.outer(K, H)
        self.P = (I - KH) @ Pm @ (I - KH).T + R * np.outer(K, K)
        self._clip_covariance()
        self.is_updated = True

    # ------------------------------------------------------------------
    # measurement models (mm_*.c の移植)
    # ------------------------------------------------------------------
    def update_with_yaw_error(self, yaw_error_rad, std_dev_rad):
        """mm_yaw_error.c の kalmanCoreUpdateWithYawError の移植。

        yaw_error_rad: 「外部から得たyaw基準」に対して、現在の内部姿勢が
        どれだけyaw方向にずれているか (小角, ラジアン)。D2 (yaw方向の姿勢誤差
        状態) に対するスカラー観測として扱う。
        """
        H = np.zeros(KC_STATE_DIM)
        H[KC_STATE_D2] = 1.0
        self.scalar_update(H, self.S[KC_STATE_D2] - yaw_error_rad, std_dev_rad)

    def update_with_flow(self, flow_dt, dpixel_x, dpixel_y, std_dev_x, std_dev_y,
                          gyro_rads, flowdeck_pos=(0.0, 0.0, 0.0)):
        """mm_flow.c の kalmanCoreUpdateWithFlow の移植 (オプション, flow deck用)。"""
        Npix = 35.0
        thetapix = 0.71674
        gx, gy, gz = gyro_rads
        dx_b, dy_b = self.S[KC_STATE_PX], self.S[KC_STATE_PY]
        z_g = max(self.S[KC_STATE_Z], 0.1)

        fx, fy, fz = flowdeck_pos
        v_cam_bx = dx_b + (gy*fz - gz*fy)
        v_cam_by = dy_b + (gz*fx - gx*fz)

        predicted_nx = (flow_dt*Npix/thetapix) * ((v_cam_bx*self.R[2, 2]/z_g) - gy)
        measured_nx = dpixel_x
        hx = np.zeros(KC_STATE_DIM)
        hx[KC_STATE_Z] = (Npix*flow_dt/thetapix) * ((self.R[2, 2]*v_cam_bx) / (-z_g*z_g))
        hx[KC_STATE_PX] = (Npix*flow_dt/thetapix) * (self.R[2, 2]/z_g)
        self.scalar_update(hx, measured_nx - predicted_nx, std_dev_x)

        predicted_ny = (flow_dt*Npix/thetapix) * ((v_cam_by*self.R[2, 2]/z_g) + gx)
        measured_ny = dpixel_y
        hy = np.zeros(KC_STATE_DIM)
        hy[KC_STATE_Z] = (Npix*flow_dt/thetapix) * ((self.R[2, 2]*v_cam_by) / (-z_g*z_g))
        hy[KC_STATE_PY] = (Npix*flow_dt/thetapix) * (self.R[2, 2]/z_g)
        self.scalar_update(hy, measured_ny - predicted_ny, std_dev_y)

    # ------------------------------------------------------------------
    def externalize_ypr_deg(self):
        """kalmanCoreExternalizeState のyaw/pitch/roll算出部分の移植 [deg] (CF2規約でpitchは符号反転)。"""
        w, x, y, z = self.q
        yaw = np.arctan2(2*(x*y + w*z), w*w + x*x - y*y - z*z)
        pitch = np.arcsin(np.clip(-2*(x*z - w*y), -1.0, 1.0))
        roll = np.arctan2(2*(y*z + w*x), w*w - x*x - y*y + z*z)
        return np.degrees([yaw, -pitch, roll])
