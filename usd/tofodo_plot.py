#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファームウェア (vl53l8cx_deck.c / tof_odometry.c) が計算した ToF オドメトリの
ログから、tof_odometry.py と同じ8枚のグラフを描画する。

tof_odometry.py との違い:
  - 平面抽出・ΔR・Δt の計算は一切行わない。すべて機体上で計算済みの
    tofodo.* をそのまま読んで描くだけ。
  - 生の16ゾーン距離 (vl53l8cx_s*.d*) は不要。

前提とするログ変数 (tools/usdlog/config.txt の tofodo.* 一式):
    tofodo.d0..d10      各センサーの平面距離 [mm]        (抽出失敗は NaN)
    tofodo.pl0..pl10    平面性スコア λ3/Σλ
    tofodo.az0..az10    法線の方位角 [deg]
    tofodo.dyaw/dpitch/droll, tofodo.dx/dy/dz   フレーム間の ΔR / Δt
    tofodo.yaw/pitch/roll, tofodo.px/py/pz      累積回転 / 累積移動
    tofodo.rankR/rankT/npair                    退化診断
    tofodo.seq                                  フレーム番号 (重複行の除去に使う)
比較用 (あれば描く):
    gyro.x/y/z, stateEstimate.x/y/z

使い方:
    python3 log_to_csv.py log44            # SDのバイナリログ → csv/log44.csv
    python3 tofodo_plot.py log44.csv
    python3 tofodo_plot.py log44.csv -s 0 1 2 --tmax 20 -o out.png
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "figs")

NUM_SENSORS = 11
PLANARITY_EPS = 0.006  # tof_odometry.c と同じ値 (グラフ中に基準線として引く)
GATE_DEG = 1.0        # tof_odometry.c の s_gateDeg (ジャイロ整合チェックの閾値)

GYRO_COLS = ("gyro.x", "gyro.y", "gyro.z")   # roll/pitch/yaw の角速度 [deg/s]
GYRO_BIAS_SEC = 0.5                          # 先頭この秒数の平均をバイアスとして除去


def load(path):
    """CSVを読む。パスが見つからなければ csv/ フォルダも探す。"""
    import pandas as pd

    if not os.path.isfile(path):
        cand = os.path.join(CSV_DIR, os.path.basename(path))
        if not os.path.isfile(cand):
            sys.exit(f"ファイルが見つかりません: {path}")
        path = cand
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        sys.exit("'timestamp' 列がありません。log_to_csv.py の出力を指定してください。")
    if not any(c.startswith("tofodo.") for c in df.columns):
        sys.exit(
            "tofodo.* 列がありません。\n"
            "ファームウェアが ToF オドメトリを出力しているか、"
            "tools/usdlog/config.txt に tofodo.* が入っているか確認してください。"
        )
    return df, path


def frame_rows(df):
    """
    ファームウェアのフレーム (5Hz) ごとに1行だけ残したインデックスを返す。

    ログは 250Hz で回るので同じ推定結果が約50行ずつ重複する。そのまま描くと
    「1フレームあたりの Δ」の意味が崩れるため tofodo.seq が変わる境目で間引く。
    各 seq の "最後" の行を採るのは、seq が変わった直後の行だと 50個の log 変数を
    書き終える前にサンプリングされている可能性があるため。
    """
    if "tofodo.seq" not in df.columns:
        print("警告: tofodo.seq が無いので全行をそのまま描画します", file=sys.stderr)
        return np.arange(len(df))

    seq = df["tofodo.seq"].to_numpy()
    # 各 seq ブロックの最終行 = 次の行で seq が変わる行、および最終行
    last = np.flatnonzero(np.diff(seq) != 0)
    return np.append(last, len(df) - 1)


def active_sensors(df, requested):
    """平面が1度でも取れたセンサーだけを返す (未実装/未接続の空プロットを避ける)。"""
    out = []
    for s in requested:
        col = f"tofodo.d{s}"
        if col not in df.columns:
            continue
        if np.isfinite(df[col].to_numpy(dtype=float)).any():
            out.append(s)
    return out


def gyro_reference(df, t):
    """ジャイロ角速度から比較用の累積回転[deg]を作る。列が無ければ None。

    gyro.x/y/z = roll/pitch/yaw 方向の角速度 [deg/s]。
    ToF 側と同じ (yaw, pitch, roll) の順に並べ替えて返す。
    tof_odometry.py の同名関数と同じ処理を、ログの全レート (250Hz) で行う。
    """
    if not all(c in df.columns for c in GYRO_COLS):
        return None

    # [roll, pitch, yaw] rate → [yaw, pitch, roll] rate
    rate = df[list(GYRO_COLS)].to_numpy(dtype=float)[:, ::-1]

    # 静止しているはずの先頭区間の平均をバイアスとして差し引く
    n_bias = max(int(np.searchsorted(t - t[0], GYRO_BIAS_SEC)), 1)
    rate = rate - np.nanmean(rate[:n_bias], axis=0)

    dt_s = np.diff(t, prepend=t[0])
    d_ypr = rate * dt_s[:, None]
    return dict(d_ypr=d_ypr, ypr=np.nancumsum(d_ypr, axis=0))


def col(df, name, idx=None):
    """列を float 配列で取り出す。無い列は NaN 埋めで返す。"""
    if name not in df.columns:
        return np.full(len(df) if idx is None else len(idx), np.nan)
    v = df[name].to_numpy(dtype=float)
    return v if idx is None else v[idx]


def apply_planarity_eps(df, eps):
    """記録済みの平面を、より厳しい平面性閾値で事後的に足切りする。

    機体上の閾値 (tof_odometry.c の PLANARITY_EPS) を下げたときに
    「どのセンサーがどれだけ残るか」を、飛ばし直す前に確認するための機能。

    注意: 影響するのは平面そのもののパネル (d_i / 平面性 / 法線方位角) だけ。
    ΔR・Δt・累積値は機体上で当時の閾値のまま計算済みで、ログには法線の
    方位角しか無く3成分の法線ベクトルが復元できないため再計算できない。
    正しい ΔR/Δt を得るには PLANARITY_EPS を変えて飛ばし直す必要がある。
    """
    df = df.copy()
    n_cut = 0
    for s in range(NUM_SENSORS):
        pcol = f"tofodo.pl{s}"
        if pcol not in df.columns:
            continue
        pl = df[pcol].to_numpy(dtype=float)
        cut = np.isfinite(pl) & (pl > eps)
        n_cut += int(cut.sum())
        for c in (f"tofodo.d{s}", pcol, f"tofodo.az{s}"):
            if c in df.columns:
                v = np.array(df[c].to_numpy(dtype=float), copy=True)
                v[cut] = np.nan
                df[c] = v
    return df, n_cut


def plot(df, sensors, out_path, tmax=None, eps_note=None):
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

    t_all = df["timestamp"].to_numpy(dtype=float) / 1000.0   # [s]
    gyro = gyro_reference(df, t_all)
    t_all = t_all - t_all[0]

    idx = frame_rows(df)          # ファームウェアのフレーム単位 (5Hz)
    t = t_all[idx]

    # 事後で ε を変えた場合、ΔR/Δt 系のパネルは機体上の値のままなので明示する
    onboard = "  ※ΔR/Δtは機体上のεのまま" if eps_note else ""

    fig, ax = plt.subplots(5, 2, figsize=(15, 17), sharex=True)
    ypr_colors = ["tab:blue", "tab:orange", "tab:green"]
    # センサーが最大11個なので色を明示的に振り分ける。
    # tab20 は濃淡が交互になり薄い色が読めないので、tab10 + 黒 を使う。
    palette = list(plt.get_cmap("tab10").colors) + ["black"]
    scolor = {s: palette[i % len(palette)] for i, s in enumerate(sensors)}
    slegend = dict(fontsize=6, ncol=2) if len(sensors) > 6 else dict(fontsize=8)

    # (0,0) 各センサーの平面距離 d_i
    for s in sensors:
        ax[0, 0].plot(t, col(df, f"tofodo.d{s}", idx), lw=1,
                      color=scolor[s], label=f"sensor{s}")
    ax[0, 0].set_ylabel("plane distance $d_i$ [mm]")
    ax[0, 0].set_title("3.1 PCA+RANSAC 平面抽出: センサー原点→平面の垂直距離")
    ax[0, 0].legend(**slegend)

    # (0,1) 平面性スコア
    for s in sensors:
        ax[0, 1].plot(t, col(df, f"tofodo.pl{s}", idx), lw=1,
                      color=scolor[s], label=f"sensor{s}")
    ax[0, 1].axhline(eps_note if eps_note else PLANARITY_EPS,
                     color="k", ls="--", lw=0.8, label="ε")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel(r"planarity $\lambda_3/\Sigma\lambda$")
    ax[0, 1].set_title("平面性スコア (小さいほど良い平面)")
    ax[0, 1].legend(**slegend)

    # (1,0) フレーム間回転量
    for i, name in enumerate(["yaw", "pitch", "roll"]):
        ax[1, 0].plot(t, col(df, f"tofodo.d{name}", idx), lw=1,
                      color=ypr_colors[i], label=f"ToF Δ{name}")
    if gyro is not None:
        # ジャイロ側も ToF と同じフレーム間隔で積分し直して比較する
        d_gyro = np.zeros((len(idx), 3))
        cum = np.nancumsum(gyro["d_ypr"], axis=0)
        d_gyro[1:] = cum[idx][1:] - cum[idx][:-1]
        for i, name in enumerate(["yaw", "pitch", "roll"]):
            ax[1, 0].plot(t, d_gyro[:, i], lw=0.8, ls="--", alpha=0.6,
                          color=ypr_colors[i], label=f"gyro Δ{name}")
    ax[1, 0].set_ylabel("Δrotation [deg/frame]")
    ax[1, 0].set_title("3.2 Kabsch による回転量 ΔR (フレーム間)"
                       + (" / ジャイロ角速度の積分" if gyro is not None else "")
                       + onboard)
    ax[1, 0].legend(fontsize=7, ncol=2)

    # (1,1) フレーム間移動量
    for name in ["x", "y", "z"]:
        ax[1, 1].plot(t, col(df, f"tofodo.d{name}", idx), lw=1, label=f"Δ{name}")
    ax[1, 1].set_ylabel("Δtranslation [mm/frame]")
    ax[1, 1].set_title("3.3 Point-to-Plane による移動量 Δt (フレーム間)" + onboard)
    ax[1, 1].legend(fontsize=8)

    # (2,0) 累積姿勢
    for i, name in enumerate(["yaw", "pitch", "roll"]):
        ax[2, 0].plot(t, col(df, f"tofodo.{name}", idx), lw=1.2,
                      color=ypr_colors[i], label=f"ToF {name}")
    if gyro is not None:
        for i, name in enumerate(["yaw", "pitch", "roll"]):
            ax[2, 0].plot(t_all, gyro["ypr"][:, i], lw=1.0, ls="--", alpha=0.75,
                          color=ypr_colors[i], label=f"gyro {name}")
    ax[2, 0].set_ylabel("cumulative rotation [deg]")
    ax[2, 0].set_title("累積回転量 (機体上で積分した ΔR)"
                       + (" と ジャイロ積分の比較" if gyro is not None else "")
                       + onboard)
    ax[2, 0].legend(fontsize=7, ncol=2)

    # (2,1) 累積移動量 (ToF) vs stateEstimate
    for name in ["x", "y", "z"]:
        ax[2, 1].plot(t, col(df, f"tofodo.p{name}", idx), lw=1.2, label=f"ToF {name}")
    for name in ["x", "y", "z"]:
        c = f"stateEstimate.{name}"
        if c in df.columns:
            v = df[c].to_numpy(dtype=float)
            ax[2, 1].plot(t_all, (v - v[0]) * 1000.0, lw=1, ls="--", alpha=0.7,
                          label=f"EKF {name}")
    ax[2, 1].set_ylabel("cumulative translation [mm]")
    ax[2, 1].set_title("累積移動量 (機体上で積分した Δt) と EKF の比較" + onboard)
    ax[2, 1].legend(fontsize=7, ncol=2)

    # (3,0) 観測可能性 (ランク)
    ax[3, 0].plot(t, col(df, "tofodo.rankT", idx), lw=1, label="rank(Δt)")
    ax[3, 0].plot(t, col(df, "tofodo.rankR", idx), lw=1, label="rank(ΔR)")
    ax[3, 0].plot(t, col(df, "tofodo.npair", idx), lw=1, ls=":", label="採用した対応数")
    if "tofodo.nfresh" in df.columns:
        ax[3, 0].plot(t, col(df, "tofodo.nfresh", idx), lw=1, ls="-.", alpha=0.7,
                      label="測距更新センサー数")
    ax[3, 0].set_ylabel("rank / count")
    ax[3, 0].set_title("幾何学的退化の診断 (rank<3 = その軸は不定)")
    ax[3, 0].legend(fontsize=8)

    # (3,1) 法線の向き (方位角) — 壁に対する相対角度
    for s in sensors:
        ax[3, 1].plot(t, col(df, f"tofodo.az{s}", idx), lw=1,
                      color=scolor[s], label=f"sensor{s}")
    ax[3, 1].set_ylabel("normal azimuth [deg]")
    ax[3, 1].set_title("抽出された法線ベクトルの方位角 (ボディ座標系)")
    ax[3, 1].legend(**slegend)


    # (4,0) ΔR の出所 (ToF / ジャイロ代替) の内訳
    SRC_LABEL = {0: "更新なし", 1: "ToF (Kabsch)",
                 2: "ジャイロ代替(乖離)", 3: "ジャイロ代替(対応なし)"}
    SRC_COLOR = {0: "0.6", 1: "tab:green", 2: "tab:red", 3: "tab:orange"}
    if "tofodo.rotsrc" in df.columns:
        src = col(df, "tofodo.rotsrc", idx)
        for v in (0, 1, 2, 3):
            m = (src == v)
            if m.any():
                ax[4, 0].plot(t[m], np.full(m.sum(), v), ".", ms=3,
                              color=SRC_COLOR[v],
                              label=f"{SRC_LABEL[v]} ({100.0*m.sum()/len(src):.0f}%)")
        ax[4, 0].set_yticks([0, 1, 2, 3])
        ax[4, 0].set_yticklabels([SRC_LABEL[v] for v in (0, 1, 2, 3)], fontsize=7)
        ax[4, 0].set_ylim(-0.5, 3.5)
    ax[4, 0].set_ylabel("ΔR の出所")
    ax[4, 0].set_xlabel("time [s]")
    ax[4, 0].set_title("回転量をどこから採ったか (ToF が使えた割合)")
    ax[4, 0].legend(fontsize=7, loc="center right")

    # (4,1) ToF と ジャイロ の食い違い
    if "tofodo.roterr" in df.columns:
        ax[4, 1].plot(t, col(df, "tofodo.roterr", idx), lw=1,
                      color="tab:red", label="合成 ΔR とジャイロの差")
    if "tofodo.gateerr" in df.columns:
        ax[4, 1].plot(t, col(df, "tofodo.gateerr", idx), lw=1, alpha=0.8,
                      color="tab:blue", label="採用した法線の最大乖離")
    if "tofodo.ngated" in df.columns:
        ax[4, 1].plot(t, col(df, "tofodo.ngated", idx), lw=0.8, ls=":",
                      color="tab:purple", label="ゲートで捨てた対応数")
    ax[4, 1].axhline(GATE_DEG, color="k", ls="--", lw=0.8, label=f"閾値 {GATE_DEG}deg")
    ax[4, 1].set_ylabel("deviation [deg] / count")
    ax[4, 1].set_xlabel("time [s]")
    ax[4, 1].set_title("ジャイロ整合チェック (閾値以上は棄却)")
    ax[4, 1].legend(fontsize=7)

    for a in ax.ravel():
        a.grid(alpha=0.3)

    # 表示範囲を絞る (着地時の外れ値でスケールが潰れるのを防ぐ)。
    if tmax is not None:
        for a in ax.ravel():
            a.set_xlim(t_all[0], tmax)
            ys = []
            for ln in a.get_lines():
                x, y = ln.get_xdata(), ln.get_ydata()
                if len(x) == len(y) and len(x) > 1:
                    ys.append(np.asarray(y, dtype=float)[np.asarray(x) <= tmax])
            ys = np.concatenate(ys) if ys else np.array([np.nan])
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


def summary(df, sensors, idx, path):
    print(f"入力: {path}  ({len(df)} 行, ファームウェアのフレーム {len(idx)} 個)")

    if "tofodo.seq" in df.columns:
        seq = df["tofodo.seq"].to_numpy()[idx]
        gaps = np.diff(seq)
        lost = int((gaps - 1)[gaps > 1].sum()) if len(gaps) else 0
        if lost:
            print(f"  警告: seq の飛び {lost} フレーム分 (SD書き込みが追いついていない可能性)")

    for s in sensors:
        v = col(df, f"tofodo.d{s}", idx)
        ok = int(np.isfinite(v).sum())
        print(f"  sensor{s:<2}: 平面抽出成功 {ok}/{len(idx)} フレーム "
              f"({100.0 * ok / max(len(idx), 1):.1f}%), 平均距離 "
              f"{np.nanmean(v) if ok else float('nan'):.1f} mm")

    npair = col(df, "tofodo.npair", idx)
    rank_t = col(df, "tofodo.rankT", idx)
    print(f"  ΔR/Δt 推定フレーム数 : {int((npair > 0).sum())}")
    print(f"  Δt のランク 3/2/1/0  : "
          f"{int((rank_t == 3).sum())}/{int((rank_t == 2).sum())}/"
          f"{int((rank_t == 1).sum())}/{int((rank_t == 0).sum())}")

    ypr = np.array([col(df, f"tofodo.{n}", idx)[-1] for n in ("yaw", "pitch", "roll")])
    pos = np.array([col(df, f"tofodo.p{n}", idx)[-1] for n in ("x", "y", "z")])
    print(f"  累積回転 (yaw,pitch,roll) [deg] : {ypr.round(2)}")
    print(f"  累積移動 (x,y,z) [mm]           : {pos.round(1)}")

    if "tofodo.calcUs" in df.columns:
        us = col(df, "tofodo.calcUs", idx)
        us = us[np.isfinite(us) & (us > 0)]
        if len(us):
            print(f"  機体上の計算時間 [us]           : "
                  f"median {np.median(us):.0f}, max {us.max():.0f} "
                  f"(測距周期 200000us に対して {100.0 * us.max() / 200000.0:.1f}%)")

    t = df["timestamp"].to_numpy(dtype=float) / 1000.0
    gyro = gyro_reference(df, t)
    if gyro is not None:
        print(f"  ジャイロ積分 (yaw,pitch,roll)    : {gyro['ypr'][-1].round(2)}")
        tof_ypr = np.stack([col(df, f"tofodo.{n}", idx)
                            for n in ("yaw", "pitch", "roll")], axis=1)
        rmse = np.sqrt(np.nanmean((tof_ypr - gyro["ypr"][idx]) ** 2, axis=0))
        print(f"  ToF vs ジャイロ RMSE [deg]      : {rmse.round(2)}")


def main():
    ap = argparse.ArgumentParser(
        description="ファームウェアが計算した ToF オドメトリのログを描画する")
    ap.add_argument("csv", help="log_to_csv.py が出力したCSV (例: log44.csv)")
    ap.add_argument("-s", "--sensors", type=int, nargs="+",
                    default=list(range(NUM_SENSORS)),
                    help="描画するセンサー番号 (既定: 0-10 のうちデータがあるもの)")
    ap.add_argument("-o", "--output", default=None, help="出力PNGパス")
    ap.add_argument("--tmax", type=float, default=None,
                    help="グラフの表示上限時刻 [s]")
    ap.add_argument("--planarity-eps", type=float, default=None,
                    help="平面性スコアの閾値を事後的に厳しくする (例: 0.01)。"
                         "平面のパネルにのみ反映され、ΔR/Δt は機体上の値のまま")
    args = ap.parse_args()

    df, path = load(args.csv)

    eps = args.planarity_eps
    if eps is not None:
        df, n_cut = apply_planarity_eps(df, eps)
        print(f"平面性スコア > {eps} の平面を事後的に除外しました ({n_cut} 件)")
        print("  ※ ΔR/Δt/累積値は機体上で当時の閾値のまま計算済みで再計算できません")
        print("     (ログには法線の方位角しか無く、3成分の法線が復元できないため)")

    sensors = active_sensors(df, args.sensors)
    if not sensors:
        sys.exit("平面が1度も抽出できていません (tofodo.d* が全て NaN)。")

    idx = frame_rows(df)
    summary(df, sensors, idx, path)

    base = os.path.splitext(os.path.basename(path))[0]
    out = args.output or os.path.join(OUT_DIR, f"{base}_tofodo.png")
    plot(df, sensors, out, tmax=args.tmax, eps_note=eps)


if __name__ == "__main__":
    main()
