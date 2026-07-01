#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_to_csv.py が出力したCSVから VL53L8CX の 4x4 深度マップを作成する。

各センサーは 16 ゾーン (d0..d15) の距離[mm]と、対応するステータス (st0..st15) を持つ。
ゾーン番号 i は 4x4 グリッド上で row = i // 4, col = i % 4 に対応する。

使い方:
    # 全フレーム平均の深度マップ (sensor0 と sensor1) を表示
    python3 depth_map.py log16.csv

    # 特定フレーム(行番号)だけを表示
    python3 depth_map.py log16.csv --frame 0

    # sensor0 だけ、無効ゾーン(status!=5)をマスクして画像保存
    python3 depth_map.py log16.csv --sensor 0 --mask-invalid -o depth.png

オプションでセンサーの向きに合わせた反転 (--flip-x / --flip-y) も可能。
"""
import argparse
import os
import sys

import csv as csvmod

import numpy as np

# pandas / matplotlib は静的マップ表示(-o)でのみ使う重い依存なので遅延importする。
# GIF生成(--gif)は numpy + Pillow + 標準csv だけで動く。

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "figs")
GIF_DIR = os.path.join(SCRIPT_DIR, "gif")

# 有効とみなす VL53L8CX の target_status (5 = レンジ有効)
VALID_STATUS = 5


def load_csv(path):
    """CSVを解決して読み込む。直接見つからなければ csv/ 内も探す。"""
    if not os.path.isfile(path):
        candidate = os.path.join(CSV_DIR, os.path.basename(path))
        if os.path.isfile(candidate):
            path = candidate
        else:
            sys.exit(f"ファイルが見つかりません: {path} (csv/ も確認しました)")
    import pandas as pd
    return pd.read_csv(path)


def resolve_csv_path(path):
    """CSVパスを解決する(読み込みはしない)。"""
    if not os.path.isfile(path):
        candidate = os.path.join(CSV_DIR, os.path.basename(path))
        if os.path.isfile(candidate):
            return candidate
        sys.exit(f"ファイルが見つかりません: {path} (csv/ も確認しました)")
    return path


def sensor_columns(df, sensor, prefix):
    """指定センサーの d0..d15 / st0..st15 列名を返す。無ければ None。"""
    cols = [f"vl53l8cx_s{sensor}.{prefix}{i}" for i in range(16)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None
    return cols


def build_grid(df, sensor, frame, mask_invalid, flip_x, flip_y):
    """センサーの 4x4 距離グリッド(マスク済み)を返す。"""
    dist_cols = sensor_columns(df, sensor, "d")
    if dist_cols is None:
        return None

    if frame is None:
        # 全フレーム平均。マスク時は無効値を NaN にしてから nanmean。
        dist = df[dist_cols].to_numpy(dtype=float)
        if mask_invalid:
            st_cols = sensor_columns(df, sensor, "st")
            if st_cols is not None:
                status = df[st_cols].to_numpy()
                dist[status != VALID_STATUS] = np.nan
        with np.errstate(invalid="ignore"):
            values = np.nanmean(dist, axis=0)
    else:
        if frame < 0 or frame >= len(df):
            sys.exit(f"--frame {frame} は範囲外です (0..{len(df) - 1})")
        values = df.loc[frame, dist_cols].to_numpy(dtype=float)
        if mask_invalid:
            st_cols = sensor_columns(df, sensor, "st")
            if st_cols is not None:
                status = df.loc[frame, st_cols].to_numpy()
                values[status != VALID_STATUS] = np.nan

    grid = values.reshape(4, 4)  # zone = row*4 + col
    grid = grid[::-1, ::-1]  # センサーの向きに合わせ既定で上下左右反転
    if flip_x:
        grid = grid[:, ::-1]
    if flip_y:
        grid = grid[::-1, :]
    return grid


# ---- 軽量GIF生成 (numpy + Pillow + 標準csv のみ。pandas/matplotlib 不要) ----

# 近い=赤, 遠い=青 のカラーマップ用アンカー (t, (R,G,B))。t は 0(近)→1(遠)。
_CMAP_ANCHORS = [
    (0.00, (180, 0, 0)),      # 近い: 赤
    (0.25, (240, 140, 0)),    # オレンジ
    (0.50, (220, 210, 0)),    # 黄
    (0.75, (0, 170, 90)),     # 緑
    (1.00, (0, 70, 200)),     # 遠い: 青
]
_INVALID_COLOR = (60, 60, 60)  # 無効ゾーン: 暗いグレー


def _colormap(t):
    """t in [0,1] -> (R,G,B)。アンカー間を線形補間する。"""
    if t <= 0:
        return _CMAP_ANCHORS[0][1]
    if t >= 1:
        return _CMAP_ANCHORS[-1][1]
    for (t0, c0), (t1, c1) in zip(_CMAP_ANCHORS, _CMAP_ANCHORS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
    return _CMAP_ANCHORS[-1][1]


def _read_csv_columns(path):
    """CSVを読み、{列名: [float,...]} と行数を返す(標準csvのみ)。"""
    with open(path, newline="") as f:
        reader = csvmod.DictReader(f)
        cols = {name: [] for name in reader.fieldnames}
        n = 0
        for row in reader:
            for name in reader.fieldnames:
                val = row[name]
                try:
                    cols[name].append(float(val))
                except (TypeError, ValueError):
                    cols[name].append(np.nan)
            n += 1
    return cols, n


def _frame_grid(cols, sensor, frame, mask_invalid, flip_x, flip_y):
    """指定フレームの 4x4 距離グリッドを返す。列が無ければ None。"""
    dist_names = [f"vl53l8cx_s{sensor}.d{i}" for i in range(16)]
    if any(name not in cols for name in dist_names):
        return None
    vals = np.array([cols[name][frame] for name in dist_names], dtype=float)
    if mask_invalid:
        st_names = [f"vl53l8cx_s{sensor}.st{i}" for i in range(16)]
        if all(name in cols for name in st_names):
            st = np.array([cols[name][frame] for name in st_names])
            vals[st != VALID_STATUS] = np.nan
    grid = vals.reshape(4, 4)
    grid = grid[::-1, ::-1]  # センサーの向きに合わせ既定で上下左右反転
    if flip_x:
        grid = grid[:, ::-1]
    if flip_y:
        grid = grid[::-1, :]
    return grid


def generate_gif(args):
    """時間変化する4x4深度マップをGIFとして gif/ フォルダに保存する。"""
    from PIL import Image, ImageDraw

    path = resolve_csv_path(args.csv)
    cols, nrows = _read_csv_columns(path)
    if nrows == 0:
        sys.exit("CSVにデータ行がありません。")

    # 対象センサーの決定
    if args.sensor is not None:
        sensors = [args.sensor]
    else:
        sensors = [0, 1]
    sensors = [s for s in sensors
               if all(f"vl53l8cx_s{s}.d{i}" in cols for i in range(16))]
    if not sensors:
        sys.exit("ToF列 (vl53l8cx_s*.d*) がCSVに見つかりません。")

    frames = list(range(0, nrows, max(1, args.step)))

    # カラースケール(全フレーム共通)
    vmin, vmax = args.vmin, args.vmax
    if vmin is None or vmax is None:
        allv = []
        for s in sensors:
            for i in range(16):
                arr = np.array(cols[f"vl53l8cx_s{s}.d{i}"], dtype=float)
                if args.mask_invalid and f"vl53l8cx_s{s}.st{i}" in cols:
                    st = np.array(cols[f"vl53l8cx_s{s}.st{i}"])
                    arr = arr.copy()
                    arr[st != VALID_STATUS] = np.nan
                allv.append(arr)
        stacked = np.concatenate(allv)
        if vmin is None:
            vmin = float(np.nanmin(stacked))
        if vmax is None:
            vmax = float(np.nanmax(stacked))
    span = (vmax - vmin) if vmax > vmin else 1.0

    # 描画レイアウト
    CELL = 70          # 1ゾーンの一辺[px]
    GRID = CELL * 4
    TITLE_H = 28       # センサー名+フレーム番号の帯
    PAD = 12
    panel_w = GRID + PAD
    width = PAD + len(sensors) * panel_w
    height = TITLE_H + GRID + PAD

    images = []
    for fi in frames:
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for si, s in enumerate(sensors):
            x0 = PAD + si * panel_w
            grid = _frame_grid(cols, s, fi, args.mask_invalid,
                               args.flip_x, args.flip_y)
            draw.text((x0, 6), f"sensor {s}  frame {fi}", fill=(0, 0, 0))
            for r in range(4):
                for c in range(4):
                    v = grid[r, c]
                    cx0 = x0 + c * CELL
                    cy0 = TITLE_H + r * CELL
                    if np.isnan(v):
                        color = _INVALID_COLOR
                        label = "-"
                    else:
                        t = (v - vmin) / span
                        color = _colormap(t)
                        label = f"{v:.0f}"
                    draw.rectangle([cx0, cy0, cx0 + CELL, cy0 + CELL],
                                   fill=color, outline=(255, 255, 255))
                    # 文字色は背景輝度で白黒切替
                    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    tc = (0, 0, 0) if lum > 140 else (255, 255, 255)
                    tw = draw.textlength(label)
                    draw.text((cx0 + (CELL - tw) / 2, cy0 + CELL / 2 - 6),
                              label, fill=tc)
        images.append(img)

    # 出力先: gif/ フォルダ
    out = args.gif
    if not os.path.dirname(out):
        os.makedirs(GIF_DIR, exist_ok=True)
        out = os.path.join(GIF_DIR, out)
    else:
        os.makedirs(os.path.dirname(out), exist_ok=True)
    if not out.lower().endswith(".gif"):
        out += ".gif"

    duration = int(round(1000.0 / args.fps))
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=duration, loop=0, optimize=False)
    print(f"GIFを保存しました: {out}")
    print(f"  フレーム数: {len(images)} / {args.fps} fps / スケール {vmin:.0f}-{vmax:.0f} mm")
    print(f"  対象センサー: {sensors}")


def main():
    parser = argparse.ArgumentParser(description="VL53L8CX 4x4 depth map from uSD CSV")
    parser.add_argument("csv", help="入力CSV (例: log16.csv)")
    parser.add_argument("--sensor", type=int, default=None,
                        help="表示するセンサー番号。省略時は存在する全センサー(0,1)")
    parser.add_argument("--frame", type=int, default=None,
                        help="表示する行(サンプル)番号。省略時は全フレーム平均")
    parser.add_argument("--mask-invalid", action="store_true",
                        help="status!=5 のゾーンを除外(NaN)する")
    parser.add_argument("--flip-x", action="store_true", help="左右反転")
    parser.add_argument("--flip-y", action="store_true", help="上下反転")
    parser.add_argument("--vmin", type=float, default=None, help="カラースケール最小[mm]")
    parser.add_argument("--vmax", type=float, default=None, help="カラースケール最大[mm]")
    parser.add_argument("-o", "--output", default=None, help="画像保存パス(省略時は表示のみ)")
    parser.add_argument("--gif", default=None,
                        help="時間変化のGIFを保存するパス。指定すると全フレームをアニメーション化する")
    parser.add_argument("--fps", type=float, default=20.0, help="GIFのフレームレート (既定: 20)")
    parser.add_argument("--step", type=int, default=1, help="GIFで何フレームおきに描くか (既定: 1)")
    args = parser.parse_args()

    # GIFは pandas/matplotlib 不要の軽量経路で処理する
    if args.gif:
        generate_gif(args)
        return

    import matplotlib.pyplot as plt

    df = load_csv(args.csv)

    if args.sensor is not None:
        sensors = [args.sensor]
    else:
        sensors = [s for s in (0, 1) if sensor_columns(df, s, "d") is not None]
    if not sensors:
        sys.exit("ToF列 (vl53l8cx_s*.d*) がCSVに見つかりません。")

    grids = []
    for s in sensors:
        g = build_grid(df, s, args.frame, args.mask_invalid, args.flip_x, args.flip_y)
        if g is None:
            print(f"  警告: sensor{s} の列が揃っていません。スキップします。")
            continue
        grids.append((s, g))
    if not grids:
        sys.exit("有効なセンサーがありませんでした。")

    fig, axes = plt.subplots(1, len(grids), figsize=(5 * len(grids), 4.5), squeeze=False)
    title_frame = "mean of all frames" if args.frame is None else f"frame {args.frame}"
    for ax, (s, grid) in zip(axes[0], grids):
        im = ax.imshow(grid, cmap="viridis_r", vmin=args.vmin, vmax=args.vmax,
                       interpolation="nearest")
        ax.set_title(f"sensor {s}  ({title_frame})")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        # 各セルに距離値[mm]を表示
        for r in range(4):
            for c in range(4):
                v = grid[r, c]
                txt = "-" if np.isnan(v) else f"{v:.0f}"
                ax.text(c, r, txt, ha="center", va="center",
                        color="white", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="distance [mm]")
    fig.tight_layout()

    if args.output:
        out = args.output
        if not os.path.dirname(out):
            os.makedirs(OUT_DIR, exist_ok=True)
            out = os.path.join(OUT_DIR, out)
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"保存しました: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
