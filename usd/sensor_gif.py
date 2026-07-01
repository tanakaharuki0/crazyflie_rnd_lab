#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指定したセンサー番号の VL53L8CX について、16ゾーン (4x4) の
距離[mm]とステータスの時間変化を1枚のGIFアニメーションにして出力する。

左パネル: 距離マップ (近い=赤 → 遠い=青)
右パネル: ステータスマップ (status値ごとに色分け)

ゾーン番号 i は 4x4 グリッド上で row = i // 4, col = i % 4 に対応する。
センサーの取り付け向きに合わせ、既定で上下左右を反転して表示する
(depth_map.py と同じ向き)。

依存: numpy, Pillow のみ (pandas/matplotlib 不要)。

使い方:
    # sensor 0 のGIFを gif/ に出力
    python3 sensor_gif.py log16.csv 0

    # sensor 8 を 5フレームおき・15fpsで出力
    python3 sensor_gif.py log16.csv 8 --step 5 --fps 15

    # 出力ファイル名を指定
    python3 sensor_gif.py log16.csv 1 -o sensor1.gif
"""
import argparse
import csv as csvmod
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
GIF_DIR = os.path.join(SCRIPT_DIR, "gif")

VALID_STATUS = 5

# 距離用カラーマップ (t: 0=近 → 1=遠)
_DIST_ANCHORS = [
    (0.00, (180, 0, 0)),
    (0.25, (240, 140, 0)),
    (0.50, (220, 210, 0)),
    (0.75, (0, 170, 90)),
    (1.00, (0, 70, 200)),
]

# ステータス値ごとの色 (VL53L8CX の代表的な target_status)
_STATUS_COLORS = {
    5: (0, 150, 60),     # 有効 (緑)
    6: (120, 190, 60),   # 信頼やや低 (黄緑)
    9: (230, 190, 0),    # 半信頼 (黄)
    10: (235, 140, 0),   # ターゲット不一致 (橙)
    255: (90, 90, 90),   # 無効 (グレー)
}
_STATUS_OTHER = (200, 60, 60)  # 上記以外 (赤)


def _dist_color(t):
    if t <= 0:
        return _DIST_ANCHORS[0][1]
    if t >= 1:
        return _DIST_ANCHORS[-1][1]
    for (t0, c0), (t1, c1) in zip(_DIST_ANCHORS, _DIST_ANCHORS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
    return _DIST_ANCHORS[-1][1]


def _status_color(st):
    if np.isnan(st):
        return (90, 90, 90)
    return _STATUS_COLORS.get(int(st), _STATUS_OTHER)


def resolve_csv_path(path):
    if not os.path.isfile(path):
        candidate = os.path.join(CSV_DIR, os.path.basename(path))
        if os.path.isfile(candidate):
            return candidate
        sys.exit(f"ファイルが見つかりません: {path} (csv/ も確認しました)")
    return path


def read_csv_columns(path):
    with open(path, newline="") as f:
        reader = csvmod.DictReader(f)
        cols = {name: [] for name in reader.fieldnames}
        n = 0
        for row in reader:
            for name in reader.fieldnames:
                try:
                    cols[name].append(float(row[name]))
                except (TypeError, ValueError):
                    cols[name].append(np.nan)
            n += 1
    return cols, n


def orient(grid):
    """4x4 を既定の向き(上下左右反転)に整える。"""
    return grid[::-1, ::-1]


def main():
    parser = argparse.ArgumentParser(
        description="指定センサーの16ゾーン距離+ステータスの時間変化GIFを出力")
    parser.add_argument("csv", help="入力CSV (例: log16.csv)")
    parser.add_argument("sensor", type=int, help="センサー番号 (例: 0, 1, 8)")
    parser.add_argument("--fps", type=float, default=20.0, help="GIFのフレームレート (既定: 20)")
    parser.add_argument("--step", type=int, default=1, help="何フレームおきに描くか (既定: 1)")
    parser.add_argument("--vmin", type=float, default=None, help="距離カラースケール最小[mm]")
    parser.add_argument("--vmax", type=float, default=None, help="距離カラースケール最大[mm]")
    parser.add_argument("-o", "--output", default=None,
                        help="出力GIFパス (省略時は gif/<csv名>_s<番号>.gif)")
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    path = resolve_csv_path(args.csv)
    cols, nrows = read_csv_columns(path)
    if nrows == 0:
        sys.exit("CSVにデータ行がありません。")

    s = args.sensor
    dist_names = [f"vl53l8cx_s{s}.d{i}" for i in range(16)]
    st_names = [f"vl53l8cx_s{s}.st{i}" for i in range(16)]
    if any(name not in cols for name in dist_names):
        avail = sorted({c.split(".")[0] for c in cols if c.startswith("vl53l8cx_s")})
        sys.exit(f"sensor {s} の距離列がCSVにありません。利用可能: {avail}")
    has_status = all(name in cols for name in st_names)
    if not has_status:
        print(f"警告: sensor {s} のステータス列が見つからないため距離のみ表示します。")

    frames = list(range(0, nrows, max(1, args.step)))

    # 距離カラースケール (全フレーム共通)
    vmin, vmax = args.vmin, args.vmax
    if vmin is None or vmax is None:
        allv = np.concatenate([np.array(cols[n], dtype=float) for n in dist_names])
        if vmin is None:
            vmin = float(np.nanmin(allv))
        if vmax is None:
            vmax = float(np.nanmax(allv))
    span = (vmax - vmin) if vmax > vmin else 1.0

    # レイアウト
    CELL = 70
    GRID = CELL * 4
    TITLE_H = 28
    PAD = 12
    n_panels = 2 if has_status else 1
    panel_w = GRID + PAD
    width = PAD + n_panels * panel_w
    height = TITLE_H + GRID + PAD

    def cell_text(draw, x0, y0, color, label):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        tc = (0, 0, 0) if lum > 140 else (255, 255, 255)
        tw = draw.textlength(label)
        draw.text((x0 + (CELL - tw) / 2, y0 + CELL / 2 - 6), label, fill=tc)

    images = []
    for fi in frames:
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 左: 距離
        dist = orient(np.array([cols[n][fi] for n in dist_names],
                               dtype=float).reshape(4, 4))
        x0 = PAD
        draw.text((x0, 6), f"sensor {s} distance[mm]  frame {fi}", fill=(0, 0, 0))
        for r in range(4):
            for c in range(4):
                v = dist[r, c]
                cx = x0 + c * CELL
                cy = TITLE_H + r * CELL
                if np.isnan(v):
                    color, label = (90, 90, 90), "-"
                else:
                    color = _dist_color((v - vmin) / span)
                    label = f"{v:.0f}"
                draw.rectangle([cx, cy, cx + CELL, cy + CELL],
                               fill=color, outline=(255, 255, 255))
                cell_text(draw, cx, cy, color, label)

        # 右: ステータス
        if has_status:
            st = orient(np.array([cols[n][fi] for n in st_names],
                                 dtype=float).reshape(4, 4))
            x0 = PAD + panel_w
            draw.text((x0, 6), f"sensor {s} status  frame {fi}", fill=(0, 0, 0))
            for r in range(4):
                for c in range(4):
                    v = st[r, c]
                    cx = x0 + c * CELL
                    cy = TITLE_H + r * CELL
                    color = _status_color(v)
                    label = "-" if np.isnan(v) else f"{int(v)}"
                    draw.rectangle([cx, cy, cx + CELL, cy + CELL],
                                   fill=color, outline=(255, 255, 255))
                    cell_text(draw, cx, cy, color, label)

        images.append(img)

    # 出力先
    if args.output:
        out = args.output
        if not os.path.dirname(out):
            os.makedirs(GIF_DIR, exist_ok=True)
            out = os.path.join(GIF_DIR, out)
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
    else:
        base = os.path.splitext(os.path.basename(path))[0]
        os.makedirs(GIF_DIR, exist_ok=True)
        out = os.path.join(GIF_DIR, f"{base}_s{s}.gif")
    if not out.lower().endswith(".gif"):
        out += ".gif"

    duration = int(round(1000.0 / args.fps))
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=duration, loop=0, optimize=False)
    print(f"GIFを保存しました: {out}")
    print(f"  sensor {s} / フレーム数 {len(images)} / {args.fps} fps "
          f"/ 距離スケール {vmin:.0f}-{vmax:.0f} mm "
          f"/ ステータス {'あり' if has_status else 'なし'}")


if __name__ == "__main__":
    main()
