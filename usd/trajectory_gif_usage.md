# trajectory_gif.py 使い方

Flow deck の生データ (`motion.deltaX` / `motion.deltaY`) と高さ (`range.zrange`) から
`x, y, z` の軌跡を再構成し、時間とともに軌跡が伸びていく **3D アニメーションGIF** を出力する。

---

## 前提

- 入力CSVに以下の列が必要:
  - `range.zrange` … 高さ
  - `motion.deltaX`, `motion.deltaY` … Flow deck の生オプティカルフロー量
- 必要パッケージ: `numpy`, `matplotlib`（GIF書き出しに `Pillow`）。

---

## 基本的な使い方

```bash
# 既定 (4サンプルおき・20fps)。gif/<csv名>_traj.gif に保存
python3 trajectory_gif.py log23.csv

# 間引きとフレームレートを指定して出力名も指定
python3 trajectory_gif.py log23.csv --step 4 --fps 20 -o traj.gif

# 直近100サンプルだけ尾を引いて表示
python3 trajectory_gif.py log23.csv --tail 100
```

入力CSVはファイル名だけ指定すると `csv/` フォルダ内も自動で探す。
出力は `gif/` フォルダに保存される（ファイル名のみ指定時）。

---

## オプション一覧

| オプション | 説明 | 既定 |
|---|---|---|
| `csv` (必須) | 入力CSV（例: `log23.csv`） | — |
| `--fps FLOAT` | GIFのフレームレート | 20 |
| `--step N` | 何サンプルおきに1フレーム描くか（間引き） | 4 |
| `--zunit {auto,mm,m}` | `range.zrange` の単位。`auto` は最大値>10でmm判定 | auto |
| `--tail N` | 直近Nサンプルのみ描画（0で全履歴） | 0 |
| `-o, --output PATH` | 出力GIFパス | `gif/<csv名>_traj.gif` |

---

## 位置の再構成方法

Crazyflie の Kalman フローモデル（firmware `mm_flow.c`）と同じ定数を使用:

```
measured_pixels = deltaX * FLOW_RESOLUTION            (FLOW_RESOLUTION = 0.10)
measured_pixels = (dt * Npix / thetapix) * (v / z)    (Npix = 35, thetapix = 0.71674)
```

これを解くと、1サンプルあたりの水平移動量は `dt` に依存せず次式になる:

```
dx = deltaX * FLOW_RESOLUTION * thetapix / Npix * z
dy = deltaY * FLOW_RESOLUTION * thetapix / Npix * z
```

これを累積して `x, y` を得る。`z` は `range.zrange`（mmならm換算）をそのまま高さとして用いる。

---

## 注意点

- **簡易再構成**: 機体ヨー回転やジャイロ補償は無視している。水平方向には絶対基準が
  無いため、ドリフトは残る（`stateEstimate.x/y` が発散するのと同じ原因）。
- **フロー生データがゼロの場合**: `motion.deltaX/deltaY` が全ゼロだと `x, y` は原点から
  動かない（`z` だけ変化）。Flow deck が未装着・未認識、あるいは `motion` 群が
  更新されていない可能性がある。スクリプトはこの場合に警告を表示する。
  - 例: `log23.csv` は `motion.deltaX/deltaY` が全ゼロのため、x,y は原点のままになる。
- **z の単位**: `range.zrange` が mm の場合（最大値が数百）と m の場合の両方に対応。
  自動判定が誤るときは `--zunit mm` / `--zunit m` で明示する。
