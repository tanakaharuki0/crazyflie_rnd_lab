# depth_map.py 使い方

`log_to_csv.py` が出力したCSVから、VL53L8CX の **4×4 深度マップ**を描画するスクリプト。

各センサーは 16 ゾーンの距離 `d0..d15` [mm] と、対応するステータス `st0..st15` を持つ。
ゾーン番号 `i` は 4×4 グリッド上で `row = i // 4`, `col = i % 4` に対応する。

---

## 前提

- 入力は `log_to_csv.py` で変換済みのCSV（列 `vl53l8cx_s0.d0..d15`, `vl53l8cx_s0.st0..st15`, sensor1 も同様）。
- 必要パッケージ:
  - 静的マップ表示/保存 (`-o`): `numpy`, `pandas`, `matplotlib`
  - GIF生成 (`--gif`): `numpy`, `Pillow` のみ（pandas/matplotlib 不要・標準csvで読み込み）

---

## 基本的な使い方

```bash
# sensor0,1 の「全フレーム平均」の深度マップを表示
python3 depth_map.py log16.csv

# 特定フレーム(行番号)だけを表示
python3 depth_map.py log16.csv --frame 0

# sensor0 のみ、無効ゾーン(status!=5)を除外して画像保存
python3 depth_map.py log16.csv --sensor 0 --mask-invalid -o depth.png

# 時間変化をGIFアニメーションで保存
python3 depth_map.py log16.csv --gif log16.gif
```

入力CSVは直接パスのほか、ファイル名だけ指定すると `csv/` フォルダ内も自動で探す。

---

## オプション一覧

| オプション | 説明 | 既定 |
|---|---|---|
| `csv` (必須) | 入力CSVファイル（例: `log16.csv`） | — |
| `--sensor N` | 表示するセンサー番号 | 存在する全センサー(0,1) |
| `--frame N` | 表示する行(サンプル)番号 | 省略時は全フレーム平均 |
| `--mask-invalid` | `target_status != 5` のゾーンを除外(NaN) | 無効 |
| `--flip-x` | 左右反転（取り付け向き調整用） | 無効 |
| `--flip-y` | 上下反転（取り付け向き調整用） | 無効 |
| `--vmin FLOAT` | カラースケール最小 [mm] | 自動 |
| `--vmax FLOAT` | カラースケール最大 [mm] | 自動 |
| `-o, --output PATH` | 画像保存パス。省略時は画面表示のみ | 表示のみ |
| `--gif PATH` | 時間変化をGIFアニメーション化して `gif/` に保存 | 無効 |
| `--fps FLOAT` | GIFのフレームレート | 10 |
| `--step N` | GIFで何フレームおきに描くか（間引き） | 1 |

- `-o` でファイル名のみ指定した場合は `figs/`、`--gif` でファイル名のみ指定した場合は `gif/` フォルダに保存される。
- 全フレーム平均時に `--mask-invalid` を併用すると、無効ゾーンを NaN にしてから `nanmean` で平均する。
- `--gif` 指定時は `--frame` は無視され、全フレームを時間順にアニメーション化する。
  カラースケールは全フレーム共通（`--vmin/--vmax` 未指定ならデータ全体の最小/最大から自動決定）。

---

## 表示内容

- ゾーン距離を 4×4 グリッドで色表示（近いほど濃い `viridis_r`）。
- 各セルに距離値 [mm] を数値で重ね描き（無効ゾーンは `-`）。
- センサーごとにサブプロットを横並びで表示し、カラーバー（distance [mm]）を付与。

---

## 使用例

```bash
# 全フレーム平均・無効ゾーンを除外して2センサー比較を保存
python3 depth_map.py log16.csv --mask-invalid -o log16_mean.png

# sensor1 の 10 フレーム目を 0〜2000mm のスケール固定で表示
python3 depth_map.py log16.csv --sensor 1 --frame 10 --vmin 0 --vmax 2000

# 取り付け向きに合わせて左右反転
python3 depth_map.py log16.csv --frame 0 --flip-x

# 5フレームおきに間引いて 15fps のGIFを作成、無効ゾーンは除外
python3 depth_map.py log16.csv --gif log16.gif --step 5 --fps 15 --mask-invalid
```

---

## 注意点

- **ゾーンの向き**: VL53L8CX は取り付け向きやミラー設定により X/Y が反転することがある。
  実物と見比べ、必要に応じて `--flip-x` / `--flip-y` で合わせる。
- **無効値**: `status=255`（範囲外）や `9`（半信頼）のゾーンが混ざる場合がある。
  実距離マップとして見るときは `--mask-invalid` を推奨。
- **記録対象**: ファームウェア側で sensor0,1 の全16ゾーン＋ステータスがログされている必要がある
  （SDログ変数上限・`config.txt` の設定に依存）。
