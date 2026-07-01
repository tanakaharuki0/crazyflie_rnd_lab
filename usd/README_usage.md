# uSDログ → CSV → グラフ 実行手順

このフォルダ (`crazyflie_rnd_lab/usd`) のスクリプト実行メモ。

## 使う環境（重要）

使う仮想環境は **`myenv`** ひとつだけ。場所は:

```
/home/h-tanaka/workspace/crazyflie/myenv
```

`venv` という名前の環境は無い。迷ったら `myenv` を有効化すればよい。

### 有効化

```bash
source /home/h-tanaka/workspace/crazyflie/myenv/bin/activate
```

有効化すると先頭に `(myenv)` が付く。解除は `deactivate`。

### 必要パッケージの一括インストール

クローン直後は、このフォルダの `requirements.txt` でまとめて入れられる:

```bash
# myenv を有効化した状態で
pip install -r requirements.txt
```

`requirements.txt` には `numpy` / `pandas` / `matplotlib` / `Pillow` が含まれる
（`log_to_csv.py`, `plot_csv.py`, `depth_map.py`, `sensor_gif.py`,
`trajectory_gif.py` が必要とするもの）。

> Crazyflie 実機と通信するスクリプトを使う場合は `cflib` も追加で入れる:
> ```bash
> pip install cflib
> ```

---

## 1. log_to_csv.py （バイナリログ → CSV）

microSDデッキのバイナリログ（`log00` など）をCSVに変換する。

- 入力ログ: 引数で渡す。フルパスが無ければ `usd/logs/` 内を自動で探す。
- 出力CSV: `-o` 省略時は `usd/csv/<ログ名>.csv` に保存。

### 実行

```bash
# 環境を有効化してから
cd /home/h-tanaka/workspace/crazyflie/crazyflie_rnd_lab/usd

python3 log_to_csv.py log00              # -> csv/log00.csv
python3 log_to_csv.py log00 -o run1.csv  # -> csv/run1.csv
```

環境を有効化せず一発で動かす場合:

```bash
/home/h-tanaka/workspace/crazyflie/myenv/bin/python log_to_csv.py log00
```

主なオプション:
- `-o, --output PATH` : 出力先（ディレクトリ未指定なら `csv/` 下）
- `--event NAME` : 抽出するイベント名（既定 `fixedFrequency`）

---

## 2. plot_csv.py （CSV → グラフPNG）

`log_to_csv.py` が作ったCSVから距離をグラフ化する。
出力は必ず `usd/figs/` 下にPNG保存される。

### 実行

```bash
cd /home/h-tanaka/workspace/crazyflie/crazyflie_rnd_lab/usd

python3 plot_csv.py log00.csv              # -> figs/log00.png （1枚に重ね描き）
python3 plot_csv.py log00.csv --separate   # -> figs/log00_separate.png
python3 plot_csv.py log00.csv --save x.png # -> figs/x.png
```

主なオプション:
- `--save PATH` : 保存名（ディレクトリ未指定なら `figs/` 下）
- `--separate`  : センサーごとにサブプロットを分けて描画

スクリプト冒頭の `EXCLUDE_SENSORS` で除外センサー番号を指定できる
（例 `EXCLUDE_SENSORS = [0, 3, 10]`、空リストなら全描画）。

---

## まとめ（最短手順）

```bash
source /home/h-tanaka/workspace/crazyflie/myenv/bin/activate
cd /home/h-tanaka/workspace/crazyflie/crazyflie_rnd_lab/usd

python3 log_to_csv.py log00     # logs/log00 -> csv/log00.csv
python3 plot_csv.py  log00.csv  # csv/log00.csv -> figs/log00.png
```
