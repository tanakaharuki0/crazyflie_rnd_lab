# chamber実験の設定 (app_cf_brushless_chamber)

50cm×50cm×50cmのチャンバー内で、機体を中央に留めたまま飛行させるための
appレイヤーの設定をまとめる。ベースは
`crazyflie-firmware-vl53l8cx/examples/demos/app_cf_brushless_slam_tunnel`
で、これを以下のように拡張・変更したものが本実験で使用したapp
(`app_cf_brushless_chamber`)。

## 1. ベースアプリからの変更点

`app_cf_brushless_slam_tunnel` はトンネル追従(片軸方向のP制御)のみを行う
アプリだったが、本実験用には以下のように拡張した。

- **P制御を前後・左右の両軸に拡張**
  トンネル追従は片方向のみのP制御だったが、chamber実験では前後(front/back)
  ・左右(left/right)の両方の壁距離差に対してP制御をかけ、チャンバー中央への
  復元力を働かせるようにした。
- **高さは25cm固定**
  チャンバーは50cm角なので、床から25cmの高さでHOVERするよう高さ目標を固定。
- **yawは離陸時の値に固定**
  離陸した瞬間のyaw角を目標値として保持し、以降yawはその値からズレないよう
  制御する(chamber内での回転は想定しない)。

## 2. 水平方向センタリング制御則

前後・左右それぞれの壁距離差に比例したP制御で、目標速度を生成する。

```
v_cmd = clip(kCenter * (a - b), -centerMaxV, +centerMaxV)   [m/s]
```

- `a - b` は対向センサ間の距離差 (前後: front - back、左右: left - right)。
  差が0 = チャンバー中央。
- `kCenter`: Pゲイン。本実験ではこの値を **0.3 / 0.5 / 0.7** と変化させて
  比較した (`kousatsu_wall_centering_k.md` 参照)。
- `centerMaxV = 0.05 m/s`: 水平方向の指令速度の上限。kCenterの値によらず
  一律でこの値にクランプする。

壁距離はVL53L8CXの中央4ゾーン平均を使用し、割り当ては
front=s1, back=s7, left=s5, right=s6 (up=s8は天井/ハンドトリガー用)。

## 3. P制御なし条件 (without_Pcontrol)

比較用のベースライン条件として、壁センサに基づくセンタリング制御を行わず、
HOVER中は単純に離陸位置を原点とした固定目標点 `(x, y, z) = (0, 0, 0.25) [m]`
(チャンバー中央、高さ25cm)に留まるよう位置制御するのみとした。

## 4. フライトシーケンス (FSM)

`IDLE -> WAIT_DECK -> PREARM -> ARMING -> ARM_DWELL -> TAKEOFF -> HOVER -> LAND`
の状態遷移で飛行する。センタリング制御(またはP制御なしの固定点保持)が
働くのはHOVER状態の間のみ。

## 5. ログ・解析

- ログ変換: `chamber/log_to_csv_chamber.py` (kCenterの値ごとに
  `chamber/logs_k_0.3` / `logs_k_0.5` / `logs_k_0.7` / `logs_without_Pcontrol`
  フォルダに分けて記録し、`chamber/csv/` にCSV化)
- 診断プロット: `chamber/plot_chamber.py` (壁距離・中心オフセット・
  再現コマンド速度・位置推定・姿勢・XY軌跡を1枚にまとめて出力、
  `chamber/figs/` に保存)。ファイル名の `_k_<値>` からkCenterを自動判定して
  再現コマンド・飽和しきい値を計算する。
- kCenter比較の考察: `chamber/kousatsu_wall_centering_k.md`
