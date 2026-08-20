# ToF由来の回転量をkalman_core(EKF)へ組み込む設計 — Python検証まとめ

`usd/tof_ekf_fusion*.py` で行った、ToF点群(法線ベクトル+Kabsch)由来のyaw情報を
CrazyflieのEKF (`kalman_core`) に統合する実験の設計・数式・結果をまとめる。

## 0. 前提: 実ファームウェアの現状

`~/workspace/tmp/crazyflie-firmware-vl53l8cx` (`feat#5-tof-pca-plane-fitting`
ブランチ) を確認したところ、以下の通りだった。

- `src/deck/drivers/src/tof_odometry.c` に、本リポジトリの `usd/tof_odometry.py`
  と同じ PCA+RANSAC平面抽出 → Kabsch(ΔR) → Point-to-Plane(Δt) のアルゴリズムが
  C移植済みで存在する。
- しかし **`kalman_core`(EKF) への統合は一切行われていない**。
  `tof_odometry.c` は `stabilizer.roll/pitch/yaw` (ジャイロ由来の姿勢) を読んで
  自身の整合性チェックに使うだけで、`kalmanCoreUpdateWith*`/`kalmanCoreScalarUpdate`
  等EKF側の関数は一度も呼ばれておらず、結果はログ/microSDに記録されるのみ。
- `kalman_core`側の測定モデル(`mm_*.c`)はスタックの
  `mm_tdoa.c mm_tdoa_robust.c mm_yaw_error.c mm_tof.c mm_absolute_height.c
  mm_flow.c mm_sweep_angles.c mm_position.c mm_distance.c mm_distance_robust.c
  mm_pose.c` のみで、ToFリング用の新規ファイルは追加されていない
  (`mm_tof.c`は単点レーザー測距(z-ranger)用で無関係)。

つまり本ドキュメントは「既存の統合をリバースエンジニアリングしたもの」ではなく、
**本リポジトリで新たに設計・Python検証した統合方式**であり、将来 `kalman_core`
にCで実装する際の設計資料として書いている。

---

## 1. kalman_core (EKF) の状態と予測モデル (前提知識)

`kalman_core.c` は姿勢誤差(error-state)型のEKF。状態ベクトル $S$ は9次元:

$$S = [X, Y, Z,\ P_X, P_Y, P_Z,\ D_0, D_1, D_2]$$

- $X,Y,Z$: ワールド座標系の位置
- $P_X,P_Y,P_Z$: ボディ座標系の速度
- $D_0,D_1,D_2$: 姿勢誤差(小角近似, ロール・ピッチ・ヨー方向)

姿勢そのものは別にクォータニオン $q=[w,x,y,z]$ で持ち、`predict()`のたびに
ジャイロで直接回転させる:

$$q_{k+1} = \Delta q(\boldsymbol{\omega}\, dt) \otimes q_k,\qquad
\Delta q(\mathbf{v}) = \left[\cos\frac{|\mathbf v|}{2},\ \sin\frac{|\mathbf v|}{2}\,\frac{\mathbf v}{|\mathbf v|}\right]$$

$D_0,D_1,D_2$ は毎ループ `finalize()` で $q$ に畳み込まれてから0に戻される
(詳細は `kalman_core.py` のdocstring、元実装は `kalman_core.c` の
`predictDt`/`kalmanCoreFinalize` を参照)。

### 汎用スカラー観測更新 (`kalmanCoreScalarUpdate`)

すべての測定モデル(`mm_*.c`)はこの関数を経由する。観測行列 $H$(1×9)、
イノベーション $e$、観測ノイズ標準偏差 $\sigma$ を渡すと、

$$
K = \frac{P H^\top}{H P H^\top + \sigma^2},\qquad
S \leftarrow S + K e,\qquad
P \leftarrow (I-KH)\,P\,(I-KH)^\top + \sigma^2 K K^\top
$$

(Joseph形式。共分散が非負定値に保たれる)。

### 既存の `mm_yaw_error.c`

外部から得たyaw基準を組み込むための、まさに今回使いたい形の測定モデルが
スタックに既にある:

```c
void kalmanCoreUpdateWithYawError(kalmanCoreData_t *this, yawErrorMeasurement_t *error) {
    float h[KC_STATE_DIM] = {0};
    h[KC_STATE_D2] = 1;
    kalmanCoreScalarUpdate(this, &H, this->S[KC_STATE_D2] - error->yawError, error->stdDev);
}
```

$H=[0,\dots,0,1]$ ($D_2$成分のみ1)、イノベーションは
$e = S[D_2] - \text{yawError} = -\text{yawError}$ ($D_2$は毎回0にリセットされるため)。
つまり「今の姿勢に対して、外部基準からyaw方向にどれだけズレているか」を
1本のスカラー観測として与える形になっている。本実装はこれをそのまま使う。

---

## 2. ToF側で作る「観測」: `tof_odometry.py` の出力

`run_odometry()` はフレームごとに回転行列 $dR_b$ (機体の1ステップ分の回転)を
計算する。詳細は `tof_odometry_usage.md` 参照。要点だけ書く。

- 法線ペアが2方向以上そろう ($\mathrm{rank}_R\ge2$) フレームでは、Kabsch法で
  **幾何学だけから**(ジャイロ非依存で) $dR_b$ を解く:
  $$dR = \arg\min_{R\in SO(3)} \sum_i w_i \|n'_i - R\,n_i\|^2$$
  ($H=\sum_i w_i n_i n_i'^\top$ のSVD $H=U\Sigma V^\top$ から $R=VDU^\top$)。
- 1方向しかそろわない ($\mathrm{rank}_R<2$) フレームでは `solve_yaw_only()` に
  フォールバックし、**ジャイロ積分由来のpitch/rollを既知として**yawだけを
  閉形式で解く。
- `rot_src` で由来を区別する: `1`=ToF(フルKabsch, rank≥2), `4`=ToF(yawのみ,
  rank<2), `2`=ジャイロ乖離で棄却, `3`=対応なし(純ジャイロ), `0`=なし。
- `dR_ypr[f]` = $dR_b$ をオイラー角YPRに分解した値、`dR_mat[f]` = $dR_b$ 行列
  そのもの (今回EKF統合のために追加した出力)。

---

## 3. 採用した統合方式: yaw誤差の測定更新 (`tof_ekf_fusion.py`)

### 3.1 設計方針

- `tof_odometry.py` は完全に独立に走らせ、その出力だけを消費する(疎結合)。
  ファームウェア側でも `tof_odometry.c` に手を入れず、EKF側に新しい
  `mm_tof_yaw.c` 相当を1つ追加するだけで移植できる想定。
- EKFに渡すのは **rank_R≥2 (`rot_src==1`) のフレームだけ**。rank<2
  (`rot_src==4`) は `solve_yaw_only` がジャイロのpitch/rollを既知として
  使っており、EKF自身の出力に依存した情報が混ざる(循環に近い)ため除外する。
- 渡す量は「累積yaw」ではなく **前回のToF成功フレームからの増分**
  ($dR_b$のyaw成分)。累積値をそのまま観測にすると、誤差が時間相関を持ち
  (一度ズレたらズレたまま)、EKFが「独立な確認を何度も得た」と過信する
  (Joseph形式のP更新は観測ノイズが真に独立であることを前提にしている)。

### 3.2 数式

ToFフレーム $f$ (rot_src==1) ごとに、直前のToF成功フレーム時点のEKF自身のyaw
$\hat\psi_{\text{prev}}$ (アンカー) と、`tof_odometry.py` が計算した増分
$\Delta\psi_f = \texttt{kabsch\_dR\_ypr}[f,0]$ から、

$$\psi_{\text{pred}} = \hat\psi_{\text{prev}} + \Delta\psi_f$$

を「ToF幾何学が示す現在のyaw」とし、EKFの現在のyaw推定 $\hat\psi_{\text{now}}$
との差を(±180度境界を正規化して)

$$\text{yawError} = \mathrm{wrap}(\psi_{\text{pred}} - \hat\psi_{\text{now}})$$

とする。これを `kalmanCoreUpdateWithYawError` と同じ形で $D_2$ に観測更新し、
更新後は $\hat\psi_{\text{prev}} \leftarrow$ 更新後のEKF yaw に置き換える
(`tof_odometry.py`自身の `R_kabsch_prev1 = R_cum` と同じ再アンカー)。

観測ノイズ $\sigma$ はヒューリスティック(平面フィットの残差から統計的に導出した
ものではない):

$$\sigma = \max\!\left(\frac{\sigma_0}{\sqrt{n_{\text{pairs}}}},\ \sigma_{\min}\right),\qquad
\sigma_0=3.0^\circ,\ \sigma_{\min}=1.0^\circ$$

(rank<2フレームも使う実験では追加でrank_deficient係数2倍)。**共分散は0では
なく、有効ペア数だけを見た簡易式**である(この点は前回の質疑で確認した通り)。

---

## 4. 試して却下した(または注意が必要な)統合方式

### 4.1 gyro.zをToF由来の角速度で置換する方式(オイラー分解経由) — 破綻

`tof_odometry.py`の $dR_{\text{ypr}}[f,0]$(オイラー角yaw成分)を区間の平均角速度
$\Delta\psi_f/\Delta t$ に変換し、生ジャイロの $g_z$ の代わりにEKFの`predict()`
へ入力(roll/pitch軸の $g_x,g_y$ は生データのまま)。

**結果**: `tof_odometry.py`自身の累積yawとの差が平均37°、最大103°まで拡大し破綻。
原因は、$dR_b$ のyawオイラー成分は本来pitch/roll成分と一体で決まった量であり、
それを取り出して**別々に測定されたジャイロのroll/pitchレートと合成し直す**と
SO(3)の合成として不整合になるため。オイラー分解自体よりも「pitch/roll成分を
捨てて別の情報に差し替えた」ことが根本原因。

### 4.2 dRb行列を直接クォータニオン合成する方式 — 動くが非推奨

オイラー分解を経由せず、$dR_b$ を直接クォータニオンに変換
(`quat_from_rotmat`, Shepperd法)し、`tof_odometry.py`自身の再アンカー則
($\texttt{since\_ref}$)と同じ規則でEKFの姿勢クォータニオンに直接合成:

$$q \leftarrow \mathrm{quat}(dR_b) \otimes q_{\text{at\_last\_fix}} \quad (\text{since\_ref==True のとき})$$

**結果**: `tof_odometry.py`自身の累積yawとの差は平均3.07°・最大5.75°まで改善し、
ほぼ再現できた。ただしこれは**EKFの状態を共分散を伴わずに直接上書きする**
ため、
- 共分散 $P$ が実際の不確かさと食い違ったまま残る
- 補正量の大きさに歯止めがなく、実飛行では姿勢制御ループへ不連続なジャンプ
  として伝わりうる

という理由で、実装検証・診断ツールとしては有用だが**実運用には推奨しない**
(3.のスカラー測定更新の方が、精度も上で滑らかだった。次表参照)。

---

## 5. 実験結果 (log50.csv, quadIsFlying=True, t=0〜50s の真値プロファイルとの比較)

| 方式 | 使用したToF観測数 | 平均絶対誤差 | 積分絶対誤差 |
|---|---|---|---|
| EKFなし: `tof_odometry.py`自身の累積yaw | - | 1.69° | 84.1 deg·s |
| EKFなし: 生ジャイロ単純積分 | - | 4.26° | 212.3 deg·s |
| EKF(gyroのみ、ToF未使用) | 0 | 1.44° | 72.2 deg·s |
| **EKF + yaw誤差の測定更新 (3節, rank≥2のみ)** | 6/296 | **1.27°** | **63.6 deg·s** |
| EKF + 測定更新 (①②の棄却を無効化, gate_deg=0) | 209/296 | 22.27° | 1112.7 deg·s |
| EKF + 測定更新 (rank<2も使用) | 116/296 | 4.97° | 248.6 deg·s |
| EKF + 測定更新 (KabschをIRLS+Huberでロバスト化) | 44/296 | 11.27° | 563.2 deg·s |
| EKF + gyro.z置換 (オイラー分解経由, 4.1節) | 116/296 | 42.69° | 2134.5 deg·s |
| EKF + dRb行列直接合成 (4.2節) | 116/296 | 2.93° | 146.3 deg·s |

観測数を増やす(棄却を緩める・rank制限を外す・ロバスト化する)ほど精度が
悪化する、という一貫した傾向が出ている。**「対応付け自体の誤り」は、統計的な
頑健化(IRLS)や単純な観測数増加では防げず、gate_degのような物理的妥当性
チェックが引き続き必要**、というのがこの一連の実験からの結論。

---

## 6. 今後の課題

- 観測ノイズ $\sigma$ が「有効ペア数」だけのヒューリスティックであり、
  Kabschの残差やRANSACインライア数、平面性スコアから統計的に導出した値では
  ない。より正確な共分散モデルへの改善余地がある。
- 並進(Δt)側はこの検証では真値照合をしていない。`solve_translation()`の
  truncated SVDで得られる観測可能な部分空間ごとに、`mm_flow.c`と同様の
  per-axisスカラー更新としてflow deckと共分散付きで融合すれば、両者の
  弱点(flowはテクスチャ/高度依存、ToFは平面の有無に依存)を補い合える
  可能性がある。ただし着手前にΔt単体の精度検証が必要。
- 実ファームウェアへの移植時は、`tof_odometry.c`の出力(rot_src相当の由来
  フラグ, $dR_b$のyaw増分, n_pairs)をログ変数として公開し、新しい
  `mm_tof_yaw.c`が非同期に読み取って`kalmanCoreScalarUpdate`を呼ぶ、という
  構成が、既存の`mm_flow.c`/`mm_yaw_error.c`と同じ非同期measurement-update
  アーキテクチャに最も自然に乗る。
