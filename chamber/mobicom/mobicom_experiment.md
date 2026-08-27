# 注意点
 - PCによってファイル構造が異なります．
 - h-tanaka@htanaka:
   - ~/workspace/tmp/crazyflie-firmware-vl53l8cx$
   - ~/workspace/watalab_wsl/crazy_fly/crazyflie_rnd_lab$
 - h-tanaka@h-tanaka-ThinkPad-E14-Gen-6:
   - ~/test/crazyflie_tmp/tmp/crazyflie-firmware-vl53l8cx$
   - ~/workspace/crazyflie/crazyflie_rnd_lab$


# 実験と評価における決めごと
 - 50m * 50cm * 50cmの立方体において、上方のみ面を取り外して開放
 - 離陸して2秒で高さ25cmに行き、10秒間高さ25cmにおいて中央による制御を10秒間実施（ホバリング）、着陸は1秒で行う。
 - 衝突回数, 着陸時の中心からのズレ(着陸時に壁に衝突した場合は25cmとする), 着陸時の角度（または墜落したときにはcrashと書いている。aは着陸時に壁に衝突したことを示すこととする。）

 - 4つの比較を行う（ex3,4は今回の論文のメインで特に4が優れていることが示せれば理想）
    - ex1: 離陸から着陸まですべてflow deckでposition holdする（/home/h-tanaka/test/crazyflie_tmp/tmp/crazyflie-firmware-vl53l8cx/examples/demos/app_cf_brushless_chamber_without_Pcontrol）
    - ex2: 離陸と着陸はflow deckで水平方向のposition holdをし、ホバリング時はP制御を行う（/home/h-tanaka/test/crazyflie_tmp/tmp/crazyflie-firmware-vl53l8cx/examples/demos/app_cf_brushless_chamber）
    - ex3: 離陸から着陸まですべてP制御で中央によせる制御を行う（/home/h-tanaka/test/crazyflie_tmp/tmp/crazyflie-firmware-vl53l8cx/examples/demos/app_cf_brushless_chamber_takeoff_Pcontrol）
    - ex4: 離陸と着陸はP制御を行い、ホバリング時はhybrid（平面を捉えてyaw誤差補正を行いながらその平面からの距離でP制御を行い、平面を捉えることができなかった場合はex2や3と同様のP制御）で制御を行う（/home/h-tanaka/test/crazyflie_tmp/tmp/crazyflie-firmware-vl53l8cx/examples/demos/app_cf_brushless_chamber_takeoff_Pcontrol_hybrid）

# ex1_flow_flow_flow (log360~374)
 - 回，cm, crash
 - logナシ、回，cm, crash
 - 1回，25cm, a
 - 3回，25cm, a
 - 4回，25cm, a
 - 回，cm, crash
 - 回，cm, crash
 - 回，cm, crash
 - logナシ、3回，25cm, 
 - 3回，25cm, 
 - 回，cm, crash
 - 2回，10cm, 
 - 0回，25cm, 
 - 回，cm, crash
 - crash
 - crash
 - 2回，25cm
 - crash
 - 3回，15cm
 - 1回，15cm
 - 4回，25cm

# ex2_flow_P_flow (log340~353)
 - 回，cm, crash
 - 3回，25cm, a
 - 3回，25cm, a
 - 0回，20cm, 
 - 回，cm, crash
 - 回，cm, crash
 - 1回，25cm, a
 - 1回，20cm, 
 - 1回，20cm, 
 - 0回，20cm, 45
 - 1回，20cm, 
 - 3回，25cm, a
 - 3回，25cm, a
 - 回，cm, crash

# ex3_P_P_P (log320~333)
 - 回，cm, crash
 - 2回，10cm, 15
 - 0回，0cm, -15
 - 1回，15cm, 20
 - 0回，5cm, 45
 - 2回，25cm, 110
 - 0回，5cm, -10
 - 2回，15cm, 25
 - 1回，10cm, 0
 - 回，cm, crash
 - 0回，5cm, 25
 - 回，cm, crash
 - 2回，10cm, 35
 - 1回，15cm, -15

# ex4_P_hybridP_P (log300~312)
 - 0回，0cm, 90
 - 0回，10cm, 20
 - 0回，5cm, 90
 - 0回，10cm, -5
 - crash
 - 0回，15cm, 20
 - 1回，5cm, 45
 - 1回，5cm, 15
 - 2回，10cm, -40
 - 2回，20cm, -10
 - 0回，10cm, 5
 - 1回，10cm, 0
 - crash

# 考察方法について
 - 着陸位置や衝突回数など目視で確認できる部分は上に記した
 - その他の飛行中の考察についてはログを解析することによって行う必要がある。
 - logの解析では一度csvに変換する必要があるので、それようの既存のスクリプトを用いる、または作成して、/mobicom/logs_ex○_○○の下にあるlogはすべてcsvに変換してほしい
 - 既存のコードや新しく作成したpythonコードでlogの解析結果で出力された図はすべて論文に乗せる可能性があるので、labelなどは英語にしてほしい。フォントはtimes new romanです。それらの図はmobicom/figsに保存してほしい。
 - logの結果やその他を含めた総合的な考察を、論文に載せられるくらいの粒度で、/mobicom/kousatu.mdに保存してほしい。

 # 評価の修正
  - 使用するデータに関して、crashしなかったデータ10個を用いてほしい。
  - ex3,4に関してはcrashしていないデータが11個あるので、最後の1個は用いないでほしい。
  - crashしなかったデータ10個を用いて、mobicom/figs/の下にある画像とpdfを再生成してほしい。
  - mobicomのdemo paperに載せる図で、fig1の(c)とfig2の(a)を用いるので、この2枚を横並びにした図もpdfで別で作成してほしい。

 # mobicom修正点
 ## 削除してほしい内容
  - Maxspeedは全て0.15m/sだが，そもそも記載する必要はない
  - k_centerも記載する必要なし
  - system designにおいて，モーター間110 mmという記述は不要
  - tof sensorのデータが拡張カルマンフィルターに融合されているというのは嘘なので，削除してほしい（別のzranger sensorは融合されているが，わざわざそれを記載する必要はない）．
  - 
  - 「チャンバーの脇に置いたノートPCが、機体がオンボードで実際に使っているのと同じ壁距離・平面フィット状態・ヨー誤差の信号をライブダッシュボードとしてストリーミングする」というのはできないので，削除してほしい．
  - ex3とex4において，それぞれの評価指標において同等であるという事実は述べなくて良い（RMSセンタリング誤差が同じくらい，墜落率がそれほど変わらない）．
 
 ## 記載してほしい内容
  - ToF sensorからは4*4のメッシュ状に取得できることを記載する
  - hybridを推しだしたいので、ex4についてはどういう方法なのか説明をしてほしい．
    - ホバリング時はそれぞれのtof sensorで得られた16個のpoint croudからRANSAC＋PCAで平面フィットすること．
    - RANSACとPCAの役割
    - 平面性を閾値で判定し，前後のペアとと左右のペアそれぞれで両方の平面が得られたときは，平面からの距離を用いてP制御をし，それ以外のときは他の手法と同様にToFセンサーの16個のデータの平均値を壁からの距離として，P制御を行う．
    - またyawの制御は，壁に対して垂直方向を向くように制御されている
    - system designにおいて，ToF sensorは前後上下左右に加え，前方は広範囲をとらえられるように斜め方向に4つのセンサーを付けていることを書く
    - 壁への平均衝突回数は表で示し，ex3とex4が安定しているということを述べてほしい．
  
 ## 内容を変更すべき点
   - 要旨において，「壁基準の制御」を「壁からの距離を用いたP制御」に変更
   - 要旨において，精度が向上したという話は量を減らしてほしい．代わりに，複数のマルチゾーンToFセンサを用いた制御が狭隘空間において有効であることが分かったことや，マルチゾーンToFセンサが平面をとらえることができ，制御方法に拡張性があることが分かったことを書いてほしい．
   - まとめにおいて「壁基準P制御に置き換えるだけで、着陸時の壁接触は完全になくなる。」の完全にというのは強調しすぎなので，飛行における壁への接触を減らせるという内容にしてほしい．
   - table1においてFig3（a）と内容が被っているので，table1を削除してください．table1を引用している文章があればそれも修正してください．
   - conclusionでは数値がどれだけ改善したかなどの情報は不要です(demo paperによってはそもそもconclusionを設けていないものもあるくらいです)．conclusionが長すぎるので削除，場合によっては無くしてしまってください．
   - figure1に関して，(a)(b)を合わした図crazyflie_overallを作成したのでそれを用いてください．そして，figure1と2は同じ段に横並びにしてください．
   - 使用する図を変更したのでそれに応じて内容を変更してほしい
   