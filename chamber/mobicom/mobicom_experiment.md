
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