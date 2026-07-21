import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

WAIT_BEFORE_SPIN = 5.0   # [s] 接続後、プロペラを回転させるまでの待機時間
SPIN_TIME = 20.0         # [s] プロペラを回転させる時間
# プロペラの出力（0-65535）。離陸しない程度の低い値にすること。
# 機体・バッテリー残量によって浮き上がる閾値は変わるため、
# 初めて実行する際はより小さい値から試して安全な範囲を確認すること。
MOTOR_POWER = 10000

logging.basicConfig(level=logging.ERROR)


def spin_propellers(scf):
    cf = scf.cf

    print(f'{WAIT_BEFORE_SPIN:.0f}秒待機します...')
    time.sleep(WAIT_BEFORE_SPIN)

    try:
        print('スタビライザを経由しないダイレクトモータ制御を有効化します...')
        cf.param.set_value('motorPowerSet.enable', '1')
        time.sleep(0.1)

        print(f'{SPIN_TIME:.0f}秒間、離陸せずにプロペラを回転させます...')
        cf.param.set_value('motorPowerSet.m1', str(MOTOR_POWER))
        cf.param.set_value('motorPowerSet.m2', str(MOTOR_POWER))
        cf.param.set_value('motorPowerSet.m3', str(MOTOR_POWER))
        cf.param.set_value('motorPowerSet.m4', str(MOTOR_POWER))

        time.sleep(SPIN_TIME)
    finally:
        # 途中でエラーが発生した場合も含め、必ずモータを停止させる
        print('プロペラを停止します...')
        cf.param.set_value('motorPowerSet.m1', '0')
        cf.param.set_value('motorPowerSet.m2', '0')
        cf.param.set_value('motorPowerSet.m3', '0')
        cf.param.set_value('motorPowerSet.m4', '0')
        cf.param.set_value('motorPowerSet.enable', '0')


if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        spin_propellers(scf)
