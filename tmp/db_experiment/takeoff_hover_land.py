import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

WAIT_BEFORE_TAKEOFF = 5.0   # [s] 接続後、離陸するまでの待機時間
TARGET_HEIGHT = 1.0         # [m] 目標高度
TAKEOFF_TIME = 1.0          # [s] 離陸にかける時間
HOVER_TIME = 15.0           # [s] ホバリングを続ける時間
TAKEOFF_VELOCITY = TARGET_HEIGHT / TAKEOFF_TIME  # 1.0 m/sで上昇し、ちょうど1秒で1mに到達させる

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)


def param_deck_flow(_, value_str):
    value = int(value_str)
    if value:
        deck_attached_event.set()
        print('Flow deck is attached!')
    else:
        print('Flow deck is NOT attached!')


def fly(scf):
    mc = MotionCommander(scf, default_height=TARGET_HEIGHT)
    try:
        print(f'{WAIT_BEFORE_TAKEOFF:.0f}秒待機します...')
        time.sleep(WAIT_BEFORE_TAKEOFF)

        print(f'{TAKEOFF_TIME:.0f}秒かけて高度{TARGET_HEIGHT:.1f}mまで上昇します...')
        mc.take_off(height=TARGET_HEIGHT, velocity=TAKEOFF_VELOCITY)

        print(f'{HOVER_TIME:.0f}秒間ホバリングします...')
        time.sleep(HOVER_TIME)
    finally:
        # 途中でエラーが発生した場合も含め、必ず着陸させる
        print('着陸します...')
        mc.land()


if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                          cb=param_deck_flow)
        time.sleep(1)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        fly(scf)
