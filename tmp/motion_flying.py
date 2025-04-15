import logging
import sys
import time
from threading import Event
import yaml
import csv

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper
from lpslib.lopoanchor import LoPoAnchor
from functools import partial
# from cflib.localization import 

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT=0.5
yml_pass = '../locoposition_locate.yaml'
csv_file = './loc.csv'

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

# 位置データを受け取るコールバック関数
def log_pos_callback(csv_writer, timestamp, data, logconf):
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    csv_writer.writerow([timestamp,x,y,z])
    print(f"[{timestamp}] Position -> X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")

# ログ設定を行う関数
def start_position_logging(cf,log_file):    
    log_config = LogConfig(name='KalmanPos', period_in_ms=100)  # 10Hz
    log_config.add_variable('kalman.stateX', 'float')
    log_config.add_variable('kalman.stateY', 'float')
    log_config.add_variable('kalman.stateZ', 'float')
    
    csv_writer = csv.writer(log_file)
    # 追加するコールバック関数に引数を渡すためにpartial使用
    cb = partial(log_pos_callback,csv_writer)

    cf.log.add_config(log_config)
    log_config.data_received_cb.add_callback(cb)
    log_config.start()
    print("位置のログ取得を開始しました")

def move_linear_simple(scf):
    with PositionHlCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(2)
        mc.up(0.5)
        time.sleep(1)
        mc.forward(0.5)
        time.sleep(1)
        mc.turn_left(180)
        # time.sleep(1)
        # mc.forward(0.5)
        # time.sleep(1)
        mc.down(0.5)

def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(2)
        mc.stop()
        
def connect_lps_anchor(cf):

    lopo = LoPoAnchor(crazyflie=cf)
    
    with open(yml_pass, 'r') as file:
        anchors = yaml.safe_load(file)
    
    for id, coords in anchors.items():
        anchor_id = int(id)
        position = [coords['x'],coords['y'],coords['z']]
        print(anchor_id,position)
        lopo.set_position(anchor_id, position)
        time.sleep(0.1)
        lopo.set_mode(anchor_id,LoPoAnchor.MODE_TWR)
        time.sleep(0.3)

def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    value = 1
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    # print(cflib.crtp.scan_interfaces())
    # print(cflib.crtp.get_interfaces_status())
    # print(cflib.crtp.get_link_driver(URI))
    with open(csv_file, mode='w', newline = '') as log_file:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

            scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                             cb=param_deck_flow)
            time.sleep(1)

            if not deck_attached_event.wait(timeout=5):
                print('No flow deck detected!')
                sys.exit(1)

            # connect_lps_anchor(scf.cf)    
            start_position_logging(scf.cf, log_file)
            time.sleep(10)

            # move_linear_simple(scf)
            # take_off_simple(scf)
