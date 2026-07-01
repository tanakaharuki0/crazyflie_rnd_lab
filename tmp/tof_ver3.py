import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

import csv

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT = 0.5

deck_attached_event = Event()
logging.basicConfig(level=logging.INFO)

# --------------------------
# ログ用変数とCSVファイル
log_variables = [
    "vl11.tick",
    "vl11.s0",
    "vl11.s1",
    "vl11.s2",
    "vl11.s3",
    "vl11.s4",
    "vl11.s5",
    "vl11.s6",
    "vl11.s7",
    "vl11.s8",
    "vl11.s9",
    "vl11.s10",
]
LOG_FILE = "crazyflie_log.csv"

# --------------------------
# データをCSVに保存する関数
def write_csv_header():
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + log_variables)

def log_data_callback(timestamp, data, logconf):
    row = [timestamp] + [data.get(var, '') for var in log_variables]
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def log_error_callback(logconf, msg):
    print("Log error:", msg)

# --------------------------
def param_deck_flow(name, value_str):
    # cflib passes (name, value_str) to param callbacks
    try:
        value = int(value_str)
    except Exception:
        value = 0
    if value:
        print('Deck is attached! (param {} = {})'.format(name, value_str))
    else:
        print('Deck is NOT attached! (param {} = {})'.format(name, value_str))
    deck_attached_event.set()

# --------------------------
if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        try:
            scf.cf.param.set_value('vl11.enable', '1')    # ドライバ/タスク ON
            scf.cf.param.set_value('vl11.testGen', '0')   # まず擬似データを OFF に
        except Exception as e:
            print("Param set error:", e)

        # Wait for blobs_ok to become 1 (firmware blobs in FLASH)
        blobs_ok = False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                val = scf.cf.param.get_value('vl11.blobs_ok')
                print("vl11.blobs_ok =", val)
                if val == '1':
                    blobs_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.2)

        if not blobs_ok:
            print("vl11.blobs_ok is not 1. Falling back to test generator (no real sensor or blobs missing).")
            try:
                scf.cf.param.set_value('vl11.testGen', '1')
            except Exception:
                pass

        # Optional: enumerate deck params for debugging
        try:
            for name in scf.cf.param.get_params():
                if name.startswith('deck.'):
                    print("deck param:", name)
        except Exception:
            pass

        # --------------------------
        # ログ設定
        log_conf = LogConfig(name='Logging', period_in_ms=100)
        for var in log_variables:
            log_conf.add_variable(var)

        scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(log_data_callback)
        log_conf.error_cb.add_callback(log_error_callback)
        write_csv_header()  # ヘッダー書き込み
        log_conf.start()

        # --------------------------
        # 離陸処理（例：アーミング）
        scf.cf.platform.send_arming_request(True)
        time.sleep(10.0)
        print("Success!")
        # --------------------------
        # ログ停止
        log_conf.stop()