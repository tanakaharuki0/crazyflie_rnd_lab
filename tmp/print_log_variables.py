import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

logging.basicConfig(level=logging.INFO)

def list_log_variables(scf):
    log_toc = scf.cf.log.toc.toc
    print("\n=== Available Log Variables ===\n")
    for group, variables in log_toc.items():
        for name, toc_item in variables.items():
            print(f"{group}.{name} ({toc_item.ctype})")

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Connected to Crazyflie!")
        time.sleep(1)  # 少し待って TOC 読み込みを確実に
        list_log_variables(scf)
