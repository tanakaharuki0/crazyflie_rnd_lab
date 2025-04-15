import logging
import sys
import time
from threading import Event
import yaml

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper
from lpslib.lopoanchor import LoPoAnchor

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT=0.5
yml_pass = '../locoposition_locate.yaml'

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

def move_linear_simple(scf):
    with PositionHlCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(1)
        mc.up(0.5)
        time.sleep(3)
        mc.down(0.5)
        # mc.forward(0.5)
        # time.sleep(1)
        # mc.turn_left(180)
        # time.sleep(1)
        # mc.forward(0.5)
        # time.sleep(1)

def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()
        
def connect_lps_anchor(scf):
    lopo = LoPoAnchor(crazyflie=scf)
    
    with open(yml_pass, 'r') as file:
        anchors = yaml.safe_load(file)
    
    for id, coords in anchors.items():
        anchor_id = int(id)
        position = [coords['x'],coords['y'],coords['z']]
        print(anchor_id,position)
        if anchor_id == 1:
            lopo.set_position(anchor_id, position)
            time.sleep(0.1)
            lopo.set_mode(anchor_id,LoPoAnchor.MODE_TWR)
            time.sleep(0.5)

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
    
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)
        
        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)
            
        connect_lps_anchor(scf)    
        
        time.sleep(1)
        
        # move_linear_simple(scf)
        # take_off_simple(scf)
