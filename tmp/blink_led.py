# blink_led.py
# Purpose: Temporarily control Crazyflie LEDs via the 'led.bitmask' parameter.
# Replace URI with your radio/USB URI (e.g. "radio://0/80/2M" or "usb://0")
import time
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
# Optional: configure logging or drivers as needed

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

def get_param_int(cf, group, name):
    """Return integer value of a param (may be string), safely converted to int."""
    val = cf.param.get_value(group, name)
    try:
        return int(val)
    except Exception:
        # param API can return bytes/str; fallback
        return int(''.join(ch for ch in str(val) if ch.isdigit()) or 0)

def set_param(cf, group, name, value):
    """Set parameter; pass value as string or int (API accepts string)."""
    cf.param.set_value(group, name, str(value))

def option_A_reboot_for_boot_blink(scf):
    """
    Option A: If you want the firmware's boot-time enumerateDecks blink to run,
    disable the bitmask (set to 0) and reboot the CF so enumerateDecks runs with param-mode off.
    Note: This requires the CF to actually reboot; use with care.
    """
    cf = scf.cf
    old = get_param_int(cf, 'led', 'bitmask')
    print("Old led.bitmask =", old)
    print("Setting led.bitmask = 0 (disable param override), then rebooting the CF...")
    set_param(cf, 'led', 'bitmask', 0)
    # Try to request a reboot via system (if supported)
    try:
        cf.system.reboot()
        print("Reboot command sent.")
    except Exception as e:
        print("Could not send reboot via API (you may need to power-cycle):", e)
    # NOTE: After reboot you must reconnect to see logs / blinks.

def option_B_temporary_host_blink(scf, led_mask, duration_s=5):
    """
    Option B (recommended to see LED now):
      - led_mask: bitmask with bit7 = ENABLE (0x80) plus LED bits (bit1=GREEN_L, bit0=BLUE_L, etc.)
      - duration_s: how long to hold the mask before restoring the previous value
    Example: To turn on GREEN_L only: led_mask = 0x80 | (1<<1) == 0x82
    """
    cf = scf.cf
    old = get_param_int(cf, 'led', 'bitmask')
    print("Old led.bitmask =", old)
    try:
        print(f"Setting led.bitmask = {hex(led_mask)} to show LED(s) for {duration_s}s")
        set_param(cf, 'led', 'bitmask', led_mask)
        time.sleep(duration_s)
    finally:
        print("Restoring previous led.bitmask =", old)
        set_param(cf, 'led', 'bitmask', old)

def main():
    # Example usage:
    # - To show GREEN_L for 5s: 0x80 | (1<<1) = 0x82
    # - To show BLUE_L for 5s:  0x80 | (1<<0) = 0x81
    led_green_left = 0x80 | (1 << 1)
    led_blue_left  = 0x80 | (1 << 0)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Connected.")
        # Option B: temporary blink using host-controlled bitmask
        option_B_temporary_host_blink(scf, led_green_left, duration_s=5)
        time.sleep(0.5)
        option_B_temporary_host_blink(scf, led_blue_left, duration_s=5)

        # Option A (uncomment if you want to try reboot approach):
        # option_A_reboot_for_boot_blink(scf)

if __name__ == '__main__':
    main()