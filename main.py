import vgamepad as vg
from pynput.keyboard import Key, Controller
import time
import socket
import csv
import math

import numpy as np

from PID_controller import PIDController
from ac_comunication import Track, AC_Connection

HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on

gamepad = vg.VX360Gamepad()
keyboard = Controller()

control_change_time = 0.1
control_change_distance = 0.5
max_deviation_dist = 15


# TODO: move this to AC_Connection class
# def check_restart():
#     keyboard.press(Key.ctrl)
#     keyboard.press('o')
#     time.sleep(0.1)
#     keyboard.release('o')
#     keyboard.release(Key.ctrl)
#     time.sleep(0.1)
#     keyboard.press(Key.ctrl)
#     keyboard.press('k')
#     time.sleep(0.1)
#     keyboard.release('k')
#     keyboard.release(Key.ctrl)
#     time.sleep(0.1)

#     conn.setblocking(False)
#     try:
#         while True:
#             _ = conn.recv(1024)
#     except BlockingIOError:
#         pass  # Buffer is now empty
#     conn.setblocking(True)


steering_controller = PIDController(Kp=15.0, Ki=0.1, Kd=1.5, setpoint=0.0)

target_velocity = 30.0
velocity_controller = PIDController(Kp=1.0, Ki=0.1, Kd=0.0, setpoint=target_velocity)

track = Track('saved_tracks/vallelunga_club.csv', control_change_distance)

ac_connection = AC_Connection(HOST, PORT, gamepad, track, velocity_controller, steering_controller)
ac_connection.connect()

while True:
    ac_connection.control_step()