import vgamepad as vg
from pynput.keyboard import Key, Controller
import time

import numpy as np

from PID_controller import PIDController
from ac_comunication import Track, AC_Connection
from PPO_class import PPO

HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on

control_change_time = 0.1
control_change_distance = 0.5
max_deviation_dist = 15

steering_controller = PIDController(Kp=0.15, Ki=0.001, Kd=0.015, setpoint=0.0)

target_velocity = 30.0
velocity_controller = PIDController(Kp=0.010, Ki=0.001, Kd=0.0, setpoint=target_velocity)

track = Track('saved_tracks/vallelunga_club.csv', control_change_distance)

ac_connection = AC_Connection(HOST, PORT, track, velocity_controller, steering_controller)
ac_connection.connect()

agent = PPO(ac_connection, input_size=3, output_size=2)

while True:
    agent.learn(total_steps=10000)