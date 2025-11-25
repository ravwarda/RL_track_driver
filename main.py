import vgamepad as vg
from pynput.keyboard import Key, Controller
import time

import numpy as np

import os
import logging

from PID_controller import PIDController
from ac_comunication import Track, AC_Connection
from PPO_class import PPO

def main():
	# Centralized logging configuration for the application
	root_logger = logging.getLogger()
	if not root_logger.handlers:
		level = getattr(logging, os.getenv("PPO_LOG_LEVEL", "INFO").upper(), logging.INFO)
		root_logger.setLevel(level)
		fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

		sh = logging.StreamHandler()
		sh.setFormatter(fmt)
		sh.setLevel(level)
		root_logger.addHandler(sh)

		log_path = os.path.join(os.getcwd(), "ppo.log")
		fh = logging.FileHandler(log_path)
		fh.setFormatter(fmt)
		fh.setLevel(level)
		root_logger.addHandler(fh)

	HOST = '127.0.0.1'  # Localhost
	PORT = 65432        # Port to listen on

	control_change_distance = 0.5

	steering_controller = PIDController(Kp=0.2, Ki=0.001, Kd=0.015, setpoint=0.0)

	target_velocity = 10.0
	velocity_controller = PIDController(Kp=0.04, Ki=0.001, Kd=0.0, setpoint=target_velocity)

	track = Track('saved_tracks/vallelunga_club.csv', control_change_distance)

	ac_connection = AC_Connection(HOST, PORT, track, velocity_controller, steering_controller, residual_scale=0.2)
	ac_connection.connect()

	agent = PPO(ac_connection, input_size=12, output_size=2, load_weights=True)

	try:
		agent.learn()
	finally:
		agent.save('ppo_model.pth')

if __name__ == "__main__":
	main()