import vgamepad as vg
from pynput.keyboard import Key, Controller
import time
import socket
import csv
import math

import numpy as np

from PID_controller import PIDController

HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on

gamepad = vg.VX360Gamepad()
keyboard = Controller()

# Configuration constants
max_steering = 105

control_change_time = 0.1
control_change_distance = 0.5
max_deviation_dist = 15

def find_nearest_point(track_x, track_y, point_x, point_y):
    min_dist = float('inf')
    nearest_idx = -1
    for i, (tx, ty) in enumerate(zip(track_x, track_y)):
        dist = math.hypot(tx - point_x, ty - point_y)
        
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    # determine signed normal deviation from the track at nearest_idx
    tx = track_x[nearest_idx]
    ty = track_y[nearest_idx]

    # compute tangent direction using neighbors (handle endpoints)
    if nearest_idx == 0:
        t_x = track_x[1] - tx
        t_y = track_y[1] - ty
    elif nearest_idx == len(track_x) - 1:
        t_x = tx - track_x[-2]
        t_y = ty - track_y[-2]
    else:
        # use central difference for a smoother tangent
        t_x = track_x[nearest_idx + 1] - track_x[nearest_idx - 1]
        t_y = track_y[nearest_idx + 1] - track_y[nearest_idx - 1]

    mag = math.hypot(t_x, t_y)
    if mag == 0:
        t_x, t_y = 1.0, 0.0
    else:
        t_x /= mag
        t_y /= mag

    # left-hand normal (unit)
    n_x = -t_y
    n_y = t_x

    # vector from track point to queried point
    v_x = point_x - tx
    v_y = point_y - ty

    # signed distance: positive if point is to the left of the track direction
    signed_dist = v_x * n_x + v_y * n_y
    return nearest_idx, signed_dist


def check_restart():
    keyboard.press(Key.ctrl)
    keyboard.press('o')
    time.sleep(0.1)
    keyboard.release('o')
    keyboard.release(Key.ctrl)
    time.sleep(0.1)
    keyboard.press(Key.ctrl)
    keyboard.press('k')
    time.sleep(0.1)
    keyboard.release('k')
    keyboard.release(Key.ctrl)
    time.sleep(0.1)

    conn.setblocking(False)
    try:
        while True:
            _ = conn.recv(1024)
    except BlockingIOError:
        pass  # Buffer is now empty
    conn.setblocking(True)


def interpolate_track_points(original_points, control_change_distance):
    track_x = []
    track_y = []

    if original_points:
        prev_x, prev_y = original_points[0]
        track_x.append(prev_x)
        track_y.append(prev_y)

        cum_seg_dist = 0.0

        for i in range(1, len(original_points)):
            curr_x, curr_y = original_points[i]
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            segment_dist = math.hypot(dx, dy)
            cum_seg_dist += segment_dist
            if segment_dist == 0:
                continue
            elif cum_seg_dist >= control_change_distance:
                track_x.append(curr_x)
                track_y.append(curr_y)
                cum_seg_dist = 0.0

    return track_x, track_y

with open('position_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    track_x = []
    track_y = []

    # Read original points from CSV
    original_points = []
    for row in reader:
        original_points.append((
            float(row[0]),  # x
            float(row[1]),  # y
        ))

    # Interpolate points so that distance between them is control_change_dist
    track_x, track_y = interpolate_track_points(original_points, control_change_distance)

last_time = time.time()
last_x_position = 0.0
last_y_position = 0.0

steering_controller = PIDController(Kp=10.0, Ki=0.1, Kd=1.0, setpoint=0.0)

target_velocity = 30.0
velocity_controller = PIDController(Kp=1.0, Ki=0.1, Kd=0.0, setpoint=target_velocity)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening on {HOST}:{PORT}...")
    conn, addr = s.accept()

    with conn:
        print('Connected by', addr)

        while True:
            data = conn.recv(1024)
            if not data:
                break
            car_position_str, velocity_str, lap_count_str, lap_time_str = data.decode().split(';', 3)

            # Remove parentheses and split by comma
            try:
                x_str, _, y_str = car_position_str.strip('()').split(',')
            except ValueError:
                continue  # Skip if the format is incorrect
            x_position = float(x_str)
            y_position = float(y_str)
            velocity = float(velocity_str)

            # Check if time has changed above the threshold
            if time.time() - last_time > control_change_time:
                last_time = time.time()

                idx, min_dist = find_nearest_point(track_x, track_y, x_position, y_position)
                speed = math.hypot(x_position - last_x_position, y_position - last_y_position) / control_change_time
                last_x_position = x_position
                last_y_position = y_position

                control_steering = steering_controller.update(min_dist, control_change_time)
                control_ap_bp = velocity_controller.update(velocity, control_change_time)
                # simple console error logs from controllers
                steering_error = steering_controller.setpoint - min_dist
                velocity_error = velocity_controller.setpoint - velocity

                print(f"Velocity: {velocity:.2f} | Dist: {min_dist:.2f}")

                if control_ap_bp >= 0:
                    control_apps = min(control_ap_bp, 100.0)
                    control_brakes = 0.0
                else:
                    control_apps = 0.0
                    control_brakes = min(-control_ap_bp, 100.0)

                control_apps = control_apps / 100.0
                control_brakes = control_brakes / 100.0
                control_steering = max(-1.0, min(1.0, control_steering / max_steering))

                # Apply controls
                gamepad.left_trigger_float(value_float=control_brakes)
                gamepad.right_trigger_float(value_float=control_apps)
                gamepad.left_joystick_float(x_value_float=control_steering, y_value_float=0.0)
                gamepad.update()