# Script used to log data from Asetto Corsa. Validates car position readings against a predefined track layout.

import time
import socket
import csv
import math

import numpy as np

HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on

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
    return nearest_idx, min_dist


def interpolate_track_points(original_points, control_change_distance):
    track_x = []
    track_y = []

    if original_points:
        prev_x, prev_y = original_points[0]
        track_x.append(prev_x)
        track_y.append(prev_y)

        for i in range(1, len(original_points)):
            curr_x, curr_y = original_points[i]
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            segment_dist = math.hypot(dx, dy)
            if segment_dist == 0:
                continue
            steps = int(segment_dist // control_change_distance)
            for step in range(1, steps + 1):
                ratio = (step * control_change_distance) / segment_dist
                new_x = prev_x + dx * ratio
                new_y = prev_y + dy * ratio
                track_x.append(new_x)
                track_y.append(new_y)
            prev_x, prev_y = curr_x, curr_y
        # Add the last point
        track_x.append(prev_x)
        track_y.append(prev_y)

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

                print(f"Velocity: {velocity:.2f} | Dist: {min_dist:.2f}")

