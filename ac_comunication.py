import socket
import time
import csv
import math
import numpy as np
import vgamepad as vg
from pynput.keyboard import Key, Controller


class Track:

    def __init__(self, track_file, control_change_distance, track_width=10.0):
        self.track_file = track_file
        self.control_change_distance = control_change_distance
        self.track_width = track_width
        self.normalized_points = []
        self.track_x = []
        self.track_y = []
        self.load_interpolated_track()
        self.normalize_track_idx()

    def load_interpolated_track(self):

        with open(self.track_file, "r") as csvfile:

            reader = csv.reader(csvfile)
            next(reader)  # Skip header

            # Read original points from CSV
            original_points = []
            for row in reader:
                original_points.append(
                    (
                        float(row[0]),  # x
                        float(row[1]),  # y
                    )
                )

        prev_x, prev_y = original_points[0]
        self.track_x.append(prev_x)
        self.track_y.append(prev_y)

        cum_seg_dist = 0.0

        for i in range(1, len(original_points)):
            curr_x, curr_y = original_points[i]
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            segment_dist = math.hypot(dx, dy)
            cum_seg_dist += segment_dist
            if segment_dist == 0:
                continue
            elif cum_seg_dist >= self.control_change_distance:
                self.track_x.append(curr_x)
                self.track_y.append(curr_y)
                cum_seg_dist = 0.0

    def normalize_track_idx(self):
        for i in range(len(self.track_x)):
            self.normalized_points.append(i / len(self.track_x))

    def find_nearest_point(self, point_x, point_y):
        # TODO: compute track direction for each point in preprocessing step

        min_dist = float("inf")
        nearest_idx = -1
        for i, (tx, ty) in enumerate(zip(self.track_x, self.track_y)):
            dist = math.hypot(tx - point_x, ty - point_y)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # determine signed normal deviation from the track at nearest_idx
        tx = self.track_x[nearest_idx]
        ty = self.track_y[nearest_idx]

        # compute tangent direction using neighbors (handle endpoints)
        if nearest_idx == 0:
            t_x = self.track_x[1] - tx
            t_y = self.track_y[1] - ty
        elif nearest_idx == len(self.track_x) - 1:
            t_x = tx - self.track_x[-2]
            t_y = ty - self.track_y[-2]
        else:
            # use central difference for a smoother tangent
            t_x = self.track_x[nearest_idx + 1] - self.track_x[nearest_idx - 1]
            t_y = self.track_y[nearest_idx + 1] - self.track_y[nearest_idx - 1]

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

        # return normalized values for PPO stability
        return self.normalized_points[nearest_idx], signed_dist / (self.track_width / 2)


class AC_Connection:

    def __init__(
        self,
        host,
        port,
        track,
        vel_controller,
        steer_controller,
        control_time_step=0.1,
        reset_threshold=100,
    ):

        self.host = host
        self.port = port
        self.track = track
        self.vel_controller = vel_controller
        self.steer_controller = steer_controller
        self.control_time_step = control_time_step
        self.reset_threshold = reset_threshold

        self.gamepad = vg.VX360Gamepad()
        self.keyboard = Controller()

        self.server_sock = None
        self.conn = None
        self.addr = None
        self.last_control_time = time.time()
        self.last_action = 0.0, 0.0
        self.last_positions = 0.0, 0.0
        self.position_change_tracker = 0.0
        self.position_change_count = 0

    def connect(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen()
        print(f"Listening on {self.host}:{self.port}...")
        self.conn, self.addr = self.server_sock.accept()

        print(f"Connected by {self.addr}")

    def control_step(self, action):
        reset = False

        if not self.conn:
            raise RuntimeError("No active connection.")

        while True:
            if time.time() - self.last_control_time >= self.control_time_step:
                self.last_control_time = time.time()

                # Read first available chunk (blocking)
                try:
                    data = self.conn.recv(4096)
                except (BlockingIOError, InterruptedError):
                    break
                if not data:
                    self.close()
                    break

                last_line = None
                buf = data.decode(errors="replace")
                try:
                    self.conn.setblocking(False)
                    while True:
                        try:
                            more = self.conn.recv(4096)
                        except (BlockingIOError, InterruptedError):
                            break
                        if not more:
                            if not last_line and buf:
                                last_line = buf.strip()
                            break
                        buf += more.decode(errors="replace")
                    idx = buf.rfind("\n")
                    if idx != -1:
                        last_line = buf[:idx].split("\n")[-1].strip()
                    elif buf and last_line is None:
                        last_line = buf.strip()
                finally:
                    try:
                        self.conn.setblocking(True)
                    except Exception:
                        pass

                fields = last_line.replace("\r", "").split(";")
                if len(fields) < 4:
                    break

                car_position_str, velocity_str, lap_count_str, lap_time_str = fields[:4]
                x_str, _, y_str = car_position_str.strip("()").split(",")
                x_position = float(x_str)
                y_position = float(y_str)
                velocity = float(velocity_str)

                idx, min_dist = self.track.find_nearest_point(x_position, y_position)

                # Reset if locked and not moving
                self.position_change_tracker += math.hypot(
                    x_position - self.last_positions[0],
                    y_position - self.last_positions[1],
                )

                if self.position_change_count >= self.reset_threshold:
                    if self.position_change_tracker < 1.0:
                        reset = True
                    self.position_change_tracker = 0.0
                    self.position_change_count = 0

                self.last_positions = (x_position, y_position)
                self.position_change_count += 1

                # Reset if out of track bounds
                if self.track.track_width < abs(min_dist):
                    reset = True

                # Update PID controllers
                control_steering = self.steer_controller.update(
                    min_dist, self.control_time_step
                )
                control_ap_bp = self.vel_controller.update(
                    velocity, self.control_time_step
                )

                # Clip controls
                control_ap_bp = max(-1.0, min(1.0, control_ap_bp))
                control_steering = max(-1.0, min(1.0, control_steering))

                # Reward function calculation
                action_steering, action_ap_bp = action

                reward = (4
                    - (np.square(control_steering - action_steering))
                    - (np.square(control_ap_bp - action_ap_bp))
                )

                self.last_action = action_steering, action_ap_bp

                if action_ap_bp >= 0:
                    action_apps = min(action_ap_bp, 1.0)
                    action_brakes = 0.0
                else:
                    action_apps = 0.0
                    action_brakes = min(-action_ap_bp, 1.0)

                # Apply controls
                self.gamepad.left_trigger_float(value_float=action_brakes)
                self.gamepad.right_trigger_float(value_float=action_apps)
                self.gamepad.left_joystick_float(
                    x_value_float=action_steering, y_value_float=0.0
                )
                self.gamepad.update()

                sc_velocity = velocity / 10.0 # scale for PPO stability

                return (idx, min_dist, sc_velocity), reward, reset

    def reset(self):
        self.keyboard.press(Key.ctrl)
        self.keyboard.press("o")
        time.sleep(0.1)
        self.keyboard.release("o")
        self.keyboard.release(Key.ctrl)
        time.sleep(0.1)
        self.keyboard.press(Key.ctrl)
        self.keyboard.press("k")
        time.sleep(0.1)
        self.keyboard.release("k")
        self.keyboard.release(Key.ctrl)

        self.conn.setblocking(False)
        try:
            while True:
                _ = self.conn.recv(4096)
        except BlockingIOError:
            pass  # Buffer is now empty
        self.conn.setblocking(True)

        time.sleep(0.1)

        while True:
            try:
                data = self.conn.recv(4096)
                if not self.conn:
                    self.close()
                break
            except (BlockingIOError, InterruptedError):
                continue

        data = data.decode(errors="replace")
        fields = data.replace("\r", "").split(";")

        car_position_str, velocity_str, lap_count_str, lap_time_str = fields[:4]
        x_str, _, y_str = car_position_str.strip("()").split(",")
        x_position = float(x_str)
        y_position = float(y_str)
        velocity = float(velocity_str)

        idx, min_dist = self.track.find_nearest_point(x_position, y_position)

        return (idx, min_dist, velocity)

    def close(self):

        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None
