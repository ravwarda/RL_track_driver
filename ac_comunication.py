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
        self.track_direction = []
        self.track_curvature = []
        self.load_interpolated_track()
        self.normalize_track_idx()
        self.compute_track_properties()

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

    def compute_track_properties(self):
        """Compute tangent direction angle and curvature for all track points."""
        n_points = len(self.track_x)
        
        for i in range(n_points):
            # Compute tangent direction using neighbors
            if i == 0:
                t_x = self.track_x[1] - self.track_x[i]
                t_y = self.track_y[1] - self.track_y[i]
            elif i == n_points - 1:
                t_x = self.track_x[i] - self.track_x[i-1]
                t_y = self.track_y[i] - self.track_y[i-1]
            else:
                # Central difference for smoother tangent
                t_x = self.track_x[i+1] - self.track_x[i-1]
                t_y = self.track_y[i+1] - self.track_y[i-1]
            
            # Compute angle (in radians)
            angle = math.atan2(t_y, t_x)
            self.track_direction.append(angle)
            
            # Compute curvature using finite differences
            if i == 0 or i == n_points - 1:
                curvature = 0.0
            else:
                # Get angles at i-1, i, i+1
                angle_prev = math.atan2(
                    self.track_y[i] - self.track_y[i-1],
                    self.track_x[i] - self.track_x[i-1]
                )
                angle_next = math.atan2(
                    self.track_y[i+1] - self.track_y[i],
                    self.track_x[i+1] - self.track_x[i]
                )
                
                # Angular difference (handle wrapping)
                angle_diff = angle_next - angle_prev
                # Normalize to [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # Average segment length
                dist_prev = math.hypot(
                    self.track_x[i] - self.track_x[i-1],
                    self.track_y[i] - self.track_y[i-1]
                )
                dist_next = math.hypot(
                    self.track_x[i+1] - self.track_x[i],
                    self.track_y[i+1] - self.track_y[i]
                )
                avg_dist = (dist_prev + dist_next) / 2.0
                
                if avg_dist > 0:
                    curvature = angle_diff / avg_dist
                else:
                    curvature = 0.0
            
            self.track_curvature.append(curvature)

    def curvature_ahead(self, idx, amount=6, step=5):
        """Return a list of curvature values for points ahead of idx.
        Wraps around the track if necessary.
        """
        if not self.track_curvature:
            return []

        n = len(self.track_curvature)
        if n == 0:
            return []

        curvatures = [self.track_curvature[idx % n]]
        base = idx % n
        for i in range(1, int(amount)):
            next_idx = (base + i * step) % n
            curvatures.append(self.track_curvature[next_idx])

        return curvatures


    def find_nearest_point(self, point_x, point_y, car_heading):
        min_dist = float("inf")
        nearest_idx = -1
        for i, (tx, ty) in enumerate(zip(self.track_x, self.track_y)):
            dist = math.hypot(tx - point_x, ty - point_y)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Get precomputed track angle
        track_angle = self.track_direction[nearest_idx]
        
        # Left-hand normal direction (perpendicular to track)
        n_x = -math.sin(track_angle)
        n_y = math.cos(track_angle)

        # Vector from track point to queried point
        tx = self.track_x[nearest_idx]
        ty = self.track_y[nearest_idx]
        v_x = point_x - tx
        v_y = point_y - ty

        # Signed distance: positive if point is to the left of the track direction
        signed_dist = v_x * n_x + v_y * n_y
        
        # Angle deviation: difference between car heading and track direction
        angle_deviation = car_heading - track_angle
        # Normalize to [-pi, pi]
        while angle_deviation > math.pi:
            angle_deviation -= 2 * math.pi
        while angle_deviation < -math.pi:
            angle_deviation += 2 * math.pi

        # Return normalized values for PPO stability
        return (
            self.normalized_points[nearest_idx],
            signed_dist / (self.track_width / 2),
            angle_deviation,
            self.curvature_ahead(nearest_idx)
        )


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
        residual_scale=0.5,
    ):

        self.host = host
        self.port = port
        self.track = track
        self.vel_controller = vel_controller
        self.steer_controller = steer_controller
        self.control_time_step = control_time_step
        self.reset_threshold = reset_threshold
        self.residual_scale = residual_scale

        self.gamepad = vg.VX360Gamepad()
        self.keyboard = Controller()

        self.server_sock = None
        self.conn = None
        self.addr = None
        self.last_control_time = time.time()
        self.last_action = 0.0, 0.0
        self.last_track_idx = 0.0
        self.last_positions = None  # Changed to None initially
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
        penalty = 0.0

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

                # Compute car heading from movement
                if self.last_positions is not None:
                    dx = x_position - self.last_positions[0]
                    dy = y_position - self.last_positions[1]
                    car_heading = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
                else:
                    car_heading = 0.0  # First step, no previous position

                idx, min_dist, angle_deviation, curvatures = self.track.find_nearest_point(
                    x_position, y_position, car_heading
                )

                # Reset if locked and not moving
                if self.last_positions is not None:
                    self.position_change_tracker += math.hypot(
                        x_position - self.last_positions[0],
                        y_position - self.last_positions[1],
                    )

                if self.position_change_count >= self.reset_threshold:
                    if self.position_change_tracker < 1.0:
                        penalty = -5.0
                        reset = True
                    self.position_change_tracker = 0.0
                    self.position_change_count = 0

                self.last_positions = (x_position, y_position)
                self.position_change_count += 1

                # Reset if out of track bounds
                if self.track.track_width < abs(min_dist):
                    penalty = -1.0
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

                # Apply PPO action residuals
                action_steering, action_ap_bp = action
                action_steering *= self.residual_scale
                action_ap_bp *= self.residual_scale

                combined_steering = control_steering + action_steering
                combined_ap_bp = control_ap_bp + action_ap_bp

                if combined_ap_bp >= 0:
                    combined_apps = min(combined_ap_bp, 1.0)
                    combined_brakes = 0.0
                else:
                    combined_apps = 0.0
                    combined_brakes = min(-combined_ap_bp, 1.0)
                combined_steering = max(-1.0, min(1.0, combined_steering))

                # Apply controls
                self.gamepad.left_trigger_float(value_float=combined_brakes)
                self.gamepad.right_trigger_float(value_float=combined_apps)
                self.gamepad.left_joystick_float(
                    x_value_float=combined_steering, y_value_float=0.0
                )
                self.gamepad.update()

                sc_velocity = velocity / 10.0 # scale for PPO stability

                # Reward function calculation
                track_idx_diff = idx - self.last_track_idx
                steering_diff = abs(action_steering - self.last_action[0])

                reward = (
                    20 * track_idx_diff # reward for progress along track
                    + 0.001 * (-0.5 * steering_diff + 1.0) # small reward for smooth steering
                    + 0.001 * min(1 - min_dist ** 2, 0.0) # penalty for being off track
                    - 0.01 * math.exp(-2.0 * velocity + 2.0) # penalty for very low speed
                    + penalty # large penalties for resets
                )

                self.last_action = action_steering, action_ap_bp
                self.last_track_idx = idx

                return (idx, min_dist, control_ap_bp, control_steering, sc_velocity, angle_deviation, *curvatures), reward, reset

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

        idx, min_dist, angle_deviation, curvatures = self.track.find_nearest_point(
            x_position, y_position, 0.0
        )

        self.last_track_idx = 0.0
        self.steer_controller.restart()
        self.vel_controller.restart()

        return (idx, min_dist, 0.0, 0.0, 0.0, angle_deviation, *curvatures)

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
