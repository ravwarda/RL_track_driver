import vgamepad as vg
from pynput.keyboard import Key, Controller
import time
import socket
import csv
import math

import random
import numpy as np
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import logging  # <-- Add logging import

HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on

gamepad = vg.VX360Gamepad()
keyboard = Controller()

# Configuration constants
max_steering = 105
max_brake = 50
max_apps = 100

control_change_time = 0.1
control_change_distance = 0.5
max_deviation_dist = 15

LEARNING_ENABLED = True  # Set to False to disable learning and only run inference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def find_nearest_point(track_x, track_y, point_x, point_y):
    min_dist = float('inf')
    nearest_idx = -1
    for i, (tx, ty) in enumerate(zip(track_x, track_y)):
        dist = math.hypot(tx - point_x, ty - point_y)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx, min_dist


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
    apps = []
    brakes = []
    steering_positions = []

    if original_points:
        prev_x, prev_y, prev_apps, prev_brakes, prev_steering = original_points[0]
        track_x.append(prev_x)
        track_y.append(prev_y)
        apps.append(prev_apps)
        brakes.append(prev_brakes)
        steering_positions.append(prev_steering)

        for i in range(1, len(original_points)):
            curr_x, curr_y, curr_apps, curr_brakes, curr_steering = original_points[i]
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
                # Use the last checked point's control values
                apps.append(prev_apps)
                brakes.append(prev_brakes)
                steering_positions.append(prev_steering)
            prev_x, prev_y, prev_apps, prev_brakes, prev_steering = curr_x, curr_y, curr_apps, curr_brakes, curr_steering
        # Add the last point
        track_x.append(prev_x)
        track_y.append(prev_y)
        apps.append(prev_apps)
        brakes.append(prev_brakes)
        steering_positions.append(prev_steering)

    return track_x, track_y, apps, brakes, steering_positions

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def compute_reward(progress_speed, normal_deviation, speed, track_idx):
    reward = -0.1
    reward -= normal_deviation**2 * 0.1
    reward += speed * 0.01
    reward += track_idx * 0.02 * control_change_distance  # Encourage progress on the track
    if progress_speed > 0:
        reward += progress_speed
    return reward
    

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

# DQN action/state discretization
STEERING_BINS_N = 40
AP_BP_BINS_N = 40
NUM_ACTIONS = STEERING_BINS_N * AP_BP_BINS_N

STEERING_BINS = np.linspace(-1, 1, STEERING_BINS_N)
AP_BP_BINS = np.linspace(-1, 1, AP_BP_BINS_N)

# progress_speed, track_position, normal_deviation
NUM_STATES = 3

# --- DQN AGENT SETUP ---
policy_net = DQN(NUM_STATES, NUM_ACTIONS)
target_net = DQN(NUM_STATES, NUM_ACTIONS)

# Try to load saved models if they exist
import os
policy_path = "dqn_policy_net.pth"
target_path = "dqn_target_net.pth"
if os.path.exists(policy_path):
    policy_net.load_state_dict(torch.load(policy_path))
    print(f"Loaded policy_net weights from {policy_path}")
if os.path.exists(target_path):
    target_net.load_state_dict(torch.load(target_path))
    print(f"Loaded target_net weights from {target_path}")

target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

with open('optim_control_order.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    track_x = []
    track_y = []
    
    # Read original points from CSV
    original_points = []
    for row in reader:
        original_points.append((
            float(row[4]),  # x
            float(row[5]),  # y
        ))

    # Interpolate points so that distance between them is control_change_dist
    track_x, track_y = interpolate_track_points(original_points, control_change_distance)


last_time = time.time()
last_idx = 0
nonprogress_count = 0
last_x_position = 0.0
last_y_position = 0.0

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening on {HOST}:{PORT}...")
    conn, addr = s.accept()

    with conn:
        print('Connected by', addr)
        try:
            episode = 0
            episode_reward = 0.0
            episode_steps = 0
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                car_position_str, lap_count_str, lap_time_str = data.decode().split(';', 2)

                # Remove parentheses and split by comma
                try:
                    x_str, _, y_str = car_position_str.strip('()').split(',')
                except ValueError:
                    continue  # Skip if the format is incorrect
                x_position = float(x_str)
                y_position = float(y_str)

                # Check if time has changed above the threshold
                if time.time() - last_time > control_change_time:
                    last_time = time.time()

                    idx, min_dist = find_nearest_point(track_x, track_y, x_position, y_position)
                    speed = math.hypot(x_position - last_x_position, y_position - last_y_position) / control_change_time
                    last_x_position = x_position
                    last_y_position = y_position

                    # Check if the car is making progress
                    if idx <= last_idx:
                        nonprogress_count += 1
                    else:
                        nonprogress_count = 0
                    
                    # DQN states
                    progress_speed = idx - last_idx
                    track_position = idx
                    normal_deviation = min_dist

                    last_idx = idx

                    # Check for restart
                    if min_dist > max_deviation_dist or nonprogress_count > 10 / control_change_time:
                        if LEARNING_ENABLED:
                            logging.info(
                                f"Episode {episode} end | steps={episode_steps} | total_reward={episode_reward:.2f} | deviation={min_dist:.2f} | nonprogress={nonprogress_count}"
                            )
                        episode += 1
                        episode_reward = 0.0
                        episode_steps = 0
                        check_restart()
                        last_idx = 0
                        nonprogress_count = 0
                        continue

                    # --- DQN AGENT LOGIC ---
                    state = np.array([progress_speed, track_position, normal_deviation], dtype=np.float32)

                    steps_done = getattr(globals(), 'steps_done', 0)
                    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
                    if not LEARNING_ENABLED:
                        # In inference-only mode, always exploit (no exploration)
                        with torch.no_grad():
                            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                            action = policy_net(state_tensor).max(1)[1].view(1, 1).item()
                    else:
                        if random.random() > eps_threshold:
                            # Exploit: use DQN
                            with torch.no_grad():
                                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                                action = policy_net(state_tensor).max(1)[1].view(1, 1).item()
                        else:
                            # Explore: random action
                                action = random.randrange(NUM_ACTIONS)
                    globals()['steps_done'] = steps_done + 1

                    # Decode action into steering and ap_bp bins
                    steering_idx = action // AP_BP_BINS_N
                    ap_bp_idx = action % AP_BP_BINS_N
                    control_steering = float(STEERING_BINS[steering_idx])
                    control_ap_bp = float(AP_BP_BINS[ap_bp_idx])
                    
                    if control_ap_bp >= 0:
                        control_apps = control_ap_bp
                        control_brakes = 0.0
                    else:
                        control_apps = 0.0
                        control_brakes = -control_ap_bp

                    # Apply controls
                    gamepad.left_trigger_float(value_float=control_brakes)
                    gamepad.right_trigger_float(value_float=control_apps)
                    gamepad.left_joystick_float(x_value_float=control_steering, y_value_float=0.0)
                    gamepad.update()

                    # --- DQN Experience Replay ---
                    prev_state = globals().get('prev_state', None)
                    prev_action = globals().get('prev_action', None)
                    if LEARNING_ENABLED:
                        if prev_state is not None and prev_action is not None:
                            reward = compute_reward(progress_speed, normal_deviation, speed, idx)
                            episode_reward += reward
                            episode_steps += 1
                            memory.push(torch.tensor(prev_state, dtype=torch.float32),
                                        torch.tensor([[prev_action]], dtype=torch.long),
                                        torch.tensor(state, dtype=torch.float32),
                                        torch.tensor([reward], dtype=torch.float32))
                        else:
                            episode_steps += 1
                        globals()['prev_state'] = state
                        globals()['prev_action'] = action

                        # Optimize DQN
                        if len(memory) >= BATCH_SIZE:
                            transitions = memory.sample(BATCH_SIZE)
                            batch = Transition(*zip(*transitions))

                            state_batch = torch.stack(batch.state)
                            action_batch = torch.cat(batch.action)
                            reward_batch = torch.cat(batch.reward)
                            next_state_batch = torch.stack(batch.next_state)

                            # Compute Q(s_t, a)
                            state_action_values = policy_net(state_batch).gather(1, action_batch)

                            # Compute V(s_{t+1})
                            with torch.no_grad():
                                next_state_values = target_net(next_state_batch).max(1)[0]
                            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                            # Compute loss
                            loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

                            # Optimize the model
                            optimizer.zero_grad()
                            loss.backward()
                            for param in policy_net.parameters():
                                param.grad.data.clamp_(-1, 1)
                            optimizer.step()

                            # Soft update of target network
                            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                                target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
                            # Log loss and Q statistics every 100 optimization steps
                            optimize_count = getattr(globals(), 'optimize_count', 0) + 1
                            if optimize_count % 100 == 0:
                                avg_loss = loss.item()
                                avg_q = state_action_values.mean().item()
                                logging.info(
                                    f"Ep {episode} Step {episode_steps} | Loss: {avg_loss:.4f} | Avg Q: {avg_q:.4f} | Memory: {len(memory)}"
                                )
                            globals()['optimize_count'] = optimize_count
        finally:
            if LEARNING_ENABLED:
                torch.save(policy_net.state_dict(), "dqn_policy_net.pth")
                torch.save(target_net.state_dict(), "dqn_target_net.pth")
                print("Saved DQN model weights.")