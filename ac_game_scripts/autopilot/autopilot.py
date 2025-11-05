import sys
import ac
import acsys
import csv
import socket

drive = False
sock = None

def toggle_collect(*args):
    global drive
    drive = not drive
    ac.setText(switch_button, "Stop Autopilot" if drive else "Start Autopilot")

def acMain(ac_version):

    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', 65432))
    except Exception as e:
        ac.log("Socket connection failed: " + str(e))

    global autopilot_app_window
    autopilot_app_window = ac.newApp("autopilot")
    global switch_button
    switch_button = ac.addButton(autopilot_app_window, "Start Autopilot")
    ac.setPosition(switch_button, 10, 30)
    ac.setSize(switch_button, 200, 30)
    ac.addOnClickedListener(switch_button, toggle_collect)

    return "autopilot"

def acUpdate(deltaT):
    global drive
    if drive:
        car_position = ac.getCarState(0, acsys.CS.WorldPosition)
        velocity = ac.getCarState(0, acsys.CS.SpeedKMH)
        lap_count = ac.getCarState(0, acsys.CS.LapCount)
        lap_time = ac.getCarState(0, acsys.CS.LapTime)

        data = "{};{};{};{}".format(car_position, velocity, lap_count, lap_time)
        try:
            if sock:
                sock.sendall(data.encode())
        except Exception as e:
            ac.log("Socket send failed: " + str(e))