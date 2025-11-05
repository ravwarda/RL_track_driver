import sys
import ac
import acsys

car_position_data = []
car_inputs_data = []
collect = False

def acMain(ac_version):
    def toggle_collect(*args):
        global collect
        collect = not collect
        ac.setText(collect_button, "Stop Collecting" if collect else "Start Collecting")

    global appWindow
    appWindow = ac.newApp("track_extractor")
    global collect_button
    collect_button = ac.addButton(appWindow, "Start Collecting")
    ac.setPosition(collect_button, 10, 30)
    ac.setSize(collect_button, 200, 30)
    ac.addOnClickedListener(collect_button, toggle_collect)

    return "track_extractor"

def acUpdate(deltaT):
    global car_position_data
    global car_inputs_data
    global collect
    if collect:
        car_position = ac.getCarState(0, acsys.CS.WorldPosition)
        gas_pedal = ac.getCarState(0, acsys.CS.Gas)
        brake_pedal = ac.getCarState(0, acsys.CS.Brake)
        steering_position = ac.getCarState(0, acsys.CS.Steer)
        lap_count = ac.getCarState(0, acsys.CS.LapCount)
        lap_time = ac.getCarState(0, acsys.CS.LapTime)
        car_position_data.append((car_position[0], car_position[1], car_position[2]))
        car_inputs_data.append((gas_pedal, brake_pedal, steering_position, lap_count, lap_time))

def acShutdown():
    global car_position_data

    # Save the collected data to CSV files when the app is closed
    with open("C:/Users/Wrafi/Documents/position_data.csv", "w", newline="") as file:
        file.write("x,y,z\n")
        for x, z, y in car_position_data: # AC gives position as x,z,y
            file.write("{},{},{}\n".format(x, y, z))

    with open("C:/Users/Wrafi/Documents/inputs_data.csv", "w", newline="") as file:
        file.write("gas,brake,steer,lap_count,lap_time\n")
        for gas, brake, steer, lap_count, lap_time in car_inputs_data:
            file.write("{},{},{},{},{}\n".format(gas, brake, steer, lap_count, lap_time))