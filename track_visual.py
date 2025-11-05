import matplotlib.pyplot as plt

with open("position_data.csv", "r") as file:
    lines = file.readlines()

    x_positions = []
    y_positions = []
    z_positions = []

    for line in lines[1:]:  # Skip header
        x, y, z = map(float, line.split(','))
        x_positions.append(x)
        y_positions.append(y)
        z_positions.append(z)

plt.plot(x_positions, y_positions, label='Track Path')
plt.axis('equal')
plt.show()