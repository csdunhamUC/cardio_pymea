# Scratch File
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random


class ElectrodeConfig:
    # Electrode names and coordinates, using the system defined by CSD where 
    # origin (0,0) is at upper left corner of MEA.  Configured for 200x30um 
    # inter-electrode spacing and electrode diameter, respectively.

    def __init__(self):
        self.mea_120_coordinates = {
            'F7': [1150, 1380], 'F8': [1150, 1610], 'F12': [1150, 2530], 
            'F11': [1150, 2300], 'F10': [1150, 2070], 'F9': [1150, 1840], 
            'E12': [920, 2530], 'E11': [920, 2300], 'E10': [920, 2070], 
            'E9': [920, 1840], 'D12': [690, 2530], 'D11': [690, 2300], 
            'D10': [690, 2070], 'D9': [690, 1840], 'C11': [460, 2300],
            'C10': [460, 2070], 'B10': [230, 2070], 'E8': [920, 1610], 
            'C9': [460, 1840], 'B9': [230, 1840], 'A9': [0, 1840], 
            'D8': [690, 1610], 'C8': [460, 1610], 'B8': [230, 1610], 
            'A8': [0, 1610], 'D7': [690, 1380], 'C7': [460, 1380], 
            'B7': [230, 1380], 'A7': [0, 1380], 'E7': [920, 1380], 
            'F6': [1150, 1150], 'E6': [920, 1150], 'A6': [0, 1150], 
            'B6': [230, 1150], 'C6': [460, 1150], 'D6': [690, 1150], 
            'A5': [0, 920], 'B5': [230, 920], 'C5': [460, 920], 
            'D5': [690, 920], 'A4': [0, 690], 'B4': [230, 690], 
            'C4': [460, 690], 'D4': [690, 690], 'B3': [230, 460],
            'C3': [460, 460], 'C2': [460, 230], 'E5': [920, 920], 
            'D3': [690, 460], 'D2': [690, 230], 'D1': [690, 0], 
            'E4': [920, 690], 'E3': [920, 460], 'E2': [920, 230], 
            'E1': [920, 0], 'F4': [1150, 690], 'F3': [1150, 460], 
            'F2': [1150, 230], 'F1': [1150, 0], 'F5': [1150, 920], 
            'G6': [1380, 1150], 'G5': [1380, 920], 'G1': [1380, 0], 
            'G2': [1380, 230], 'G3': [1380, 460], 'G4': [1380, 690], 
            'H1': [1610, 0], 'H2': [1610, 230], 'H3': [1610, 460], 
            'H4': [1610, 690], 'J1': [1840, 0], 'J2': [1840, 230], 
            'J3': [1840, 460], 'J4': [1840, 690], 'K2': [2070, 230], 
            'K3': [2070, 460], 'L3': [2300, 460], 'H5': [1610, 920], 
            'K4': [2070, 690], 'L4': [2300, 690], 'M4': [2530, 690], 
            'J5': [1840, 920], 'K5': [2070, 920], 'L5': [2300, 920], 
            'M5': [2530, 920], 'J6': [1840, 1150], 'K6': [2070, 1150], 
            'L6': [2300, 1150], 'M6': [2530, 1150], 'H6': [1610, 1150],
            'G7': [1380, 1380], 'H7': [1610, 1380], 'M7': [2530, 1380], 
            'L7': [2300, 1380], 'K7': [2070, 1380], 'J7': [1840, 1380], 
            'M8': [2530, 1610], 'L8': [2300, 1610], 'K8': [2070, 1610], 
            'J8': [1840, 1610], 'M9': [2530, 1840], 'L9': [2300, 1840], 
            'K9': [2070, 1840], 'J9': [1840, 1840], 'L10': [2300, 2070],
            'K10': [2070, 2070], 'K11': [2070, 2300], 'H8': [1610, 1610], 
            'J10': [1840, 2070], 'J11': [1840, 2300], 'J12': [1840, 2530], 
            'H9': [1610, 1840], 'H10': [1610, 2070], 'H11': [1610, 2300], 
            'H12': [1610, 2530], 'G9': [1380, 1840], 'G10': [1380, 2070], 
            'G11': [1380, 2300], 'G12': [1380, 2530], 'G8': [1380, 1610]}
 

my_elecs = ElectrodeConfig()
my_elec_names = [elec for elec in my_elecs.mea_120_coordinates.keys()]
print(my_elec_names)

test_df = pd.DataFrame(np.random.randint(-10,300, size=(10000,120)),
    columns=my_elec_names)
tpose_df = test_df.T

electrode_coords_x = np.array(
    [i[0] for i in my_elecs.mea_120_coordinates.values()])
electrode_coords_y = np.array(
    [i[1] for i in my_elecs.mea_120_coordinates.values()])

tpose_df.insert(0, "Electrode", my_elec_names)
tpose_df.insert(1, "X", electrode_coords_x)
tpose_df.insert(2, "Y", electrode_coords_y)

print(tpose_df)

pivot_tpose_df = tpose_df.pivot(
    index='Y', 
    columns='X', 
    values='Electrode')

print(pivot_tpose_df)

K_max = 12
K_min = 0
L_max = 12
L_min = 0
ax = plt.subplot(111)
x_offset = 7 # tune these
y_offset = 7 # tune these
plt.setp(ax, 'frame_on', False)
ax.set_ylim([0, (K_max-K_min +1)*y_offset ])
ax.set_xlim([0, (L_max - L_min+1)*x_offset])
ax.set_xticks([])
ax.set_yticks([])
ax.grid('off')


# Iterate through the 12x12 grid for each electrode from my_elec_names?
# Iterate through the coordinate values?

# Code below loops in a column then row manner, starting from the bottom of the
# plot axis.
# for k in np.arange(K_min, K_max + 1): # Column
#     for l in np.arange(L_min, L_max + 1): # Row
#         ax.plot(np.arange(5) + l*x_offset, 5+np.random.rand(5) + k*y_offset,
#                 'r-o', ms=1, mew=0, mfc='r')
#         ax.plot(np.arange(5) + l*x_offset, 3+np.random.rand(5) + k*y_offset,
#                 'b-o', ms=1, mew=0, mfc='b')
#         ax.annotate(
#             'K={},L={}'.format(k, l), 
#             (2.5+ (k)*x_offset,l*y_offset), 
#             size=8,
#             ha='center')

for col_pos, col in enumerate(pivot_tpose_df.columns):
    for row_pos, val in enumerate(pivot_tpose_df[col]):
        if val != val:
            ax.plot(np.arange(5) + row_pos*x_offset, 
            5+np.random.rand(5) + col_pos*y_offset, 
            color='white')
            print(f"NaN: {val}")
        else:
            ax.plot(np.arange(5) + row_pos*x_offset, 
            5+np.random.rand(5) + col_pos*y_offset, 
            color='black')
            print(f"{val} is NOT NaN!")
    # print(f"Column {col}:\n{pivot_tpose_df[col]}")

plt.show()
