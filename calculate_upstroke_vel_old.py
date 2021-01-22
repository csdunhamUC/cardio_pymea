# Author: Christopher S. Dunham
# 12/4/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import time
from numba import njit
import numpy as np
import pandas as pd


# Calculates upstroke velocity (dV/dt)
def calculate_upstroke_vel(analysisGUI, cm_beats, upstroke_vel, heat_map, input_param, electrode_config):
    try:
        if hasattr(upstroke_vel, 'param_dist_raw') is True:
            print("Clearing old dV/dt max data before running new calculation...")
            delattr(upstroke_vel, 'param_dist_raw')

        # Clock the time it takes to run the calculation.
        start_time = time.process_time()
        print("Calculating upstroke velocity per beat.")

        num_of_electrodes = len(cm_beats.dist_beats.columns)
        mode_of_beats = int(cm_beats.beat_count_dist_mode[0])
        cm_beats_dist_beats_as_array = cm_beats.dist_beats.to_numpy()
        cm_beats_y_axis_as_array = cm_beats.y_axis.to_numpy()
        per_electrode_mode_of_beats = np.zeros(num_of_electrodes)

        for electrode in range(num_of_electrodes):
            per_electrode_mode_of_beats[electrode] = len(cm_beats.dist_beats.iloc[0:, electrode].dropna())

        dvdt_max = dvdt_calc_for_numba(mode_of_beats, num_of_electrodes, cm_beats_dist_beats_as_array,
                                       cm_beats_y_axis_as_array, per_electrode_mode_of_beats)

        upstroke_vel.final_dist_beat_count = []
        for beat in range(int(cm_beats.beat_count_dist_mode[0])):
            upstroke_vel.final_dist_beat_count.append('Beat ' + str(beat + 1))

        upstroke_vel.param_dist_raw = pd.DataFrame(dvdt_max, index=upstroke_vel.final_dist_beat_count).T
        upstroke_vel.param_dist_normalized = upstroke_vel.param_dist_raw.sub(upstroke_vel.param_dist_raw.min(axis=0),
                                                                             axis=1)

        # Normalized dV/dt across the dataset.
        upstroke_vel.param_dist_normalized_max = upstroke_vel.param_dist_normalized.max().max()

        # Mean dV/dt across the dataset.
        upstroke_vel.param_dist_normalized_mean = np.nanmean(upstroke_vel.param_dist_normalized)

        upstroke_vel.param_dist_normalized.index = electrode_config.electrode_names
        upstroke_vel.param_dist_normalized.insert(0, 'Electrode', electrode_config.electrode_names)
        upstroke_vel.param_dist_normalized.insert(1, 'X', electrode_config.electrode_coords_x)
        upstroke_vel.param_dist_normalized.insert(2, 'Y', electrode_config.electrode_coords_y)

        # Assign name to resulting dataframe.
        upstroke_vel.param_dist_normalized.name = 'Upstroke Velocity'

        # Set slider value to maximum number of beats
        analysisGUI.mea_beat_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))

        print("Done")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        print(end_time - start_time)
    except AttributeError:
        print("Please use Find Peaks first.")


@njit
def dvdt_calc_for_numba(mode_of_beats, num_of_electrodes, cm_beats_dist_beats_as_array, cm_beats_y_axis_as_array, per_electrode_mode_of_beats):
    temp_slope = np.zeros((num_of_electrodes, 5))
    temp_dvdt_max = np.zeros(num_of_electrodes)
    x_values = np.zeros((2, 5))
    y_values = np.zeros((2, 5))
    dvdt_max = np.zeros((mode_of_beats, num_of_electrodes))

    for beat in range(mode_of_beats):
        for electrode in range(num_of_electrodes):
            if mode_of_beats == per_electrode_mode_of_beats[electrode]:
                x_2_1 = cm_beats_dist_beats_as_array[beat, electrode]
                x_2_2 = x_2_1 - 1
                x_2_3 = x_2_2 - 1
                x_2_4 = x_2_3 - 1
                x_2_5 = x_2_4 - 1

                x_1_1 = x_2_1 - 1
                x_1_2 = x_2_2 - 1
                x_1_3 = x_2_3 - 1
                x_1_4 = x_2_4 - 1
                x_1_5 = x_2_5 - 1

                y_2_1 = cm_beats_y_axis_as_array[int(x_2_1), electrode]
                y_2_2 = cm_beats_y_axis_as_array[int(x_2_2), electrode]
                y_2_3 = cm_beats_y_axis_as_array[int(x_2_3), electrode]
                y_2_4 = cm_beats_y_axis_as_array[int(x_2_4), electrode]
                y_2_5 = cm_beats_y_axis_as_array[int(x_2_5), electrode]

                y_1_1 = cm_beats_y_axis_as_array[int(x_1_1), electrode]
                y_1_2 = cm_beats_y_axis_as_array[int(x_1_2), electrode]
                y_1_3 = cm_beats_y_axis_as_array[int(x_1_3), electrode]
                y_1_4 = cm_beats_y_axis_as_array[int(x_1_4), electrode]
                y_1_5 = cm_beats_y_axis_as_array[int(x_1_5), electrode]

                x_values[0, :] = [x_2_1, x_2_2, x_2_3, x_2_4, x_2_5]
                x_values[1, :] = [x_1_1, x_1_2, x_1_3, x_1_4, x_1_5]
                y_values[0, :] = [y_2_1, y_2_2, y_2_3, y_2_4, y_2_5]
                y_values[1, :] = [y_1_1, y_1_2, y_1_3, y_1_4, y_1_5]

                y_diff = y_values[0, :] - y_values[1, :]
                x_diff = x_values[0, :] - x_values[1, :]

                calc_slope = np.divide(y_diff, x_diff)
                temp_slope[electrode] = calc_slope
            else:
                temp_slope[electrode] = [np.nan, np.nan, np.nan, np.nan, np.nan]
            temp_dvdt_max[electrode] = max(temp_slope[electrode])

        dvdt_max[beat] = temp_dvdt_max

    return dvdt_max