# Author: Christopher S. Dunham
# 12/4/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import time
from numba import njit
import numpy as np
import pandas as pd


# Calculates local activation time (LAT)
def calculate_lat(analysisGUI, cm_beats, local_act_time, heat_map, input_param, electrode_config):
    try:
        if hasattr(local_act_time, 'param_dist_raw') is True:
            print("Clearing old LAT data before running new calculation...")
            delattr(local_act_time, 'param_dist_raw')

        # Clock the time it takes to run the calculation.
        start_time = time.process_time()
        print("Calculating LAT per beat.")

        num_of_electrodes = len(cm_beats.dist_beats.columns)
        mode_of_beats = int(cm_beats.beat_count_dist_mode[0])
        cm_beats_dist_beats_as_array = cm_beats.dist_beats.to_numpy()
        cm_beats_y_axis_as_array = cm_beats.y_axis.to_numpy()
        per_electrode_mode_of_beats = np.zeros(num_of_electrodes)

        for electrode in range(num_of_electrodes):
            per_electrode_mode_of_beats[electrode] = len(cm_beats.dist_beats.iloc[0:, electrode].dropna())

        local_at = lat_calc_for_numba(mode_of_beats, num_of_electrodes, cm_beats_dist_beats_as_array,
            cm_beats_y_axis_as_array, per_electrode_mode_of_beats)

        local_act_time.final_dist_beat_count = []
        for beat in range(int(cm_beats.beat_count_dist_mode[0])):
            local_act_time.final_dist_beat_count.append('Beat ' + str(beat + 1))

        local_act_time.param_dist_raw = pd.DataFrame(local_at, index=local_act_time.final_dist_beat_count).T
        local_act_time.param_dist_normalized = local_act_time.param_dist_raw.sub(local_act_time.param_dist_raw.min(axis=0), axis=1)

        # Find maximum time lag, LAT version (interval)
        local_act_time.param_dist_normalized_max = local_act_time.param_dist_normalized.max().max()

        # Find the mean LAT
        local_act_time.param_dist_normalized_mean = np.nanmean(local_act_time.param_dist_normalized)
        # print("Mean LAT: " + str(local_act_time.param_dist_normalized_mean))

        local_act_time.param_dist_normalized.index = electrode_config.electrode_names
        local_act_time.param_dist_normalized.insert(0, 'Electrode', electrode_config.electrode_names)
        local_act_time.param_dist_normalized.insert(1, 'X', electrode_config.electrode_coords_x)
        local_act_time.param_dist_normalized.insert(2, 'Y', electrode_config.electrode_coords_y)

        # Assign name to resulting dataframe.
        local_act_time.param_dist_normalized.name = 'Local Activation Time'

        # Set slider value to maximum number of beats
        analysisGUI.mea_beat_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))

        print("Done")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        print(end_time - start_time)
        calculate_distances(local_act_time, electrode_config)
    except AttributeError:
        print("Please use Find Peaks first.")


@njit
def lat_calc_for_numba(mode_of_beats, num_of_electrodes, cm_beats_dist_beats_as_array, cm_beats_y_axis_as_array, per_electrode_mode_of_beats):
    temp_slope = np.zeros((num_of_electrodes, 5))
    temp_index = np.zeros((num_of_electrodes, 5))
    temp_local_at = np.zeros(num_of_electrodes)
    x_values = np.zeros((2, 5))
    y_values = np.zeros((2, 5))
    local_at = np.zeros((mode_of_beats, num_of_electrodes))

    for beat in range(mode_of_beats):
        for electrode in range(num_of_electrodes):
            if mode_of_beats == per_electrode_mode_of_beats[electrode]:
                x_2_1 = cm_beats_dist_beats_as_array[beat, electrode]
                x_2_2 = x_2_1 + 1
                x_2_3 = x_2_2 + 1
                x_2_4 = x_2_3 + 1
                x_2_5 = x_2_4 + 1

                x_1_1 = x_2_1 + 1
                x_1_2 = x_2_2 + 1
                x_1_3 = x_2_3 + 1
                x_1_4 = x_2_4 + 1
                x_1_5 = x_2_5 + 1

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

                temp_index[electrode] = x_values[0]
                temp_slope[electrode] = calc_slope
            else:
                temp_index[electrode] = [np.nan, np.nan, np.nan, np.nan, np.nan]
                temp_slope[electrode] = [np.nan, np.nan, np.nan, np.nan, np.nan]
            temp_local_at[electrode] = temp_index[electrode, np.argmin(temp_slope[electrode])]

        local_at[beat] = temp_local_at

    return local_at

# Function that calculates distances from the minimum electrode for each beat.  Values for use in conduction velocity.
def calculate_distances(local_act_time, electrode_config):
    if hasattr(local_act_time, 'distance_from_min') is True:
        print("Clearing old distance data before running new calculation...")
        delattr(local_act_time, 'distance_from_min')

    start_time = time.process_time()
    print("Calculating distances from electrode minimum, per beat.")

    calc_dist_from_min = [0]*len(local_act_time.final_dist_beat_count)

    for num, beat in enumerate(local_act_time.final_dist_beat_count):
        min_beat_location = local_act_time.param_dist_normalized[[beat]].idxmin()
        min_coords = np.array([local_act_time.param_dist_normalized.loc[min_beat_location[0], 'X'],
                              local_act_time.param_dist_normalized.loc[min_beat_location[0], 'Y']])
        calc_dist_from_min[num] = ((local_act_time.param_dist_normalized[['X', 'Y']] - min_coords).pow(2).sum(1).pow(0.5))

    local_act_time.distance_from_min = pd.DataFrame(calc_dist_from_min, index=local_act_time.final_dist_beat_count,
                                                    columns=electrode_config.electrode_names).T

    print("Done.")
    end_time = time.process_time()
    print(end_time - start_time)