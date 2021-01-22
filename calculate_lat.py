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
        sample_frequency = input_param.sample_frequency

        for electrode in range(num_of_electrodes):
            per_electrode_mode_of_beats[electrode] = len(cm_beats.dist_beats.iloc[0:, electrode].dropna())

        local_at = lat_calc_for_numba(mode_of_beats, num_of_electrodes, 
            cm_beats_dist_beats_as_array, cm_beats_y_axis_as_array, 
            per_electrode_mode_of_beats, sample_frequency)

        local_act_time.final_dist_beat_count = []
        for beat in range(int(cm_beats.beat_count_dist_mode[0])):
            local_act_time.final_dist_beat_count.append('Beat ' + str(beat + 1))

        # Consolidate raw data into local_act_time.
        local_act_time.param_dist_raw = pd.DataFrame(local_at, index=local_act_time.final_dist_beat_count).T
        
        # Normalize LAT.
        if 1000 / input_param.sample_frequency == 1.0:
            local_act_time.param_dist_normalized = local_act_time.param_dist_raw.sub(
                local_act_time.param_dist_raw.min(axis=0), axis=1)
        elif 1000 / input_param.sample_frequency == 0.1:
            local_act_time.param_dist_normalized = local_act_time.param_dist_raw.sub(
                local_act_time.param_dist_raw.min(axis=0), axis=1).div(10)

        # Find maximum time lag, LAT version (interval)
        local_act_time.param_dist_normalized_max = local_act_time.param_dist_normalized.max().max()

        # Find the mean LAT
        local_act_time.param_dist_normalized_mean = np.nanmean(
            local_act_time.param_dist_normalized.max())
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
def lat_calc_for_numba(mode_of_beats, num_of_electrodes, 
cm_beats_dist_beats_as_array, cm_beats_y_axis_as_array, 
per_electrode_mode_of_beats, sample_frequency):
    temp_local_at = np.zeros(num_of_electrodes)
    local_at = np.zeros((mode_of_beats, num_of_electrodes))

    
    if 1000 / sample_frequency == 1.0:
        end_point = 5
        temp_slope = np.zeros((num_of_electrodes, end_point))
        temp_index = np.zeros((num_of_electrodes, end_point))
        x_values = np.zeros((2, end_point))
        y_values = np.zeros((2, end_point))
        for beat in range(mode_of_beats):
            for electrode in range(num_of_electrodes):
                if mode_of_beats == per_electrode_mode_of_beats[electrode]:
                    x_values[0, 0] = cm_beats_dist_beats_as_array[beat, electrode]
                    x_values[1, 0] = x_values[0, 0] + 1
                    y_values[0, 0] = cm_beats_y_axis_as_array[
                        int(x_values[0, 0]), electrode]
                    y_values[1, 0] = cm_beats_y_axis_as_array[
                        int(x_values[1, 0]), electrode]
                    for index in range(1, end_point):
                        x_values[0, index] = x_values[0, 0] + index
                        x_values[1, index] = x_values[1, 0] + index
                        
                        y_values[0, index] = cm_beats_y_axis_as_array[
                            int(x_values[0, index]), electrode]
                        y_values[1, index] = cm_beats_y_axis_as_array[
                            int(x_values[1, index]), electrode]

                    y_diff = y_values[0, :] - y_values[1, :]
                    x_diff = x_values[0, :] - x_values[1, :]

                    calc_slope = np.divide(y_diff, x_diff)

                    temp_index[electrode] = x_values[0]
                    temp_slope[electrode] = calc_slope
                else:
                    temp_index[electrode] = [np.nan] * end_point
                    temp_slope[electrode] = [np.nan] * end_point
                
                temp_local_at[electrode] = temp_index[electrode, np.argmin(temp_slope[electrode])]
            
            local_at[beat] = temp_local_at
    
    elif 1000 / sample_frequency == 0.1:
        end_point = 50
        temp_slope = np.zeros((num_of_electrodes, end_point))
        temp_index = np.zeros((num_of_electrodes, end_point))
        x_values = np.zeros((2, end_point))
        y_values = np.zeros((2, end_point))
        for beat in range(mode_of_beats):
            for electrode in range(num_of_electrodes):
                if mode_of_beats == per_electrode_mode_of_beats[electrode]:
                    x_values[0, 0] = cm_beats_dist_beats_as_array[beat, electrode]
                    x_values[1, 0] = x_values[0, 0] + 1
                    y_values[0, 0] = cm_beats_y_axis_as_array[
                        int(x_values[0, 0]), electrode]
                    y_values[1, 0] = cm_beats_y_axis_as_array[
                        int(x_values[1, 0]), electrode]
                    for index in range(1, end_point):
                        x_values[0, index] = x_values[0, 0] + index
                        x_values[1, index] = x_values[1, 0] + index
                        
                        y_values[0, index] = cm_beats_y_axis_as_array[
                            int(x_values[0, index]), electrode]
                        y_values[1, index] = cm_beats_y_axis_as_array[
                            int(x_values[1, index]), electrode]
                    
                    y_diff = y_values[0, :] - y_values[1, :]
                    x_diff = x_values[0, :] - x_values[1, :]

                    calc_slope = np.divide(y_diff, x_diff)

                    temp_index[electrode] = x_values[0]
                    temp_slope[electrode] = calc_slope
                else:
                    temp_index[electrode] = [np.nan] * end_point
                    temp_slope[electrode] = [np.nan] * end_point
                
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