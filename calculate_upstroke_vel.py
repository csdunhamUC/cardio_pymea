# Author: Christopher S. Dunham
# Date: 12/4/2020
# Principal Investigator: James K. Gimzewski
# Organization: University of California, Los Angeles
# Department of Chemistry and Biochemistry
# Original work by CSD

import time
from numba import njit
import numpy as np
import pandas as pd
import seaborn as sns


# Calculates upstroke velocity (dV/dt)
def calculate_upstroke_vel(analysisGUI, cm_beats, upstroke_vel, heat_map, 
input_param, electrode_config):
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
        sample_frequency = input_param.sample_frequency

        for electrode in range(num_of_electrodes):
            per_electrode_mode_of_beats[electrode] = len(cm_beats.dist_beats.iloc[0:, electrode].dropna())

        dvdt_max = dvdt_calc_for_numba(mode_of_beats, num_of_electrodes, 
            cm_beats_dist_beats_as_array, cm_beats_y_axis_as_array, 
            per_electrode_mode_of_beats, sample_frequency)

        upstroke_vel.final_dist_beat_count = []
        for beat in range(int(cm_beats.beat_count_dist_mode[0])):
            upstroke_vel.final_dist_beat_count.append('Beat ' + str(beat + 1))

        upstroke_vel.param_dist_raw = pd.DataFrame(dvdt_max, index=upstroke_vel.final_dist_beat_count).T
        
        # Normalized dV/dt across the dataset.
        if 1000 / input_param.sample_frequency == 1.0:
            upstroke_vel.param_dist_normalized = upstroke_vel.param_dist_raw.sub(
                upstroke_vel.param_dist_raw.min(axis=0), axis=1)
        elif 1000 / input_param.sample_frequency == 0.1:
            upstroke_vel.param_dist_normalized = upstroke_vel.param_dist_raw.sub(
                upstroke_vel.param_dist_raw.min(axis=0), axis=1).mul(10)

        # Find maximum dV/dt.
        upstroke_vel.param_dist_normalized_max = upstroke_vel.param_dist_normalized.max().max()

        # Mean dV/dt across the dataset.
        upstroke_vel.param_dist_normalized_mean = np.nanmean(upstroke_vel.param_dist_normalized)

        upstroke_vel.param_dist_normalized.index = electrode_config.electrode_names
        upstroke_vel.param_dist_normalized.insert(0, 'Electrode', electrode_config.electrode_names)
        upstroke_vel.param_dist_normalized.insert(1, 'X', electrode_config.electrode_coords_x)
        upstroke_vel.param_dist_normalized.insert(2, 'Y', electrode_config.electrode_coords_y)

        # Assign name to resulting dataframe.
        upstroke_vel.param_dist_normalized.name = 'Upstroke Velocity'

        print("Done")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        # print(end_time - start_time)
    except AttributeError:
        print("Please use Find Peaks first.")


@njit
def dvdt_calc_for_numba(mode_of_beats, num_of_electrodes, 
cm_beats_dist_beats_as_array, cm_beats_y_axis_as_array, 
per_electrode_mode_of_beats, sample_frequency): 
    temp_dvdt_max = np.zeros(num_of_electrodes)
    dvdt_max = np.zeros((mode_of_beats, num_of_electrodes))

    # Checks to see whether sample frequency is on millisecond time scale.
    # If so, dividing it into 1000 should give 1.0, and the calculations
    # will proceed assuming millisecond scales, such that each index is some
    # millisecond.  If not and the answer is 0.1 (as 10,000Hz yields), this 
    # means that the time scale is now in tenths of milliseconds.  In order to 
    # comprise the same 5 millisecond window, 50 indices must be analyzed rather 
    # than 5.  The end_point variable dynamically re-sizes the appropriate
    # arrays in order to handle these two scenarios.
    if 1000 / sample_frequency == 1.0:
        end_point = 5
        temp_slope = np.zeros((num_of_electrodes, end_point))
        x_values = np.zeros((2, end_point))
        y_values = np.zeros((2, end_point))
        for beat in range(mode_of_beats):
            for electrode in range(num_of_electrodes):
                if mode_of_beats == per_electrode_mode_of_beats[electrode]:
                    x_values[0, 0] = cm_beats_dist_beats_as_array[beat, electrode]
                    x_values[1, 0] = x_values[0, 0] - 1
                    y_values[0, 0] = cm_beats_y_axis_as_array[
                        int(x_values[0, 0]), electrode]
                    y_values[1, 0] = cm_beats_y_axis_as_array[
                        int(x_values[1, 0]), electrode]
                    for index in range(1, end_point):
                        x_values[0, index] = x_values[0, 0] - index
                        x_values[1, index] = x_values[1, 0] - index
                        
                        y_values[0, index] = cm_beats_y_axis_as_array[
                            int(x_values[0, index]), electrode]
                        y_values[1, index] = cm_beats_y_axis_as_array[
                            int(x_values[1, index]), electrode]

                    y_diff = y_values[0, :] - y_values[1, :]
                    x_diff = x_values[0, :] - x_values[1, :]

                    calc_slope = np.divide(y_diff, x_diff)
                    temp_slope[electrode] = calc_slope
                else:
                    temp_slope[electrode] = [np.nan] * end_point
            
                temp_dvdt_max[electrode] = max(temp_slope[electrode])

            dvdt_max[beat] = temp_dvdt_max

    elif 1000 / sample_frequency == 0.1:
        end_point = 50
        temp_slope = np.zeros((num_of_electrodes, end_point))
        x_values = np.zeros((2, end_point))
        y_values = np.zeros((2, end_point))
        for beat in range(mode_of_beats):
            for electrode in range(num_of_electrodes):
                if mode_of_beats == per_electrode_mode_of_beats[electrode]:
                    x_values[0, 0] = cm_beats_dist_beats_as_array[beat, electrode]
                    x_values[1, 0] = x_values[0, 0] - 1
                    y_values[0, 0] = cm_beats_y_axis_as_array[
                        int(x_values[0, 0]), electrode]
                    y_values[1, 0] = cm_beats_y_axis_as_array[
                        int(x_values[1, 0]), electrode]
                    for index in range(1, end_point):
                        x_values[0, index] = x_values[0, 0] - index
                        x_values[1, index] = x_values[1, 0] - index
                        
                        y_values[0, index] = cm_beats_y_axis_as_array[
                            int(x_values[0, index]), electrode]
                        y_values[1, index] = cm_beats_y_axis_as_array[
                            int(x_values[1, index]), electrode]
                    
                    y_diff = y_values[0, :] - y_values[1, :]
                    x_diff = x_values[0, :] - x_values[1, :]

                    calc_slope = np.divide(y_diff, x_diff)

                    temp_slope[electrode] = calc_slope
                else:
                    temp_slope[electrode] = [np.nan] * end_point
                temp_dvdt_max[electrode] = max(temp_slope[electrode])

            dvdt_max[beat] = temp_dvdt_max

    return dvdt_max


def graph_upstroke(analysisGUI, heat_map, upstroke_vel, input_param):
    try:
        if hasattr(heat_map, 'dvdt_solo_cbar') is True:
            heat_map.dvdt_solo_cbar.remove()
            delattr(heat_map, 'dvdt_solo_cbar')

        analysisGUI.dvdtWindow.paramPlot.axis1.cla()
        input_param.dvdt_solo_beat_choice = analysisGUI.dvdtWindow.paramSlider.value()

        electrode_names_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table_2 = upstroke_vel.param_dist_normalized.pivot(
            index='Y', 
            columns='X', 
            values=upstroke_vel.final_dist_beat_count[
                input_param.dvdt_solo_beat_choice])

        dvdt_solo_temp = sns.heatmap(heatmap_pivot_table_2, cmap="jet", 
            annot=electrode_names_2, fmt="", 
            ax=analysisGUI.dvdtWindow.paramPlot.axis1, 
            vmax=upstroke_vel.param_dist_normalized_max, cbar=False)
        mappable_2 = dvdt_solo_temp.get_children()[0]
        heat_map.dvdt_solo_cbar = analysisGUI.dvdtWindow.paramPlot.axis1.figure.colorbar(
            mappable_2, 
            ax=analysisGUI.dvdtWindow.paramPlot.axis1)
        heat_map.dvdt_solo_cbar.ax.set_title("μV/ms", fontsize=10)

        analysisGUI.dvdtWindow.paramPlot.axis1.set(
            title="Upstroke Velocity, Beat " + 
            str(input_param.dvdt_solo_beat_choice+1), 
            xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")
        analysisGUI.dvdtWindow.paramPlot.fig.tight_layout()
        analysisGUI.dvdtWindow.paramPlot.draw()

    except AttributeError:
        print("Please calculate dV/dt first.")
    except IndexError:
        print("You entered a beat that does not exist.")
