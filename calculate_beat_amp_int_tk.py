# Author: Christopher S. Dunham
# 02/06/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from scipy.optimize import curve_fit


# Obtain beat amplitudes using indices from pace_maker.raw data, values from
# cm_beats.y_axis, store in variable beat_amp_int
# cm_beats.y_axis format: columns = electrodes, rows = voltages
def calculate_beat_amp(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    
    if hasattr(beat_amp_int, 'beat_amp') is True:
        print("Deleting previous data.")
        delattr(beat_amp_int, 'beat_amp')
        delattr(beat_amp_int, 'beat_interval')
        delattr(beat_amp_int, 'mean_beat_int')

    print("Calculating beat amplitude and interval.")

    # Find indices of electrodes with NaN values.
    nan_electrodes_idx = np.where(pace_maker.param_dist_raw['Beat 1'].isna())[0]
    # Remove electrodes with NaN values
    x_elec = np.delete(electrode_config.electrode_coords_x, nan_electrodes_idx)
    y_elec = np.delete(electrode_config.electrode_coords_y, nan_electrodes_idx)
    # Generate 2xN matrix, where N = number of non-NaN electrodes, 
    # of elec. coords.
    elec_nan_removed = np.array([x_elec, y_elec])
        
    # Generate new list with the electrode names with NaN values removed.
    elec_to_remove = [electrode_config.electrode_names[i] for i in nan_electrodes_idx]
    elec_removed_names = [
        i for i in electrode_config.electrode_names if i not in elec_to_remove]

    amps_array = np.zeros((int(cm_beats.beat_count_dist_mode[0]), 
        int(len(elec_nan_removed[0]))))
    temp_amps = np.zeros(int(len(elec_nan_removed[0])))

    for num, beat in enumerate(pace_maker.param_dist_raw):
        for num2, elec in enumerate(elec_removed_names):
            temp_idx = int(pace_maker.param_dist_raw.loc[elec, beat])
            temp_amps[num2] = cm_beats.y_axis.loc[temp_idx, elec]
        
        amps_array[num] = temp_amps.T

    beat_amp_int.beat_amp = pd.DataFrame(amps_array)
    beat_amp_int.beat_amp.columns = elec_removed_names
    beat_amp_int.beat_amp.index = pace_maker.param_dist_raw.columns

    # Fill in the void of omitted electrodes with NaN values.
    missing_elec_fill = [np.nan] * int(cm_beats.beat_count_dist_mode[0])
    for missing in nan_electrodes_idx:
        nan_elec = electrode_config.electrode_names[missing]
        beat_amp_int.beat_amp.insert(int(missing), nan_elec, 
            missing_elec_fill)

    beat_amp_int.beat_amp = beat_amp_int.beat_amp.T
    beat_amp_int.beat_amp.insert(0, 'Electrode', electrode_config.electrode_names)
    beat_amp_int.beat_amp.insert(1, 'X', electrode_config.electrode_coords_x)
    beat_amp_int.beat_amp.insert(2, 'Y', electrode_config.electrode_coords_y)
    
    calculate_beat_interval(beat_amp_int, pace_maker)
    calculate_delta_amp(beat_amp_int)
    print("Done.")


def calculate_beat_interval(beat_amp_int, pace_maker):
    # Using pace_maker.param_dist_raw data, calculate the time between each beat.
    mean_beat_time = pace_maker.param_dist_raw.mean(axis=0, skipna=True)
    mbt_start_removed = mean_beat_time.iloc[1:]
    mbt_end_removed = mean_beat_time.iloc[:-1]
    beat_amp_int.beat_interval = mbt_start_removed.values - mbt_end_removed.values
    beat_amp_int.mean_beat_int = np.nanmean(beat_amp_int.beat_interval)
    print("Mean beat interval: " + str(beat_amp_int.mean_beat_int))
    # Calculation needs to take into account input_param.sample_frequency


def calculate_delta_amp(beat_amp_int):
    mean_beat_amp = beat_amp_int.beat_amp.mean(axis=0, skipna=True)
    mba_start_removed = mean_beat_amp.iloc[1:]
    mba_end_removed = mean_beat_amp.iloc[:-1]
    beat_amp_int.delta_beat_amp = mba_start_removed.values - mba_end_removed.values


def beat_amp_interval_graph(analysisGUI, electrode_config, beat_amp_int, 
pace_maker, local_act_time, input_param):
    beat_amp_int.axis1.cla()
    beat_amp_int.axis2.cla()
    beat_amp_int.axis3.cla()
    beat_amp_int.axis4.cla()

    analysisGUI.amp_int_start_beat_value['values'] = (
        local_act_time.final_dist_beat_count)
    analysisGUI.amp_int_end_beat_value['values'] = (
        local_act_time.final_dist_beat_count)

    start_beat = analysisGUI.amp_int_start_beat_value.get()
    end_beat = analysisGUI.amp_int_end_beat_value.get()

    start_beat_idx = local_act_time.final_dist_beat_count.index(start_beat)
    end_beat_idx = local_act_time.final_dist_beat_count.index(end_beat)

    if hasattr(beat_amp_int, 'amp_cbar') is True:
        beat_amp_int.amp_cbar.remove()
        delattr(beat_amp_int, 'amp_cbar')

    if hasattr(beat_amp_int, 'int_plot') is True:
        beat_amp_int.int_plot.remove()
        delattr(beat_amp_int, 'int_plot')

    input_param.beat_amp_int_slider = int(
        analysisGUI.beat_amp_beat_select.get()) - 1
    curr_beat = local_act_time.final_dist_beat_count[
        input_param.beat_amp_int_slider]
    
    # Heatmap for beat amplitude across all electrodes, per beat.
    electrode_names = beat_amp_int.beat_amp.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table = beat_amp_int.beat_amp.pivot(index='Y', 
        columns='X', values=curr_beat)
    beat_amp_int.beat_amp_temp = sns.heatmap(heatmap_pivot_table, cmap="jet", 
        annot=electrode_names, fmt="", ax=beat_amp_int.axis1, cbar=False)
    mappable = beat_amp_int.beat_amp_temp.get_children()[0]
    beat_amp_int.amp_cbar = beat_amp_int.axis1.figure.colorbar(mappable, 
        ax=beat_amp_int.axis1)
    beat_amp_int.amp_cbar.ax.set_title("μV", fontsize=10)
    beat_amp_int.axis1.set(title="Beat Amplitude, " + str(curr_beat))

    # Plot delta beat amplitude across dataset.
    color = "tab:red"
    beat_amp_int.axis2.scatter(np.arange(1, (len(beat_amp_int.delta_beat_amp) +1)), 
        beat_amp_int.delta_beat_amp, color=color)
    beat_amp_int.axis2.set(title="Delta Beat Amp & Beat interval", 
        xlabel="Beat Pair")
    beat_amp_int.axis2.set_ylabel("Voltage (μV)", color=color)
    beat_amp_int.axis2.tick_params(axis='y', labelcolor=color)

    beat_amp_int.int_plot = beat_amp_int.axis2.twinx()
    color = "tab:blue"
    beat_amp_int.int_plot.scatter(np.arange(1, (len(beat_amp_int.beat_interval) +1)), 
        beat_amp_int.beat_interval, color=color)
    beat_amp_int.int_plot.set_ylabel("Beat Interval (ms)", color=color)
    beat_amp_int.int_plot.tick_params(axis='y', labelcolor=color)

    # Plot beat intervals across dataset.
    # beat_amp_int.axis2.scatter(np.arange(1, (len(beat_amp_int.beat_interval) +1)), 
    #     beat_amp_int.beat_interval)
    # beat_amp_int.axis2.set(title="Beat Interval", xlabel="Beat Pair", 
    #     ylabel="Interval (ms)")
    
    # Statistical plot for beat amplitude vs distance, per beat.
    # This plot isn't communicating much of value... need to consider something
    # else.  Max time lag per beat, perhaps?
    beat_amp_int.axis3.scatter(np.arange(1, 
        (len(pace_maker.param_dist_normalized_per_beat_max) +1)),
        pace_maker.param_dist_normalized_per_beat_max)
    # beat_amp_int.axis3.scatter(local_act_time.distance_from_min[curr_beat],
    #     beat_amp_int.beat_amp[curr_beat])
    beat_amp_int.axis3.set(title="Maximum Observed Time Lag vs Beat", 
        xlabel="Beat", ylabel="Time Lag (ms)")
    # beat_amp_int.axis3.set(title="Beat Amplitude vs Distance", 
    #     xlabel="Distance", ylabel="Beat Amplitude (μV)")

    # Boxplot of beat amp.
    beats_selected = beat_amp_int.beat_amp.columns[start_beat_idx+3:end_beat_idx+4]
    beat_amp_int.axis4.boxplot(beat_amp_int.beat_amp[beats_selected].dropna(),
        vert=True, patch_artist=True)
    beat_amp_int.axis4.set(title="Beat Amplitude Boxplot", 
        ylabel="Amplitude (μV)")
    beat_amp_int.axis4.set_xticklabels(labels=beats_selected, rotation = 45)

    beat_amp_int.amp_int_plot.tight_layout()
    beat_amp_int.amp_int_plot.canvas.draw()