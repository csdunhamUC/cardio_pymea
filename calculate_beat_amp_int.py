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


def calculate_beat_amp(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    # Obtain beat amplitudes using indices from pace_maker.raw data, values from
    # cm_beats.y_axis, store in variable beat_amp_int
    # cm_beats.y_axis format: columns = electrodes, rows = voltages

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
    
    calculate_beat_interval(beat_amp_int, pace_maker, input_param)

    beat_amp_interval_graph(analysisGUI, electrode_config, beat_amp_int, 
        local_act_time, input_param)


def calculate_beat_interval(beat_amp_int, pace_maker, input_param):
    # Using pace_maker.param_dist_raw data, calculate the time between each beat.
    mean_beat_time = pace_maker.param_dist_raw.mean(axis=0, skipna=True)
    mbt_start_removed = mean_beat_time.iloc[1:]
    mbt_end_removed = mean_beat_time.iloc[:-1]
    beat_amp_int.beat_interval = mbt_start_removed.values - mbt_end_removed.values
    beat_amp_int.mean_beat_int = np.nanmean(beat_amp_int.beat_interval)
    # Calculation needs to take into account input_param.sample_frequency


def beat_amp_interval_graph(analysisGUI, electrode_config, beat_amp_int, 
local_act_time, input_param):
    beat_amp_int.axis1.cla()
    beat_amp_int.axis2.cla()
    beat_amp_int.axis3.cla()
    beat_amp_int.axis4.cla()
    if hasattr(beat_amp_int, 'amp_cbar') is True:
        beat_amp_int.amp_cbar.remove()
        delattr(beat_amp_int, 'amp_cbar')

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
    beat_amp_int.axis1.set(title="Beat Amplitude")

    # Plot beat intervals across dataset.
    beat_amp_int.axis2.scatter(np.arange(1, (len(beat_amp_int.beat_interval) +1)), 
        beat_amp_int.beat_interval)
    beat_amp_int.axis2.set(title="Beat Interval", xlabel="Beat", 
        ylabel="Interval (ms)")
    
    # Statistical plot for beat amplitude vs distance, per beat.
    beat_amp_int.axis3.scatter(local_act_time.distance_from_min[curr_beat],
        beat_amp_int.beat_amp[curr_beat])
    beat_amp_int.axis3.set(title="Beat Amplitude vs Distance", 
        xlabel="Distance", ylabel="Beat Amplitude")

    # What will axis4 be a plot of?

    beat_amp_int.amp_int_plot.tight_layout()
    beat_amp_int.amp_int_plot.canvas.draw()