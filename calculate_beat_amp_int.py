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
from numba import njit

# Obtain beat amplitudes using indices from pace_maker.raw data, values from
# cm_beats.y_axis, store in variable beat_amp_int
# cm_beats.y_axis format: columns = electrodes, rows = voltages
def calculate_beat_amp(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    try:
        if hasattr(beat_amp_int, 'beat_amp') is True:
            print("Deleting previous data.")
            delattr(beat_amp_int, 'beat_amp')
            delattr(beat_amp_int, 'beat_interval')
            delattr(beat_amp_int, 'mean_beat_int')

        print("Calculating beat amplitude and interval.")

        # Find indices of electrodes with NaN values.
        nan_electrodes_idx = np.where(
            pace_maker.param_dist_raw['Beat 1'].isna())[0]
        # Remove electrodes with NaN values
        x_elec = np.delete(
            electrode_config.electrode_coords_x, nan_electrodes_idx)
        y_elec = np.delete(
            electrode_config.electrode_coords_y, nan_electrodes_idx)
        # Generate 2xN matrix, where N = number of non-NaN electrodes, 
        # of elec. coords.
        elec_nan_removed = np.array([x_elec, y_elec])
            
        # Generate new list with the electrode names with NaN values removed.
        elec_to_remove = [
            electrode_config.electrode_names[i] for i in nan_electrodes_idx]
        elec_removed_names = [
            i for i in electrode_config.electrode_names if i not in elec_to_remove]

        # Format data for reconciling negative beat amplitude maxima.
        avg_raw = pace_maker.param_dist_raw.mean()
        raw_neg_amps = cm_beats.negative_dist_beats.transpose()
        avg_raw_labels = avg_raw.index.to_numpy(dtype=np.str_)
        avg_raw_values = avg_raw.values
        neg_amps_values = raw_neg_amps.values
        num_beats = len(avg_raw_labels)

        # Direct data to Numba-powered function to reconcile beat indices
        # and find the true indices associated with beat minima, for every
        # beat and every electrode.
        neg_amps_matrix = determine_negative_amps(
            num_beats, avg_raw_values, neg_amps_values)

        neg_amps_df = pd.DataFrame(
            data=neg_amps_matrix, 
            index=electrode_config.electrode_names, 
            columns=avg_raw_labels)

        pos_amps_array = np.zeros((int(cm_beats.beat_count_dist_mode[0]), 
            int(len(elec_nan_removed[0]))))
        temp_pos_amps = np.zeros(int(len(elec_nan_removed[0])))

        neg_amps_array = np.zeros((int(cm_beats.beat_count_dist_mode[0]), 
            int(len(elec_nan_removed[0]))))
        temp_neg_amps = np.zeros(int(len(elec_nan_removed[0])))

        for num, beat in enumerate(pace_maker.param_dist_raw):
            for num2, elec in enumerate(elec_removed_names):
                temp_idx = int(pace_maker.param_dist_raw.loc[elec, beat])
                temp_pos_amps[num2] = cm_beats.y_axis.loc[temp_idx, elec]

                temp_idx2 = int(neg_amps_df.loc[elec, beat])
                temp_neg_amps[num2] = abs(cm_beats.y_axis.loc[temp_idx2, elec])
            
            pos_amps_array[num] = temp_pos_amps.T
            neg_amps_array[num] = temp_neg_amps.T
        
        # Calculate full beat amplitude using positive and negative amps.
        # Negative abs using the absolute value of the negative signal.
        full_amp = pos_amps_array + neg_amps_array

        print(pos_amps_array)
        print(neg_amps_array)

        beat_amp_int.beat_amp = pd.DataFrame(full_amp)
        beat_amp_int.beat_amp.columns = elec_removed_names
        beat_amp_int.beat_amp.index = pace_maker.param_dist_raw.columns
        
        # Fill in the void of omitted electrodes with NaN values.
        missing_elec_fill = [np.nan] * int(cm_beats.beat_count_dist_mode[0])
        for missing in nan_electrodes_idx:
            nan_elec = electrode_config.electrode_names[missing]
            beat_amp_int.beat_amp.insert(int(missing), nan_elec, 
                missing_elec_fill)

        beat_amp_int.beat_amp = beat_amp_int.beat_amp.T
        beat_amp_int.beat_amp.insert(
            0, 'Electrode', electrode_config.electrode_names)
        beat_amp_int.beat_amp.insert(
            1, 'X', electrode_config.electrode_coords_x)
        beat_amp_int.beat_amp.insert(
            2, 'Y', electrode_config.electrode_coords_y)
        
        calculate_beat_interval(beat_amp_int, pace_maker)
        calculate_delta_amp(beat_amp_int)

        print("Done.")
    except KeyError:
        print("Please find beats before performing other calculations.")


def calculate_beat_interval(beat_amp_int, pace_maker):
    # Using pace_maker.param_dist_raw data, calculate the time between each beat.
    raw_beat_time = pace_maker.param_dist_raw.dropna()
    rbt_start_removed = raw_beat_time.iloc[0:, 1:]
    rbt_end_removed = raw_beat_time.iloc[0:, :-1]
    raw_beat_interval_array = rbt_start_removed.values - rbt_end_removed.values
    beat_amp_int.raw_beat_interval = pd.DataFrame(
        raw_beat_interval_array,
        index = rbt_start_removed.index,
        columns = rbt_end_removed.columns)

    mean_beat_time = pace_maker.param_dist_raw.mean(axis=0, skipna=True)
    mbt_start_removed = mean_beat_time.iloc[1:]
    mbt_end_removed = mean_beat_time.iloc[:-1]
    beat_amp_int.beat_interval = mbt_start_removed.values - mbt_end_removed.values
    beat_amp_int.mean_beat_int = np.nanmean(beat_amp_int.beat_interval)
    beat_amp_int.beat_interval = pd.Series(beat_amp_int.beat_interval,
        index=rbt_end_removed.columns)
    print("Mean beat interval: " + str(beat_amp_int.mean_beat_int))
    # Calculation needs to take into account input_param.sample_frequency


# Reconciles between detected beats in the positive amplitude and some of the 
# possible misnomer beats in the negative amplitude. Uses numba for speed.
@njit
def determine_negative_amps(num_beats, avg_raw_values, neg_amps_values):
    neg_amps_shape = np.shape(neg_amps_values)
    real_neg_amps = np.zeros((neg_amps_shape[0], num_beats), dtype=np.float64)

    for beat in range(num_beats):
        for row in range(neg_amps_shape[0]):
            for column in range(neg_amps_shape[1]):
                if (
                neg_amps_values[row, column] < (avg_raw_values[beat] + 50)) and ( 
                neg_amps_values[row, column] > (avg_raw_values[beat] - 50)):
                    real_neg_amps[row, beat] = neg_amps_values[row, column]
                    break
                else:
                    continue
    
    return real_neg_amps


def calculate_delta_amp(beat_amp_int):
    raw_beat_amp = beat_amp_int.beat_amp.drop(
        columns=['Electrode', 'X', 'Y']).dropna()
    rba_start_removed = raw_beat_amp.iloc[0:, 1:]
    rba_end_removed = raw_beat_amp.iloc[0:, :-1]
    raw_delta_beat_amp_array = rba_start_removed.values - rba_end_removed.values
    beat_amp_int.raw_delta_beat_amp = pd.DataFrame(
        raw_delta_beat_amp_array,
        index = rba_start_removed.index,
        columns = rba_end_removed.columns)
    
    mean_beat_amp = beat_amp_int.beat_amp.drop(
        columns=['Electrode', 'X', 'Y']).mean(axis=0, skipna=True)
    mba_start_removed = mean_beat_amp.iloc[1:]
    mba_end_removed = mean_beat_amp.iloc[:-1]
    beat_amp_int.delta_beat_amp = ((mba_start_removed.values - 
        mba_end_removed.values) / mba_end_removed.values) * 100
    beat_amp_int.delta_beat_amp = pd.Series(beat_amp_int.delta_beat_amp,
        index=rba_end_removed.columns)


def beat_amp_interval_graph(analysisGUI, electrode_config, beat_amp_int, 
pace_maker, local_act_time, input_param):
    analysisGUI.ampIntWindow.paramPlot.axis1.cla()
    analysisGUI.ampIntWindow.paramPlot.axis2.cla()
    analysisGUI.ampIntWindow.paramPlot.axis3.cla()
    analysisGUI.ampIntWindow.paramPlot.axis4.cla()

    if analysisGUI.ampIntWindow.startBeat.count() != len(
    local_act_time.final_dist_beat_count):
        analysisGUI.ampIntWindow.startBeat.clear()
        analysisGUI.ampIntWindow.endBeat.clear()

    if analysisGUI.ampIntWindow.startBeat.count() < 2:
        analysisGUI.ampIntWindow.startBeat.addItems(
            local_act_time.final_dist_beat_count)
        analysisGUI.ampIntWindow.endBeat.addItems(
            local_act_time.final_dist_beat_count)

    start_beat = analysisGUI.ampIntWindow.startBeat.currentText()
    end_beat = analysisGUI.ampIntWindow.endBeat.currentText()

    start_beat_idx = local_act_time.final_dist_beat_count.index(start_beat)
    end_beat_idx = local_act_time.final_dist_beat_count.index(end_beat)

    if hasattr(beat_amp_int, 'amp_cbar') is True:
        beat_amp_int.amp_cbar.remove()
        delattr(beat_amp_int, 'amp_cbar')

    if hasattr(beat_amp_int, 'int_plot') is True:
        beat_amp_int.int_plot.remove()
        delattr(beat_amp_int, 'int_plot')

    input_param.beat_amp_int_slider = analysisGUI.ampIntWindow.paramSlider.value()
    curr_beat = local_act_time.final_dist_beat_count[
        input_param.beat_amp_int_slider]
    
    # Heatmap for beat amplitude across all electrodes, per beat.
    # electrode_names = beat_amp_int.beat_amp.pivot(index='Y', 
    #     columns='X', values='Electrode')
    # heatmap_pivot_table = beat_amp_int.beat_amp.pivot(index='Y', 
    #     columns='X', values=curr_beat)
    # beat_amp_temp = sns.heatmap(heatmap_pivot_table, cmap="jet", 
    #     annot=electrode_names, fmt="", 
    #     ax=analysisGUI.ampIntWindow.paramPlot.axis1, cbar=False)
    # mappable = beat_amp_temp.get_children()[0]
    # beat_amp_int.amp_cbar = analysisGUI.ampIntWindow.paramPlot.axis1.figure.colorbar(mappable, 
    #     ax=analysisGUI.ampIntWindow.paramPlot.axis1)
    # beat_amp_int.amp_cbar.ax.set_title("μV", fontsize=10)
    # analysisGUI.ampIntWindow.paramPlot.axis1.set(
    #     title="Beat Amplitude, " + str(curr_beat))

    analysisGUI.ampIntWindow.paramPlot.axis1.scatter(
        np.arange(1, (len(beat_amp_int.delta_beat_amp) +1)), 
        beat_amp_int.delta_beat_amp, color="tab:red")
    analysisGUI.ampIntWindow.paramPlot.axis1.set(
        title="ΔBeat Amp vs Beat Pair", 
        xlabel="Beat Pair")

    # Plot delta beat amplitude & beat interval across dataset.
    color = "tab:red"
    analysisGUI.ampIntWindow.paramPlot.axis2.scatter(
        np.arange(1, (len(beat_amp_int.delta_beat_amp) +1)), 
        beat_amp_int.delta_beat_amp, color=color)
    analysisGUI.ampIntWindow.paramPlot.axis2.set(
        title="ΔBeat Amp & Beat interval", 
        xlabel="Beat Pair")
    analysisGUI.ampIntWindow.paramPlot.axis2.set_ylabel(
        "ΔBeat Amp (Percent Diff.)", color=color)
    analysisGUI.ampIntWindow.paramPlot.axis2.tick_params(
        axis='y', labelcolor=color)

    beat_amp_int.int_plot = analysisGUI.ampIntWindow.paramPlot.axis2.twinx()
    color = "tab:blue"
    beat_amp_int.int_plot.scatter(np.arange(1, 
        (len(beat_amp_int.beat_interval) +1)), 
        beat_amp_int.beat_interval, color=color)
    beat_amp_int.int_plot.set_ylabel("Beat Interval (ms)", color=color)
    beat_amp_int.int_plot.tick_params(axis='y', labelcolor=color)

    # # Scatter plot of maximum time lag for each beat.
    # analysisGUI.ampIntWindow.paramPlot.axis3.scatter(np.arange(1, 
    #     (len(pace_maker.param_dist_normalized_per_beat_max) +1)),
    #     pace_maker.param_dist_normalized_per_beat_max)
    # analysisGUI.ampIntWindow.paramPlot.axis3.set(
    #     title="Maximum Observed Time Lag vs Beat", 
    #     xlabel="Beat", ylabel="Time Lag (ms)")

    # Scatter plot of delta beat amp vs beat interval
    analysisGUI.ampIntWindow.paramPlot.axis3.scatter(
        beat_amp_int.beat_interval, beat_amp_int.delta_beat_amp)
    analysisGUI.ampIntWindow.paramPlot.axis3.set(
        title="ΔBeat Amp (Percent Diff.) vs Beat Interval", 
        xlabel="Interval (ms)", ylabel="ΔBeat Amp (Percent Diff.)")

    # Boxplot of beat amp.
    beats_selected = beat_amp_int.beat_amp.columns[
        start_beat_idx+3:end_beat_idx+4]
    analysisGUI.ampIntWindow.paramPlot.axis4.boxplot(
        beat_amp_int.beat_amp[beats_selected].dropna(),
        vert=True, patch_artist=True)
    analysisGUI.ampIntWindow.paramPlot.axis4.set(
        title="Beat Amplitude Boxplot", 
        ylabel="Amplitude (μV)")
    analysisGUI.ampIntWindow.paramPlot.axis4.set_xticklabels(
        labels=beats_selected, rotation = 45)

    analysisGUI.ampIntWindow.paramPlot.fig.tight_layout()
    analysisGUI.ampIntWindow.paramPlot.draw()
