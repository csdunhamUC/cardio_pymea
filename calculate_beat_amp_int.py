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
heat_map, input_param, electrode_config):
    print()
    # Obtain beat amplitudes using indices from pace_maker.raw data, values from
    # cm_beats.y_axis, store in variable beat_amp_int
    # cm_beats.y_axis format: columns = electrodes, rows = voltages

    # Find indices of electrodes with NaN values.
    nan_electrodes_idx = np.where(pace_maker.param_dist_raw['Beat 1'].isna())[0]
    # Remove electrodes with NaN values for fitting modules (which cannot 
    # handle NaN values)
    x_elec = np.delete(electrode_config.electrode_coords_x, nan_electrodes_idx)
    y_elec = np.delete(electrode_config.electrode_coords_y, nan_electrodes_idx)
    # Generate 2xN, where N = number of non-NaN electrodes, of elec. coords.
    elec_nan_removed = np.array([x_elec, y_elec])
        
    # Generate new list with the electrode names with NaN values removed.
    elec_to_remove = [electrode_config.electrode_names[i] for i in nan_electrodes_idx]
    elec_removed_names = [
        i for i in electrode_config.electrode_names if i not in elec_to_remove]

    amps_array = np.zeros((int(cm_beats.beat_count_dist_mode[0]), 
        int(len(elec_nan_removed[0]))))
    temp_amps = np.zeros(int(len(elec_nan_removed[0])))

    for num, beat in enumerate(pace_maker.param_dist_raw):
        for elec in range(len(elec_nan_removed[0])):
            temp_idx = int(pace_maker.param_dist_raw.loc[elec_removed_names[elec], beat])
            print(temp_idx)
            # Yields the wrong values from cm_beats.y_axis because elec doesn't
            # exclude the bad channels.  Need to rework cm_beats.y_axis to have
            # electrode names for labels.  Requires updating plotting functions
            # in determine_beats as well, in order to do well.
            temp_amps[elec] = cm_beats.y_axis.iloc[temp_idx, elec]
            print(temp_amps[elec])
        
        amps_array[num] = temp_amps.T


    beat_amp_int.beat_amp = pd.DataFrame(amps_array.T)


def calculate_beat_interval():
    print()
    # Using pace_maker.raw data, calculate the time between each beat.
    # Calculation needs to take into account input_param.sample_frequency


def beat_amp_interval_graph(beat_amp_int):
    print()
    # Heatmap for beat amplitude across all electrodes, per beat.
    # Statistical plot for beat amplitude vs distance, per beat.

    beat_amp_int.amp_int_plot.tight_layout()
    beat_amp_int.amp_int_plot.canvas.draw()