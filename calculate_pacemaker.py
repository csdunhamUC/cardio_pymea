# Author: Christopher S. Dunham
# 12/4/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import time
import numpy as np
import pandas as pd
from colorama import Fore
from colorama import Style
from colorama import init
from colorama import deinit
import seaborn as sns

# Comment out init() if using Pycharm on Windows.
init()

# Function that calculates the pacemaker (time lag).  Performs this calculation for all electrodes, and filters
# electrodes based on mismatched beat counts relative to the mode of the beat count.
def calculate_pacemaker(analysisGUI, cm_beats, pace_maker, heat_map, input_param, electrode_config):
    try:
        if hasattr(pace_maker, 'param_dist_raw') is True:
            print("Clearing old pacemaker data before running new calculation...")
            delattr(pace_maker, 'param_dist_raw')
            delattr(pace_maker, 'param_prom_raw')
            delattr(pace_maker, 'param_width_raw')
            delattr(pace_maker, 'param_thresh_raw')

        # Clock the time it takes to run the calculation.
        start_time = time.process_time()
        print("Calculating pacemaker intervals per beat.")

        # Establishing these attributes of the pace_maker class appropriately as DataFrames.
        pace_maker.param_dist_raw = pd.DataFrame()
        pace_maker.param_prom_raw = pd.DataFrame()
        pace_maker.param_width_raw = pd.DataFrame()
        pace_maker.param_thresh_raw = pd.DataFrame()

        # Performs PM calculation for each detection parameter (peak distance, prominence, width, threshold)
        for column in range(len(cm_beats.dist_beats.columns)):
            if cm_beats.beat_count_dist_mode[0] == len(cm_beats.dist_beats.iloc[0:, column].dropna()):
                pace_maker_dist_raw = pd.Series(cm_beats.dist_beats.iloc[0:, column].dropna(), name=column+1)
                pace_maker.param_dist_raw = pd.concat([pace_maker.param_dist_raw, pace_maker_dist_raw], axis='columns')
            else:
                pace_maker_dist_raw = pd.Series(name=column+1, dtype='float64')
                pace_maker.param_dist_raw = pd.concat([pace_maker.param_dist_raw, pace_maker_dist_raw], axis='columns')

        for column in range(len(cm_beats.prom_beats.columns)):
            if cm_beats.beat_count_prom_mode[0] == len(cm_beats.prom_beats.iloc[0:, column].dropna()):
                pace_maker_prom_raw = pd.Series(cm_beats.prom_beats.iloc[0:, column].dropna(), name=column+1)
                pace_maker.param_prom_raw = pd.concat([pace_maker.param_prom_raw, pace_maker_prom_raw], axis='columns')
            else:
                pace_maker_prom_raw = pd.Series(name=column+1, dtype='float64')
                pace_maker.param_prom_raw = pd.concat([pace_maker.param_prom_raw, pace_maker_prom_raw], axis='columns')

        for column in range(len(cm_beats.width_beats.columns)):
            if cm_beats.beat_count_prom_mode[0] == len(cm_beats.width_beats.iloc[0:, column].dropna()):
                pace_maker_width_raw = pd.Series(cm_beats.width_beats.iloc[0:, column].dropna(), name=column+1)
                pace_maker.param_width_raw = pd.concat([pace_maker.param_width_raw, pace_maker_width_raw], axis='columns')
            else:
                pace_maker_width_raw = pd.Series(name=column+1, dtype='float64')
                pace_maker.param_width_raw = pd.concat([pace_maker.param_width_raw, pace_maker_width_raw], axis='columns')

        for column in range(len(cm_beats.thresh_beats.columns)):
            if cm_beats.beat_count_thresh_mode[0] == len(cm_beats.thresh_beats.iloc[0:, column].dropna()):
                pace_maker_thresh_raw = pd.Series(cm_beats.thresh_beats.iloc[0:, column].dropna(), name=column+1)
                pace_maker.param_thresh_raw = pd.concat([pace_maker.param_thresh_raw, pace_maker_thresh_raw], axis='columns')
            else:
                pace_maker_thresh_raw = pd.Series(name=column+1, dtype='float64')
                pace_maker.param_thresh_raw = pd.concat([pace_maker.param_thresh_raw, pace_maker_thresh_raw], axis='columns')

        # Normalizes the values for each beat by subtracting the minimum time of a given beat from all other electrodes
        if 1000 / input_param.sample_frequency == 1.0:
            pace_maker.param_dist_normalized = pace_maker.param_dist_raw.sub(pace_maker.param_dist_raw.min(axis=1), axis=0)
            pace_maker.param_prom_normalized = pace_maker.param_prom_raw.sub(pace_maker.param_prom_raw.min(axis=1), axis=0)
            pace_maker.param_width_normalized = pace_maker.param_width_raw.sub(pace_maker.param_width_raw.min(axis=1), axis=0)
            pace_maker.param_thresh_normalized = pace_maker.param_thresh_raw.sub(pace_maker.param_thresh_raw.min(axis=1), axis=0)
        elif 1000 / input_param.sample_frequency == 0.1:
            pace_maker.param_dist_normalized = pace_maker.param_dist_raw.sub(pace_maker.param_dist_raw.min(axis=1), axis=0).div(10)
            pace_maker.param_prom_normalized = pace_maker.param_prom_raw.sub(pace_maker.param_prom_raw.min(axis=1), axis=0).div(10)
            pace_maker.param_width_normalized = pace_maker.param_width_raw.sub(pace_maker.param_width_raw.min(axis=1), axis=0).div(10)
            pace_maker.param_thresh_normalized = pace_maker.param_thresh_raw.sub(pace_maker.param_thresh_raw.min(axis=1), axis=0).div(10)

        # Set slider value to maximum number of beats
        analysisGUI.mea_beat_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))

        # Find the number of excluded electrodes (removed for noise, etc)
        excluded_elec = np.count_nonzero(pace_maker.param_dist_normalized.count() == 0)
        print("{}Excluded electrodes: {} {}".format(Fore.YELLOW, excluded_elec,
            Style.RESET_ALL))

        # Assigns column headers (names) using the naming convention provided in the electrode_config class.
        pace_maker.param_dist_normalized.columns = electrode_config.electrode_names
        pace_maker.param_prom_normalized.columns = electrode_config.electrode_names
        pace_maker.param_width_normalized.columns = electrode_config.electrode_names
        pace_maker.param_thresh_normalized.columns = electrode_config.electrode_names

        pace_maker.param_dist_raw.columns = electrode_config.electrode_names

        # Generate index (row) labels, as a list, in order to access chosen beat heatmaps in subsequent function.
        pace_maker.final_dist_beat_count = []
        for beat in range(int(cm_beats.beat_count_dist_mode[0])):
            pace_maker.final_dist_beat_count.append('Beat ' + str(beat+1))

        # Generate index (row) labels, as a list, for assignment to dataframe, prior to transpose.
        dist_new_index = []
        for row in pace_maker.param_dist_normalized.index:
            dist_new_index.append('Beat ' + str(row+1))

        prom_new_index = []
        for row in pace_maker.param_prom_normalized.index:
            prom_new_index.append('Beat ' + str(row+1))

        width_new_index = []
        for row in pace_maker.param_width_normalized.index:
            width_new_index.append('Beat ' + str(row+1))

        thresh_new_index = []
        for row in pace_maker.param_thresh_normalized.index:
            thresh_new_index.append('Beat ' + str(row+1))

        # Adds beat number labeling to each row, pre-transpose.
        pace_maker.param_dist_normalized.index = dist_new_index
        pace_maker.param_prom_normalized.index = prom_new_index
        pace_maker.param_width_normalized.index = width_new_index
        pace_maker.param_thresh_normalized.index = thresh_new_index

        pace_maker.param_dist_raw.index = dist_new_index

        # Transpose dataframe to make future plotting easier.  Makes rows = electrodes and columns = beat number.
        pace_maker.param_dist_normalized = pace_maker.param_dist_normalized.transpose()
        pace_maker.param_prom_normalized = pace_maker.param_prom_normalized.transpose()
        pace_maker.param_width_normalized = pace_maker.param_width_normalized.transpose()
        pace_maker.param_thresh_normalized = pace_maker.param_thresh_normalized.transpose()

        pace_maker.param_dist_raw = pace_maker.param_dist_raw.transpose()

        # Find the maximum time lag per beat.
        pace_maker.param_dist_normalized_per_beat_max = pace_maker.param_dist_normalized.max()

        # Find maximum time lag (interval) in data set
        pace_maker.param_dist_normalized_max = pace_maker.param_dist_normalized.max().max()

        # Find the mean interval time.
        pace_maker.param_dist_normalized_mean = np.nanmean(pace_maker.param_dist_normalized.max())

        # Insert electrode name as column to make future plotting easier when attempting to use pivot table.
        pace_maker.param_dist_normalized.insert(0, 'Electrode', electrode_config.electrode_names)
        pace_maker.param_prom_normalized.insert(0, 'Electrode', electrode_config.electrode_names)
        pace_maker.param_width_normalized.insert(0, 'Electrode', electrode_config.electrode_names)
        pace_maker.param_thresh_normalized.insert(0, 'Electrode', electrode_config.electrode_names)

        # Insert electrode coordinates X,Y (in micrometers) as columns after electrode identifier.
        pace_maker.param_dist_normalized.insert(1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_dist_normalized.insert(2, 'Y', electrode_config.electrode_coords_y)
        # Repeat for prominence parameter.
        pace_maker.param_prom_normalized.insert(1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_prom_normalized.insert(2, 'Y', electrode_config.electrode_coords_y)
        # Repeat for width parameter.
        pace_maker.param_width_normalized.insert(1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_width_normalized.insert(2, 'Y', electrode_config.electrode_coords_y)
        # Repeat for threshold parameter.
        pace_maker.param_thresh_normalized.insert(1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_thresh_normalized.insert(2, 'Y', electrode_config.electrode_coords_y)

        pace_maker.param_dist_normalized.name = 'Pacemaker (Normalized)'

        print("Done.")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        # print(end_time - start_time)
        print("{}Maximum time lag in data set: {} {}".format(
            Fore.MAGENTA, pace_maker.param_dist_normalized_max, 
            Style.RESET_ALL))
        # print(f"{Fore.MAGENTA}Maximum time lag in data set: {pace_maker.param_dist_normalized_max}{Style.RESET_ALL}")
        deinit()
    except AttributeError:
        print("Please use Find Peaks first.")


def graph_pacemaker(analysisGUI, heat_map, pace_maker, input_param):
    try:
        if hasattr(heat_map, 'pm_solo_cbar') is True:
            heat_map.pm_solo_cbar.remove()
            delattr(heat_map, 'pm_solo_cbar')

        heat_map.pm_solo_axis.cla()
        input_param.pm_solo_beat_choice = int(analysisGUI.pm_solo_beat_select.get()) - 1

        electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table = pace_maker.param_dist_normalized.pivot(index='Y', 
            columns='X', values=pace_maker.final_dist_beat_count[input_param.pm_solo_beat_choice])

        heat_map.pm_solo_temp = sns.heatmap(heatmap_pivot_table, cmap="jet", 
            annot=electrode_names, fmt="", ax=heat_map.pm_solo_axis, vmin=0, 
            vmax=pace_maker.param_dist_normalized_max, cbar=False)
        mappable = heat_map.pm_solo_temp.get_children()[0]
        heat_map.pm_solo_cbar = heat_map.pm_solo_axis.figure.colorbar(mappable, 
            ax=heat_map.pm_solo_axis)
        heat_map.pm_solo_cbar.ax.set_title("Time Lag (ms)", fontsize=10)

        heat_map.pm_solo_axis.set(title="Pacemaker, Beat " + 
            str(input_param.pm_solo_beat_choice+1), xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")
        heat_map.pm_solo_plot.tight_layout()
        heat_map.pm_solo_plot.canvas.draw()

    except AttributeError:
        print("Please calculate PM first.")
    except IndexError:
        print("You entered a beat that does not exist.")