# Author: Christopher S. Dunham
# 12/4/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import time
import pandas as pd

# Function that calculates the pacemaker (time lag).  Performs this calculation for all electrodes, and filters
# electrodes based on mismatched beat counts relative to the mode of the beat count.
def calculate_pacemaker(elecGUI120, cm_beats, pace_maker, heat_map, input_param, ElectrodeConfig):
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
        pace_maker.param_dist_normalized = pace_maker.param_dist_raw.sub(pace_maker.param_dist_raw.min(axis=1), axis=0)
        pace_maker.param_prom_normalized = pace_maker.param_prom_raw.sub(pace_maker.param_prom_raw.min(axis=1), axis=0)
        pace_maker.param_width_normalized = pace_maker.param_width_raw.sub(pace_maker.param_width_raw.min(axis=1), axis=0)
        pace_maker.param_thresh_normalized = pace_maker.param_thresh_raw.sub(pace_maker.param_thresh_raw.min(axis=1), axis=0)

        # Find maximum time lag (interval)
        pace_maker.param_dist_normalized_max = pace_maker.param_dist_normalized.max().max()

        # Set slider value to maximum number of beats
        elecGUI120.mea_beat_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))

        # Assigns column headers (names) using the naming convention provided in the ElectrodeConfig class.
        pace_maker.param_dist_normalized.columns = ElectrodeConfig.electrode_names
        pace_maker.param_prom_normalized.columns = ElectrodeConfig.electrode_names
        pace_maker.param_width_normalized.columns = ElectrodeConfig.electrode_names
        pace_maker.param_thresh_normalized.columns = ElectrodeConfig.electrode_names

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

        # Transpose dataframe to make future plotting easier.  Makes rows = electrodes and columns = beat number.
        pace_maker.param_dist_normalized = pace_maker.param_dist_normalized.transpose()
        pace_maker.param_prom_normalized = pace_maker.param_prom_normalized.transpose()
        pace_maker.param_width_normalized = pace_maker.param_width_normalized.transpose()
        pace_maker.param_thresh_normalized = pace_maker.param_thresh_normalized.transpose()

        # Insert electrode name as column to make future plotting easier when attempting to use pivot table.
        pace_maker.param_dist_normalized.insert(0, 'Electrode', ElectrodeConfig.electrode_names)
        pace_maker.param_prom_normalized.insert(0, 'Electrode', ElectrodeConfig.electrode_names)
        pace_maker.param_width_normalized.insert(0, 'Electrode', ElectrodeConfig.electrode_names)
        pace_maker.param_thresh_normalized.insert(0, 'Electrode', ElectrodeConfig.electrode_names)

        # Insert electrode coordinates X,Y (in micrometers) as columns after electrode identifier.
        pace_maker.param_dist_normalized.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        pace_maker.param_dist_normalized.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)
        # Repeat for prominence parameter.
        pace_maker.param_prom_normalized.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        pace_maker.param_prom_normalized.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)
        # Repeat for width parameter.
        pace_maker.param_width_normalized.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        pace_maker.param_width_normalized.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)
        # Repeat for threshold parameter.
        pace_maker.param_thresh_normalized.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        pace_maker.param_thresh_normalized.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)

        pace_maker.param_dist_normalized.name = 'Pacemaker (Normalized)'

        print("Done.")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        print(end_time - start_time)
    except AttributeError:
        print("Please use Find Peaks first.")