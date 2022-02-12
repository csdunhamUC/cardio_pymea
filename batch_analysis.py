# Author:
# Christopher S. Dunham
# Email:
# csdunham@chem.ucla.edu; csdunham@protonmail.com
# Principal Investigator:
# James K. Gimzewski
# Organization:
# University of California, Los Angeles
# Department of Chemistry and Biochemistry
# This is an original work, unless other noted in comments, by CSD
# Began 9/23/21

# Batch analysis software.
# Requires the use of a batch_params.xlsx file (see example from repo)
# Program will load the *.xlsx file, identify the save file path of the batch 
# files, the individual files, the parameters for each file, etc, and perform 
# batch calculations.

import sys
import os
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
import numpy as np
import determine_beats
import calculate_pacemaker
import calculate_upstroke_vel
import calculate_lat
import calculate_cv
import calculate_beat_amp_int
import detect_transloc


def import_batch_file(analysisGUI, raw_data, batch_data, electrode_config, 
cm_beats, input_param, pace_maker, heat_map, local_act_time, upstroke_vel,
conduction_vel, beat_amp_int):
    try:
        # Set batch_config status to true
        batch_data.batch_config = True
        # Open dialog to select file (*.xlsx)
        batch_filename_and_path = QFileDialog.getOpenFileName(
            analysisGUI, "Select Batch File", "/home", "Excel files (*.xlsx)")
        # Get directory, file name
        batch_dir, batch_name = os.path.split(batch_filename_and_path[0])
        # Load data from file (*.xlsx) into dataframe
        batch_df = pd.read_excel(batch_filename_and_path[0])

        print(f"Processing batch: {batch_name}")

        # Store pacemaker translocation events for datasets in batch
        batch_data.batch_translocs = []
        
        # Store dataset event times
        batch_data.batch_times = []

        # Store dataset translocation distances
        batch_data.batch_dists = []

        # Store dataset beat count for datasets in batch
        batch_data.batch_beat_counts = []

        # Store dataset beat counts, translocations as dictionary
        batch_data.beat_event_dict = {}

        # Store dataset beat times, translocations as dictionary
        batch_data.beat_time_dict = {}

        # Store dataset events, translocation distances as a dictionary
        batch_data.trans_dist_dict = {}

        total_files = len(batch_df["file_name"].values)

        for num, (file_dir, file_name, pk_height, pk_dist, samp_freq, tog_trunc, 
        trunc_start, trunc_end, tog_silence, silenced_elecs) in enumerate(zip(
        batch_df["file_dir"], batch_df["file_name"], 
        batch_df["min_pk_height"], batch_df["min_pk_dist"], 
        batch_df["sample_frequency"], batch_df["toggle_trunc"], 
        batch_df["trunc_start"], batch_df["trunc_end"], 
        batch_df["toggle_silence"], batch_df["silenced_electrodes"])):
            print("")
            print(f"Analyzing file {num+1} of {total_files}: {file_name}.")
            print("")
            file_path = "/".join([file_dir, file_name])
            raw_data.imported = pd.read_csv(
                file_path, sep="\s+", lineterminator="\n", skiprows=3,header=0, 
                encoding='iso-8859-15', skipinitialspace=True, low_memory=False)
            # temp_data = pd.read_csv(
                # file_path, sep="\s+\t", lineterminator="\n", skiprows=[0, 1, 3],
                # header=None, nrows=1, encoding='iso-8859-15', 
                # skipinitialspace=True)
        
            print(f"Silence toggled?: {tog_silence}")

            raw_data.new_data_size = np.shape(raw_data.imported)
            electrode_config.electrode_toggle(raw_data)
            input_param.min_peak_dist = pk_dist
            input_param.min_peak_height = pk_height
            input_param.parameter_prominence = 100 # Defaults from GUI
            input_param.parameter_width = 3 # Defaults from GUI
            input_param.parameter_thresh = 50 # Defaults from GUI
            input_param.sample_frequency = samp_freq
            # Assign truncation inputs
            input_param.toggle_trunc = tog_trunc
            input_param.trunc_start = trunc_start
            input_param.trunc_end = trunc_end
            # Assign silenced electrode inputs
            input_param.toggle_silence = tog_silence
            
            if input_param.toggle_silence == True:
                if ", " in silenced_elecs:
                    silenced_elecs = silenced_elecs.split(", ")
                elif "," in silenced_elecs:
                    silenced_elecs = silenced_elecs.split(",")
                if isinstance(silenced_elecs, list):
                    input_param.silenced_elecs = silenced_elecs
                else:
                    input_param.silenced_elecs = [silenced_elecs]
            # print(input_param.silenced_elecs)
            # print(type(input_param.silenced_elecs))

            # # Perform batch calculations
            determine_beats.determine_beats(analysisGUI, raw_data, cm_beats, 
                input_param, electrode_config, batch_data)
            calculate_pacemaker.calculate_pacemaker(analysisGUI, cm_beats, 
                pace_maker, heat_map, input_param, electrode_config)
            calculate_lat.calculate_lat(analysisGUI, cm_beats, local_act_time,
                heat_map, input_param, electrode_config)
            calculate_upstroke_vel.calculate_upstroke_vel(analysisGUI, cm_beats, 
                upstroke_vel, heat_map, input_param, electrode_config)
            calculate_cv.calculate_conduction_velocity(analysisGUI, cm_beats, 
                conduction_vel, local_act_time, heat_map, input_param, 
                electrode_config)
            calculate_beat_amp_int.calculate_beat_amp(analysisGUI, cm_beats, 
                beat_amp_int, pace_maker, local_act_time, heat_map, input_param, 
                electrode_config)
            detect_transloc.pm_translocations(analysisGUI, pace_maker, 
                electrode_config, beat_amp_int)
        
            # Populate list with translocations of each data set.
            temp_translocs = pace_maker.transloc_events
            for event in temp_translocs:
                batch_data.batch_translocs.append(event)
            
            # Populate list with event times for each data set.
            temp_times = pace_maker.transloc_times
            for time in temp_times:
                batch_data.batch_times.append(time)

            # Populate list with translocation distances for each data set.
            temp_dists = pace_maker.transloc_dist
            for dist in temp_dists:
                batch_data.batch_dists.append(dist)

            # Populate list with beat count of each data set.
            batch_data.batch_beat_counts.append(pace_maker.number_beats)

            # Dictionary to store number of beats, events for normalization
            if pace_maker.number_beats is not None:
                dict_key = f"Key_{num+1}_{pace_maker.number_beats}"
                batch_data.beat_event_dict[dict_key] = temp_translocs
                batch_data.beat_time_dict[dict_key] = temp_times
                batch_data.trans_dist_dict[dict_key] = temp_dists
            
        # Remove existent transloc_events attribute from pace_maker
        del pace_maker.transloc_events

        # Remove 'None' entries.
        batch_data.batch_translocs = [
            event for event in batch_data.batch_translocs if event != None]

        # Print number of translocations found in files contained in batch
        print("Translocations in batch:\n" + f"{batch_data.batch_translocs}")
        # Print list of beat counts.
        print("Beat counts in batch:\n" + f"{batch_data.batch_beat_counts}")
        # Print event length (beats) dictionary.
        print("Event Time (beats) Dictionary:\n" + f"{batch_data.beat_event_dict}")
        # Print event length (milliseconds) dictionary.
        print("Event Time (ms) Dictionary:\n" + f"{batch_data.beat_time_dict}")
        # Print translocation distance (micrometer) dictionary
        print("Transloc. Dist (um) Dictionary:\n" + f"{batch_data.trans_dist_dict}")
        # Batch processing end.
        print("Batch processing complete.")
        # Reset batch_config flag to False to allow for single-file analysis
        batch_data.batch_config = False
    except (KeyError):
        print("Unsupported File Type.")
    except FileNotFoundError:
        print("No file selected.")

