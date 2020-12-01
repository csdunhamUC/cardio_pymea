# Author: Christopher Stuart Dunham (CSD)
# Email: csdunham@chem.ucla.edu; azarhn@hotmail.com
# Github: https://github.com/Sapphyric/Python_Learning
# Organization: University of California, Los Angeles, Department of Chemistry & Biochemistry
# Laboratory PI: James K. Gimzewski
# This is an original work, unless otherwise noted in comments, by CSD.
# Technical start date: 7/22/2020
# Effective start date: 9/10/2020
# Designed to run on Python 3.6 or newer.  Programmed under Python 3.8.
# Biggest known issues for Python versions earlier than 3.6:
# 1) Use of dictionary to contain electrode coordinates (ordered vs unordered)
# Consider using an OrderedDict instead if running under earlier versions of Python.
# 2) tkinter vs Tkinter for GUI.
# Program is currently set up to deal with 120 electrode MEAs from Multichannel Systems only.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
# import dask.dataframe as dd
import pandasgui as pgui
import seaborn as sns
import os
import time
import tkinter as tk
from numba import njit
from scipy.signal import find_peaks
from scipy import stats
from dis import dis
import datetime

#######################################################################################################################
# Classes that serve similar to Matlab structures (C "struct") to house data and allow it to be passed from
# one function to another.  Classes are generated for ImportedData (where the raw data will go), PaceMakerData
# (where PM data will go), UpstrokeVelData (where dV/dt data will go), LocalATData (where LAT data will go), and
# CondVelData, where CV data will go.
class ImportedData:
    pass


class InputParameters:
    pass


class BeatAmplitudes:
    pass


class PacemakerData:
    pass


class UpstrokeVelData:
    pass


class LocalATData:
    pass


class CondVelData:
    pass


class MEAHeatMaps:
    pass

class StatisticsData:
    pass


# Class containing electrode names and corresponding coordinates in x,y form, units of micrometers (microns, um)
class ElectrodeConfig:
    # Electrode names and coordinates, using the system defined by CSD where origin (0,0) is at upper left corner of MEA
    mea_120_coordinates = {'F7': [1150, 1380], 'F8': [1150, 1610], 'F12': [1150, 2530], 'F11': [1150, 2300], 'F10': [1150, 2070],
                           'F9': [1150, 1840], 'E12': [920, 2530], 'E11': [920, 2300], 'E10': [920, 2070], 'E9': [920, 1840],
                           'D12': [690, 2530], 'D11': [690, 2300], 'D10': [690, 2070], 'D9': [690, 1840], 'C11': [460, 2300],
                           'C10': [460, 2070], 'B10': [230, 2070], 'E8': [920, 1610], 'C9': [460, 1840], 'B9': [230, 1840],
                           'A9': [0, 1840], 'D8': [690, 1610], 'C8': [460, 1610], 'B8': [230, 1610], 'A8': [0, 1610],
                           'D7': [690, 1380], 'C7': [460, 1380], 'B7': [230, 1380], 'A7': [0, 1380], 'E7': [920, 1380],
                           'F6': [1150, 1150], 'E6': [920, 1150], 'A6': [0, 1150], 'B6': [230, 1150], 'C6': [460, 1150],
                           'D6': [690, 1150], 'A5': [0, 920], 'B5': [230, 920], 'C5': [460, 920], 'D5': [690, 920],
                           'A4': [0, 690], 'B4': [230, 690], 'C4': [460, 690], 'D4': [690, 690], 'B3': [230, 460],
                           'C3': [460, 460], 'C2': [460, 230], 'E5': [920, 920], 'D3': [690, 460], 'D2': [690, 230],
                           'D1': [690, 0], 'E4': [920, 690], 'E3': [920, 460], 'E2': [920, 230], 'E1': [920, 0],
                           'F4': [1150, 690], 'F3': [1150, 460], 'F2': [1150, 230], 'F1': [1150, 0], 'F5': [1150, 920],
                           'G6': [1380, 1150], 'G5': [1380, 920], 'G1': [1380, 0], 'G2': [1380, 230], 'G3': [1380, 460],
                           'G4': [1380, 690], 'H1': [1610, 0], 'H2': [1610, 230], 'H3': [1610, 460], 'H4': [1610, 690],
                           'J1': [1840, 0], 'J2': [1840, 230], 'J3': [1840, 460], 'J4': [1840, 690], 'K2': [2070, 230],
                           'K3': [2070, 460], 'L3': [2300, 460], 'H5': [1610, 920], 'K4': [2070, 690], 'L4': [2300, 690],
                           'M4': [2530, 690], 'J5': [1840, 920], 'K5': [2070, 920], 'L5': [2300, 920], 'M5': [2530, 920],
                           'J6': [1840, 1150], 'K6': [2070, 1150], 'L6': [2300, 1150], 'M6': [2530, 1150], 'H6': [1610, 1150],
                           'G7': [1380, 1380], 'H7': [1610, 1380], 'M7': [2530, 1380], 'L7': [2300, 1380], 'K7': [2070, 1380],
                           'J7': [1840, 1380], 'M8': [2530, 1610], 'L8': [2300, 1610], 'K8': [2070, 1610], 'J8': [1840, 1610],
                           'M9': [2530, 1840], 'L9': [2300, 1840], 'K9': [2070, 1840], 'J9': [1840, 1840], 'L10': [2300, 2070],
                           'K10': [2070, 2070], 'K11': [2070, 2300], 'H8': [1610, 1610], 'J10': [1840, 2070], 'J11': [1840, 2300],
                           'J12': [1840, 2530], 'H9': [1610, 1840], 'H10': [1610, 2070], 'H11': [1610, 2300], 'H12': [1610, 2530],
                           'G9': [1380, 1840], 'G10': [1380, 2070], 'G11': [1380, 2300], 'G12': [1380, 2530], 'G8': [1380, 1610]}

    # Key values (electrode names) from mea_120_coordinates only.
    electrode_names = list(mea_120_coordinates.keys())
    electrode_coords_x = np.array([i[0] for i in mea_120_coordinates.values()])
    electrode_coords_y = np.array([i[1] for i in mea_120_coordinates.values()])


# Import data files.  Files must be in .txt or .csv format.  May add toggles or checks to support more data types.
def data_import(elecGUI120, raw_data):
    try:
        data_filename_and_path = tk.filedialog.askopenfilename(initialdir=elecGUI120.file_path.get(), title="Select file",
                                                   filetypes=(("txt files", "*.txt"), ("all files", "*.*")))

        import_path, import_filename = os.path.split(data_filename_and_path)
        # start_time = time.process_time()

        # Checks whether data was previously imported into program.  If True, the previous data is deleted.
        if hasattr(raw_data, 'imported') is True:
            print("Raw data is not empty; clearing before reading file.")
            delattr(raw_data, 'imported')
            delattr(raw_data, 'names')

        # print("Importing data...")
        print("Import data began at: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Import electrodes for column headers from file.
        raw_data.names = pd.read_csv(data_filename_and_path, sep="\s+\t", lineterminator='\n', skiprows=[0, 1, 3], header=None,
                                     nrows=1, encoding='iso-8859-15', skipinitialspace=True, engine='python')

        # # Dask implementation, explanation to follow
        # temp_import = dd.read_csv(data_filename_and_path, sep='\s+', lineterminator='\n', skiprows=3, header=0,
        #                                 encoding='iso-8859-15', skipinitialspace=True, low_memory=False)
        # raw_data.imported = temp_import.compute()
        #
        # # Clear temp_import.  For reasons I don't understand, loading another file without deleting this causes the import
        # # to take considerably longer (6x or more?).  Presumably the temp_import variable was still hanging around.
        # del temp_import # didn't solve performance issue when opening another file...

        # # Import data from file.
        raw_data.imported = pd.read_csv(data_filename_and_path, sep='\s+', lineterminator='\n', skiprows=3, header=0,
                                        encoding='iso-8859-15', skipinitialspace=True, low_memory=False)

        print("Import data completed at: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Update file name display in GUI following import
        elecGUI120.file_name_label.configure(text=import_filename)
        elecGUI120.file_path.set(import_path)
        new_data_size = np.shape(raw_data.imported)
        print(new_data_size)
        # end_time = time.process_time()
        # print(end_time - start_time)
        # print("Import complete.")
        # return raw_data.imported
    except FileNotFoundError:
        print()
    except TypeError:
        print()


# Finds peaks based on given input parameters.
def determine_beats(elecGUI120, raw_data, cm_beats, input_param):
    try:
        print("Finding beats...\n")
        start_time = time.process_time()

        if hasattr(cm_beats, 'x_axis') is True:
            print("Beat data are not empty; clearing before finding beats.")
            delattr(cm_beats, 'x_axis')
            delattr(cm_beats, 'dist_beats')
            delattr(cm_beats, 'prom_beats')
            delattr(cm_beats, 'width_beats')
            delattr(cm_beats, 'thresh_beats')

        cm_beats.x_axis = raw_data.imported.iloc[0:, 0]
        # y_axis indexing ends at column -1, or second to last column, to remove the columns containing only \r
        if '\r' in raw_data.imported.columns:
            cm_beats.y_axis = raw_data.imported.iloc[0:, 1:-1]
        else:
            cm_beats.y_axis = raw_data.imported.iloc[0:, 1:]

        print("Y-axis data type is:: " + str(type(cm_beats.y_axis)) + "\n")
        print("Number of columns in cm_beats.y_axis: " + str(len(cm_beats.y_axis.columns)) + "\n")

        input_param.elec_choice = int(elecGUI120.elec_to_plot_val.get()) - 1
        input_param.min_peak_dist = float(elecGUI120.min_peak_dist_val.get())
        input_param.min_peak_height = float(elecGUI120.min_peak_height_val.get())
        input_param.parameter_prominence = float(elecGUI120.parameter_prominence_val.get())
        input_param.parameter_width = float(elecGUI120.parameter_width_val.get())
        input_param.parameter_thresh = float(elecGUI120.parameter_thresh_val.get())

        # Establish "strucs" as dataframes for subsequent operations.
        cm_beats.dist_beats = pd.DataFrame()
        cm_beats.prom_beats = pd.DataFrame()
        cm_beats.width_beats = pd.DataFrame()
        cm_beats.thresh_beats = pd.DataFrame()

        print("Summary of parameters: " + str(input_param.min_peak_dist) + ", " + str(input_param.min_peak_height) +
              ", " + str(input_param.parameter_prominence) + ", " + str(input_param.parameter_width) + ", " +
              str(input_param.parameter_thresh) + ".\n")

        # For loops for finding beats (peaks) in each channel (electrode).  Suitable for any given MCD-converted file
        # in which only one MEA is recorded (i.e. works for a single 120 electrode MEA, or 60 electrode MEA, etc
        # Caveat 1: have not tested for singular MEA60 data.
        # Disclaimer: Not currently equipped to handle datasets with dual-recorded MEAs (e.g. dual MEA60s)
        for column in range(len(cm_beats.y_axis.columns)):
            dist_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], height=input_param.min_peak_height,
                                              distance=input_param.min_peak_dist)[0], name=column+1)
            cm_beats.dist_beats = pd.concat([cm_beats.dist_beats, dist_beats], axis='columns')

            prom_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], height=input_param.min_peak_height,
                                              distance=input_param.min_peak_dist, prominence=input_param.parameter_prominence)[0], name=column+1)
            cm_beats.prom_beats = pd.concat([cm_beats.prom_beats, prom_beats], axis='columns')

            width_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], height=input_param.min_peak_height,
                                               distance=input_param.min_peak_dist, width=input_param.parameter_width)[0], name=column+1)
            cm_beats.width_beats = pd.concat([cm_beats.width_beats, width_beats], axis='columns')

            thresh_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], height=input_param.min_peak_height,
                                                distance=input_param.min_peak_dist, threshold=input_param.parameter_thresh)[0], name=column+1)
            cm_beats.thresh_beats = pd.concat([cm_beats.thresh_beats, thresh_beats], axis='columns')

        # Data designation to ensure NaN values are properly handled by subsequent calculations.
        cm_beats.dist_beats.astype('Int64')
        cm_beats.prom_beats.astype('Int64')
        cm_beats.width_beats.astype('Int64')
        cm_beats.thresh_beats.astype('Int64')

        # Generate beat counts for the different peakfinder methods by finding the length of each electrode (column).
        cm_beats.beat_count_dist = np.zeros(len(cm_beats.dist_beats.columns))
        for column in range(len(cm_beats.dist_beats.columns)):
            cm_beats.beat_count_dist[column] = len(cm_beats.dist_beats.iloc[0:, column].dropna(axis='index'))

        cm_beats.beat_count_prom = np.zeros(len(cm_beats.prom_beats.columns))
        for column in range(len(cm_beats.prom_beats.columns)):
            cm_beats.beat_count_prom[column] = len(cm_beats.prom_beats.iloc[0:, column].dropna(axis='index'))

        cm_beats.beat_count_width = np.zeros(len(cm_beats.width_beats.columns))
        for column in range(len(cm_beats.width_beats.columns)):
            cm_beats.beat_count_width[column] = len(cm_beats.width_beats.iloc[0:, column].dropna(axis='index'))

        cm_beats.beat_count_thresh = np.zeros(len(cm_beats.thresh_beats.columns))
        for column in range(len(cm_beats.thresh_beats.columns)):
            cm_beats.beat_count_thresh[column] = len(cm_beats.thresh_beats.iloc[0:, column].dropna(axis='index'))

        # Finds the mode of beats across the dataset for each peakfinder parameter set.
        cm_beats.beat_count_dist_mode = stats.mode(cm_beats.beat_count_dist)
        cm_beats.beat_count_prom_mode = stats.mode(cm_beats.beat_count_prom)
        cm_beats.beat_count_width_mode = stats.mode(cm_beats.beat_count_width)
        cm_beats.beat_count_thresh_mode = stats.mode(cm_beats.beat_count_thresh)

        # Prints the output from the preceding operations.
        print("Mode of beats using distance parameter: " + str(cm_beats.beat_count_dist_mode))
        print("Mode of beats using prominence parameter: " + str(cm_beats.beat_count_prom_mode))
        print("Mode of beats using width parameter: " + str(cm_beats.beat_count_width_mode))
        print("Mode of beats using threshold parameter: " + str(cm_beats.beat_count_thresh_mode))

        dist_beats_size = np.shape(cm_beats.dist_beats)
        print("Shape of cm_beats.dist_beats: " + str(dist_beats_size))
        prom_beats_size = np.shape(cm_beats.prom_beats)
        print("Shape of cm_beats.prom_beats: " + str(prom_beats_size))
        width_beats_size = np.shape(cm_beats.width_beats)
        print("Shape of cm_beats.width_beats: " + str(width_beats_size) + ".\n")
        thresh_beats_size = np.shape(cm_beats.thresh_beats)
        print("Shape of cm_beats.thresh_beats: " + str(thresh_beats_size) + ".\n")

        print("Finished.")
        end_time = time.process_time()
        print(end_time - start_time)
        print("Plotting...")
        elecGUI120.beat_detect_window(cm_beats, input_param)
        graph_beats(elecGUI120, cm_beats, input_param)
    except AttributeError:
        print("No data found. Please import data (.txt or .csv converted MCD file) first.")


# Function that calculates the pacemaker (time lag).  Performs this calculation for all electrodes, and filters
# electrodes based on mismatched beat counts relative to the mode of the beat count.
def calculate_pacemaker(elecGUI120, cm_beats, pace_maker, heat_map, input_param):
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


# For future ref: numpy.polynomial.polynomial.polyfit(x,y,order) or scipy.stats.linregress(x,y)
# Calculates upstroke velocity (dV/dt)
def calculate_upstroke_vel(elecGUI120, cm_beats, upstroke_vel, heat_map, input_param):
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

        for electrode in range(num_of_electrodes):
            per_electrode_mode_of_beats[electrode] = len(cm_beats.dist_beats.iloc[0:, electrode].dropna())

        dvdt_max = dvdt_calc_for_numba(mode_of_beats, num_of_electrodes, cm_beats_dist_beats_as_array,
                                       cm_beats_y_axis_as_array, per_electrode_mode_of_beats)

        upstroke_vel.final_dist_beat_count = []
        for beat in range(int(cm_beats.beat_count_dist_mode[0])):
            upstroke_vel.final_dist_beat_count.append('Beat ' + str(beat + 1))

        upstroke_vel.param_dist_raw = pd.DataFrame(dvdt_max, index=upstroke_vel.final_dist_beat_count).T
        upstroke_vel.param_dist_normalized = upstroke_vel.param_dist_raw.sub(upstroke_vel.param_dist_raw.min(axis=0),
                                                                             axis=1)

        # Normalized dV/dt across the dataset.
        upstroke_vel.param_dist_normalized_max = upstroke_vel.param_dist_normalized.max().max()

        upstroke_vel.param_dist_normalized.index = ElectrodeConfig.electrode_names
        upstroke_vel.param_dist_normalized.insert(0, 'Electrode', ElectrodeConfig.electrode_names)
        upstroke_vel.param_dist_normalized.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        upstroke_vel.param_dist_normalized.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)

        # Assign name to resulting dataframe.
        upstroke_vel.param_dist_normalized.name = 'Upstroke Velocity'

        # Set slider value to maximum number of beats
        elecGUI120.mea_beat_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))

        print("Done")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        print(end_time - start_time)
    except AttributeError:
        print("Please use Find Peaks first.")


@njit
def dvdt_calc_for_numba(mode_of_beats, num_of_electrodes, cm_beats_dist_beats_as_array, cm_beats_y_axis_as_array, per_electrode_mode_of_beats):
    temp_slope = np.zeros((num_of_electrodes, 5))
    temp_dvdt_max = np.zeros(num_of_electrodes)
    x_values = np.zeros((2, 5))
    y_values = np.zeros((2, 5))
    dvdt_max = np.zeros((mode_of_beats, num_of_electrodes))

    for beat in range(mode_of_beats):
        for electrode in range(num_of_electrodes):
            if mode_of_beats == per_electrode_mode_of_beats[electrode]:
                x_2_1 = cm_beats_dist_beats_as_array[beat, electrode]
                x_2_2 = x_2_1 - 1
                x_2_3 = x_2_2 - 1
                x_2_4 = x_2_3 - 1
                x_2_5 = x_2_4 - 1

                x_1_1 = x_2_1 - 1
                x_1_2 = x_2_2 - 1
                x_1_3 = x_2_3 - 1
                x_1_4 = x_2_4 - 1
                x_1_5 = x_2_5 - 1

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
                temp_slope[electrode] = calc_slope
            else:
                temp_slope[electrode] = [np.nan, np.nan, np.nan, np.nan, np.nan]
            temp_dvdt_max[electrode] = max(temp_slope[electrode])

        dvdt_max[beat] = temp_dvdt_max

    return dvdt_max


def calculate_lat(elecGUI120, cm_beats, local_act_time, heat_map, input_param):
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

        local_act_time.param_dist_normalized.index = ElectrodeConfig.electrode_names
        local_act_time.param_dist_normalized.insert(0, 'Electrode', ElectrodeConfig.electrode_names)
        local_act_time.param_dist_normalized.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        local_act_time.param_dist_normalized.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)

        # Assign name to resulting dataframe.
        local_act_time.param_dist_normalized.name = 'Local Activation Time'

        # Set slider value to maximum number of beats
        elecGUI120.mea_beat_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))

        print("Done")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        print(end_time - start_time)
        calculate_distances(local_act_time, ElectrodeConfig)
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
def calculate_distances(local_act_time, ElectrodeConfig):
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
                                                    columns=ElectrodeConfig.electrode_names).T

    print("Done.")
    end_time = time.process_time()
    print(end_time - start_time)


def calculate_conduction_velocity(elecGUI120, conduction_vel, local_act_time, heat_map, input_param):
    try:
        if hasattr(conduction_vel, 'param_dist_raw') is True:
            print("Clearing old CV data before running new calculation...")
            delattr(conduction_vel, 'param_dist_raw')
        start_time = time.process_time()
        conduction_vel.param_dist_raw = local_act_time.distance_from_min.divide(local_act_time.param_dist_normalized.loc[:, local_act_time.final_dist_beat_count])
        # Need to add a placeholder value for the minimum channel; currently gives NaN as a consequence of division by zero.
        # Want/need to display the origin for heat map purposes.  Question is how to do this efficiently.

        conduction_vel.param_dist_raw.index = ElectrodeConfig.electrode_names
        conduction_vel.param_dist_raw.insert(0, 'Electrode', ElectrodeConfig.electrode_names)
        conduction_vel.param_dist_raw.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        conduction_vel.param_dist_raw.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)

        end_time = time.process_time()
        print("CV calculation complete.")
        print(end_time - start_time)
        # graph_conduction_vel(elecGUI120, heat_map, local_act_time, conduction_vel, input_param)
    except AttributeError:
        print("Please calculate local activation time first.")


# ######################################################################################################################
# ################################################ Calculations End ####################################################
# ######################################################################################################################

# Usually just for debugging, a function that prints out values upon button press.
def data_print(elecGUI120, raw_data, pace_maker, input_param):
    # adding .iloc to a data frame allows to reference [row, column], where rows and columns can be ranges separated
    # by colons
    input_param.beat_choice = int(elecGUI120.mea_beat_select.get()) - 1
    print(pace_maker.param_dist_normalized.name)
    print(ElectrodeConfig.electrode_names[0])
    print(ElectrodeConfig.electrode_names[5])
    print(input_param.beat_choice)


def time_test():
    dis(calculate_lat)


# ######################################################################################################################
# ################################################# Graphing Starts ####################################################
# ######################################################################################################################

# Produces 4-subplot plot of peak finder data and graphs it.  Can be called via button.
# Will throw exception of data does not exist.
def graph_beats(elecGUI120, cm_beats, input_param):
    try:
        cm_beats.axis1.cla()
        cm_beats.axis2.cla()
        cm_beats.axis3.cla()
        cm_beats.axis4.cla()

        input_param.elec_choice = int(elecGUI120.elec_to_plot_val.get()) - 1
        print("Will generate graph for electrode " + str(input_param.elec_choice + 1) + ".")
        cm_beats.comp_plot.suptitle("Comparisons of find_peaks methodologies: electrode " + (str(input_param.elec_choice + 1)) + ".")

        mask_dist = ~np.isnan(cm_beats.dist_beats.iloc[0:, input_param.elec_choice].values)
        dist_without_nan = cm_beats.dist_beats.iloc[0:, input_param.elec_choice].values[mask_dist].astype('int64')
        cm_beats.axis1.plot(dist_without_nan, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values[dist_without_nan], "xr")
        cm_beats.axis1.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
        cm_beats.axis1.legend(['distance = ' + str(elecGUI120.min_peak_dist_val.get())], loc='lower left')

        mask_prom = ~np.isnan(cm_beats.prom_beats.iloc[0:, input_param.elec_choice].values)
        prom_without_nan = cm_beats.prom_beats.iloc[0:, input_param.elec_choice].values[mask_prom].astype('int64')
        cm_beats.axis2.plot(prom_without_nan, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values[prom_without_nan], "ob")
        cm_beats.axis2.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
        cm_beats.axis2.legend(['prominence = ' + str(input_param.parameter_prominence)], loc='lower left')

        mask_width = ~np.isnan(cm_beats.width_beats.iloc[0:, input_param.elec_choice].values)
        width_without_nan = cm_beats.width_beats.iloc[0:, input_param.elec_choice].values[mask_width].astype('int64')
        cm_beats.axis3.plot(width_without_nan, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values[width_without_nan], "vg")
        cm_beats.axis3.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
        cm_beats.axis3.legend(['width = ' + str(input_param.parameter_width)], loc='lower left')

        mask_thresh = ~np.isnan(cm_beats.thresh_beats.iloc[0:, input_param.elec_choice].values)
        thresh_without_nan = cm_beats.thresh_beats.iloc[0:, input_param.elec_choice].values[mask_thresh].astype('int64')
        cm_beats.axis4.plot(thresh_without_nan, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values[thresh_without_nan], "xk")
        cm_beats.axis4.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
        cm_beats.axis4.legend(['threshold = ' + str(input_param.parameter_thresh)], loc='lower left')

        cm_beats.comp_plot.canvas.draw()
        print("Plotting complete.")
    except AttributeError:
        print("Please use Find Peaks first.")


# This function is called following the use of "Calculate All Parameters" from the drop-down menu and from the GUI
# slider on the main window.  It generates the heat maps observed in the main window of the program. The code
# is largely identical with the individual functions, which (will soon) open their own windows for each calculation.
def graph_all(elecGUI120, heat_map, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param):
    # ------------------------------------------------- Pacemaker ------------------------------------------------------
    if hasattr(heat_map, 'cbar_1') is True:
        heat_map.cbar_1.remove()

    heat_map.axis1.cla()
    input_param.beat_choice = int(elecGUI120.mea_beat_select.get()) - 1

    electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', columns='X', values='Electrode')
    heatmap_pivot_table = pace_maker.param_dist_normalized.pivot(index='Y', columns='X',
                                                                 values=pace_maker.final_dist_beat_count[
                                                                     input_param.beat_choice])

    heat_map.temp = sns.heatmap(heatmap_pivot_table, cmap="jet", annot=electrode_names, fmt="", ax=heat_map.axis1,
                                vmin=0, vmax=pace_maker.param_dist_normalized_max, cbar=False)
    mappable = heat_map.temp.get_children()[0]
    heat_map.cbar_1 = heat_map.axis1.figure.colorbar(mappable, ax=heat_map.axis1)
    heat_map.cbar_1.ax.set_title("Time Lag (ms)", fontsize=10)

    heat_map.axis1.set(title="Pacemaker, Beat " + str(input_param.beat_choice + 1), xlabel="X coordinate (um)",
                       ylabel="Y coordinate (um)")

    # Upstroke velocity
    if hasattr(heat_map, 'cbar_2') is True:
        heat_map.cbar_2.remove()
    heat_map.axis2.cla()
    input_param.beat_choice_2 = int(elecGUI120.mea_beat_select.get()) - 1

    electrode_names_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', columns='X', values='Electrode')
    heatmap_pivot_table_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', columns='X',
                                                                     values=upstroke_vel.final_dist_beat_count[
                                                                         input_param.beat_choice_2])

    heat_map.temp_2 = sns.heatmap(heatmap_pivot_table_2, cmap="jet", annot=electrode_names_2, fmt="", ax=heat_map.axis2,
                                  vmax=upstroke_vel.param_dist_normalized_max, cbar=False)
    mappable_2 = heat_map.temp_2.get_children()[0]
    heat_map.cbar_2 = heat_map.axis2.figure.colorbar(mappable_2, ax=heat_map.axis2)
    heat_map.cbar_2.ax.set_title("uV/ms", fontsize=10)

    heat_map.axis2.set(title="Upstroke Velocity, Beat " + str(input_param.beat_choice_2 + 1),
                       xlabel="X coordinate (um)", ylabel="Y coordinate (um)")

    # Local activation time
    if hasattr(heat_map, 'cbar_3') is True:
        heat_map.cbar_3.remove()
    heat_map.axis3.cla()
    input_param.beat_choice_3 = int(elecGUI120.mea_beat_select.get()) - 1

    electrode_names_3 = local_act_time.param_dist_normalized.pivot(index='Y', columns='X', values='Electrode')
    heatmap_pivot_table_3 = local_act_time.param_dist_normalized.pivot(index='Y', columns='X',
                                                                       values=local_act_time.final_dist_beat_count[
                                                                           input_param.beat_choice_3])

    heat_map.temp_3 = sns.heatmap(heatmap_pivot_table_3, cmap="jet", annot=electrode_names_3, fmt="", ax=heat_map.axis3,
                                  vmax=local_act_time.param_dist_normalized_max, cbar=False)
    mappable_3 = heat_map.temp_3.get_children()[0]
    heat_map.cbar_3 = heat_map.axis3.figure.colorbar(mappable_3, ax=heat_map.axis3)
    heat_map.cbar_3.ax.set_title("Time Lag (ms)", fontsize=10)

    heat_map.axis3.set(title="Local Activation Time, Beat " + str(input_param.beat_choice_3 + 1),
                       xlabel="X coordinate (um)", ylabel="Y coordinate (um)")

    # Conduction velocity
    if hasattr(heat_map, 'cbar_4') is True:
        heat_map.cbar_4.remove()
    heat_map.axis4.cla()
    input_param.beat_choice_4 = int(elecGUI120.mea_beat_select.get()) - 1

    electrode_names_4 = conduction_vel.param_dist_raw.pivot(index='Y', columns='X', values='Electrode')
    heatmap_pivot_table_4 = conduction_vel.param_dist_raw.pivot(index='Y', columns='X',
                                                                values=local_act_time.final_dist_beat_count[
                                                                    input_param.beat_choice_4])

    heat_map.temp_4 = sns.heatmap(heatmap_pivot_table_4, cmap="jet", annot=electrode_names_4, fmt="", ax=heat_map.axis4,
                                  cbar=False)
    mappable_4 = heat_map.temp_4.get_children()[0]
    heat_map.cbar_4 = heat_map.axis4.figure.colorbar(mappable_4, ax=heat_map.axis4)
    heat_map.cbar_4.ax.set_title("um/(ms)", fontsize=10)

    heat_map.axis4.set(title="Conduction Velocity, Beat " + str(input_param.beat_choice_4 + 1),
                       xlabel="X coordinate (um)", ylabel="Y coordinate (um)")

    heat_map.curr_plot.tight_layout()
    heat_map.curr_plot.canvas.draw()


# Construct heatmap from previously calculated pacemaker data.  Function is called each time the slider is moved to
# select a new beat.
def graph_pacemaker(elecGUI120, heat_map, pace_maker, input_param):
    try:
        if hasattr(heat_map, 'pm_solo_cbar') is True:
            heat_map.pm_solo_cbar.remove()

        heat_map.pm_solo_axis.cla()
        input_param.pm_solo_beat_choice = int(elecGUI120.pm_solo_beat_select.get()) - 1

        electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', columns='X', values='Electrode')
        heatmap_pivot_table = pace_maker.param_dist_normalized.pivot(index='Y', columns='X', values=pace_maker.final_dist_beat_count[input_param.pm_solo_beat_choice])

        heat_map.pm_solo_temp = sns.heatmap(heatmap_pivot_table, cmap="jet", annot=electrode_names, fmt="", ax=heat_map.pm_solo_axis, vmin=0, vmax=pace_maker.param_dist_normalized_max, cbar=False)
        mappable = heat_map.pm_solo_temp.get_children()[0]
        heat_map.pm_solo_cbar = heat_map.pm_solo_axis.figure.colorbar(mappable, ax=heat_map.pm_solo_axis)
        heat_map.pm_solo_cbar.ax.set_title("Time Lag (ms)", fontsize=10)

        heat_map.pm_solo_axis.set(title="Pacemaker, Beat " + str(input_param.pm_solo_beat_choice+1), xlabel="X coordinate (um)", ylabel="Y coordinate (um)")
        heat_map.pm_solo_plot.tight_layout()
        heat_map.pm_solo_plot.canvas.draw()

    except AttributeError:
        print("Please calculate PM first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def graph_upstroke(elecGUI120, heat_map, upstroke_vel, input_param):
    try:
        if hasattr(heat_map, 'dvdt_solo_cbar') is True:
            heat_map.dvdt_solo_cbar.remove()

        heat_map.dvdt_solo_axis.cla()
        input_param.dvdt_solo_beat_choice = int(elecGUI120.dvdt_solo_beat_select.get()) - 1

        electrode_names_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', columns='X', values='Electrode')
        heatmap_pivot_table_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', columns='X', values=upstroke_vel.final_dist_beat_count[input_param.dvdt_solo_beat_choice])

        heat_map.dvdt_solo_temp = sns.heatmap(heatmap_pivot_table_2, cmap="jet", annot=electrode_names_2, fmt="", ax=heat_map.dvdt_solo_axis, vmax=upstroke_vel.param_dist_normalized_max, cbar=False)
        mappable_2 = heat_map.dvdt_solo_temp.get_children()[0]
        heat_map.dvdt_solo_cbar = heat_map.dvdt_solo_axis.figure.colorbar(mappable_2, ax=heat_map.dvdt_solo_axis)
        heat_map.dvdt_solo_cbar.ax.set_title("uV/ms", fontsize=10)

        heat_map.dvdt_solo_axis.set(title="Upstroke Velocity, Beat " + str(input_param.dvdt_solo_beat_choice+1), xlabel="X coordinate (um)", ylabel="Y coordinate (um)")
        heat_map.dvdt_solo_plot.tight_layout()
        heat_map.dvdt_solo_plot.canvas.draw()

    except AttributeError:
        print("Please calculate dV/dt first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def graph_local_act_time(elecGUI120, heat_map, local_act_time, input_param):
    try:
        if hasattr(heat_map, 'lat_solo_cbar') is True:
            heat_map.lat_solo_cbar.remove()
        heat_map.lat_solo_axis.cla()
        input_param.lat_solo_beat_choice = int(elecGUI120.lat_solo_beat_select.get()) - 1

        electrode_names_3 = local_act_time.param_dist_normalized.pivot(index='Y', columns='X', values='Electrode')
        heatmap_pivot_table_3 = local_act_time.param_dist_normalized.pivot(index='Y', columns='X', values=local_act_time.final_dist_beat_count[input_param.lat_solo_beat_choice])

        heat_map.lat_solo_temp = sns.heatmap(heatmap_pivot_table_3, cmap="jet", annot=electrode_names_3, fmt="", ax=heat_map.lat_solo_axis, vmax=local_act_time.param_dist_normalized_max, cbar=False)
        mappable_3 = heat_map.lat_solo_temp.get_children()[0]
        heat_map.lat_solo_cbar = heat_map.lat_solo_axis.figure.colorbar(mappable_3, ax=heat_map.lat_solo_axis)
        heat_map.lat_solo_cbar.ax.set_title("Time Lag (ms)", fontsize=10)

        heat_map.lat_solo_axis.set(title="Local Activation Time, Beat " + str(input_param.lat_solo_beat_choice+1), xlabel="X coordinate (um)", ylabel="Y coordinate (um)")
        heat_map.lat_solo_plot.tight_layout()
        heat_map.lat_solo_plot.canvas.draw()

    except AttributeError:
        print("Please calculate LAT first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def graph_conduction_vel(elecGUI120, heat_map, local_act_time, conduction_vel, input_param):
    try:
        if hasattr(heat_map, 'cv_solo_cbar') is True:
            heat_map.cv_solo_cbar.remove()
        heat_map.cv_solo_axis.cla()
        input_param.cv_solo_beat_choice = int(elecGUI120.cv_solo_beat_select.get()) - 1

        electrode_names_4 = conduction_vel.param_dist_raw.pivot(index='Y', columns='X', values='Electrode')
        heatmap_pivot_table_4 = conduction_vel.param_dist_raw.pivot(index='Y', columns='X', values=local_act_time.final_dist_beat_count[input_param.cv_solo_beat_choice])

        heat_map.cv_solo_temp = sns.heatmap(heatmap_pivot_table_4, cmap="jet", annot=electrode_names_4, fmt="", ax=heat_map.cv_solo_axis, cbar=False)
        mappable_4 = heat_map.cv_solo_temp.get_children()[0]
        heat_map.cv_solo_cbar = heat_map.cv_solo_axis.figure.colorbar(mappable_4, ax=heat_map.cv_solo_axis)
        heat_map.cv_solo_cbar.ax.set_title("um/(ms)", fontsize=10)

        heat_map.cv_solo_axis.set(title="Conduction Velocity, Beat " + str(input_param.cv_solo_beat_choice+1), xlabel="X coordinate (um)", ylabel="Y coordinate (um)")
        heat_map.cv_solo_plot.tight_layout()
        heat_map.cv_solo_plot.canvas.draw()

    except AttributeError:
        print("Please make sure you've calculated Local Activation Time first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def show_dataframes(raw_data, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel):
    try:
        cm_beats_dist_data = cm_beats.dist_beats
        pm_normalized = pace_maker.param_dist_normalized
        dVdt_normalized = upstroke_vel.param_dist_normalized
        lat_normalized = local_act_time.param_dist_normalized
        lat_distances = local_act_time.distance_from_min
        cv_raw = conduction_vel.param_dist_raw
        pgui.show(cm_beats_dist_data, pm_normalized, dVdt_normalized, lat_normalized, lat_distances, cv_raw, settings={'block': True})
    except(AttributeError):
        print("Please run all of your calculations first.")


def param_vs_distance_analysis(elecGUI120, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats):
    print()
    input_param.stats_param_dist_slider = int(elecGUI120.param_vs_dist_beat_select.get()) - 1
    # First thing first: plot stuff vs distance.  Distances must be x-values, parameters must be y-values from sel. beat
    #         cm_beats.comp_plot.suptitle("Comparisons of find_peaks methodologies: electrode " + (str(input_param.elec_choice + 1)) + ".")
    #
    #         mask_dist = ~np.isnan(cm_beats.dist_beats.iloc[0:, input_param.elec_choice].values)
    #         dist_without_nan = cm_beats.dist_beats.iloc[0:, input_param.elec_choice].values[mask_dist].astype('int64')
    #         cm_beats.axis1.plot(dist_without_nan, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values[dist_without_nan], "xr")
    #         cm_beats.axis1.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
    #         cm_beats.axis1.legend(['distance = ' + str(elecGUI120.min_peak_dist_val.get())], loc='lower left')


    # x-values @: local_act_time.distance_from_min
    # y-values @: pace_maker.param_dist_normalized, upstroke_vel.param_dist_normalized,
    # local_act_time.param_dist_normalized, conduction_vel.param_dist_raw
    cm_stats.param_vs_dist_axis_pm.cla()
    cm_stats.param_vs_dist_axis_dvdt.cla()
    cm_stats.param_vs_dist_axis_lat.cla()
    cm_stats.param_vs_dist_axis_cv.cla()

    # mask_coords = ~np.isnan(local_act_time.distance_from_min[input_param.stats_param_dist_slider])
    cm_stats.param_vs_dist_plot.suptitle("Parameter vs. Distance from Minimum.  Beat: " + str(input_param.stats_param_dist_slider + 1) + ".")
    cm_stats.param_vs_dist_axis_pm.scatter(local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                        pace_maker.param_dist_normalized[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                           c='red')
    cm_stats.param_vs_dist_axis_pm.set(title="Pacemaker")
    cm_stats.param_vs_dist_axis_dvdt.scatter(local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                         upstroke_vel.param_dist_normalized[upstroke_vel.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                             c='green')
    cm_stats.param_vs_dist_axis_dvdt.set(title="Upstroke Velocity")
    cm_stats.param_vs_dist_axis_lat.scatter(local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                         local_act_time.param_dist_normalized[local_act_time.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                            c='orange')
    cm_stats.param_vs_dist_axis_lat.set(title="Local Activation Time")
    cm_stats.param_vs_dist_axis_cv.scatter(local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                        conduction_vel.param_dist_raw[local_act_time.final_dist_beat_count[input_param.stats_param_dist_slider]],
                                           c='blue')
    cm_stats.param_vs_dist_axis_cv.set(title="Conduction Velocity")

    cm_stats.param_vs_dist_plot.tight_layout()
    cm_stats.param_vs_dist_plot.canvas.draw()

    # Necessary operations:
    # 1) Elimination of outliers (calculate mean, stdev, remove data > mean*3 sigma)
    # 2) Calculate R^2 values, per beat, for each parameter vs distance
    # 3) Plot data

    # Necessary parameters:
    # 1) Sigma
    # 2) Percentile of R^2 to display/indicate

    # Necessary readouts:
    # 1) Dataset averages and standard deviation for each parameter (dV/dt, CV, PM, LAT)
    # 2) Dataset average and standard deviation of R^2 for each parameter (sorted high to low).
    # 3) Mode of PM (LAT) min & max channels.
    # 4) Mode of CV min and max channels.
    # 5) Number of unique min channels for PM (LAT)


class ElecGUI120(tk.Frame):
    def __init__(self, master, raw_data, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, heat_map, cm_stats):
        tk.Frame.__init__(self, master)
        # The fun story about grid: columns and rows cannot be generated past the number of widgets you have (or at
        # least I've yet to learn the way to do so, and will update if I find out how).  It's all relative geometry,
        # where the relation is with other widgets.  If you have 3 widgets you can have 3 rows or 3 columns and
        # organize accordingly.  If you have 2 widgets you can only have 2 rows or 2 columns.
        # use .bind("<Enter>", "color") or .bind("<Leave>", "color") to change mouse-over color effects.
        self.grid()
        self.winfo_toplevel().title("MEA Analysis - v2")

        # Directory information for file import is stored here and called upon by import function.  Default/initial "/"
        self.file_path = tk.StringVar()
        self.file_path.set("/")

        # ############################################### Menu ########################################################
        self.master = master
        menu = tk.Menu(self.master, tearoff=False)
        self.master.config(menu=menu)
        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Data (.csv or .txt)", command=lambda: data_import(self, raw_data))
        file_menu.add_command(label="Save Processed Data", command=None)
        file_menu.add_command(label="Save Heatmaps", command=None)
        file_menu.add_command(label="Print (Debug)", command=lambda: data_print(self, raw_data, pace_maker, input_param))

        view_menu = tk.Menu(menu)
        menu.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="View Pandas Dataframes", command=lambda: show_dataframes(raw_data, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel))

        calc_menu = tk.Menu(menu)
        menu.add_cascade(label="Calculations", menu=calc_menu)
        calc_menu.add_command(label="Beat Detect (Run First!)", command=lambda: determine_beats(self, raw_data, cm_beats, input_param))
        calc_menu.add_command(label="Calculate All Parameters",
                              command=lambda: [calculate_pacemaker(self, cm_beats, pace_maker, heat_map, input_param),
                                               calculate_upstroke_vel(self, cm_beats, upstroke_vel, heat_map, input_param),
                                               calculate_lat(self, cm_beats, local_act_time, heat_map, input_param),
                                               calculate_conduction_velocity(self, conduction_vel, local_act_time, heat_map, input_param),
                                               graph_all(self, heat_map, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param)])
        # Add extra command for each solitary calculation that calls the appropriate graphing function.
        # The graphing function will call the appropriate method to open the window.  This will allow for individual
        # parameter analysis.
        calc_menu.add_command(label="Pacemaker", command=lambda: [calculate_pacemaker(self, cm_beats, pace_maker, heat_map, input_param),
                                                                  self.pacemaker_heatmap_window(cm_beats, pace_maker, heat_map, input_param),
                                                                  graph_pacemaker(self, heat_map, pace_maker, input_param)])
        calc_menu.add_command(label="Upstroke Velocity", command=lambda: [calculate_upstroke_vel(self, cm_beats, upstroke_vel, heat_map, input_param),
                                                                          self.dvdt_heatmap_window(cm_beats, upstroke_vel, heat_map, input_param),
                                                                          graph_upstroke(self, heat_map, upstroke_vel, input_param)])
        calc_menu.add_command(label="Local Activation Time", command=lambda: [calculate_lat(self, cm_beats, local_act_time, heat_map, input_param),
                                                                              self.lat_heatmap_window(cm_beats, local_act_time, heat_map, input_param),
                                                                              graph_local_act_time(self, heat_map, local_act_time, input_param)])
        calc_menu.add_command(label="Conduction Velocity", command=lambda: [calculate_conduction_velocity(self, conduction_vel, local_act_time, heat_map, input_param),
                                                                            self.cv_heatmap_window(cm_beats, local_act_time, conduction_vel, heat_map, input_param),
                                                                            graph_conduction_vel(self, heat_map, local_act_time, conduction_vel, input_param)])

        statistics_menu = tk.Menu(menu)
        menu.add_cascade(label="Statistics", menu=statistics_menu)
        statistics_menu.add_command(label="Parameter vs Distance Plot w/ R-Square", command=lambda: [self.param_vs_dist_stats_window(cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats),
                                    param_vs_distance_analysis(self, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats)])
        statistics_menu.add_command(label="Radial Binning Plot w/ R-Square", command=None)
        statistics_menu.add_command(label="Q-Q Plot",  command=None)

        advanced_tools_menu = tk.Menu(menu)
        menu.add_cascade(label="Advanced Tools", menu=advanced_tools_menu)
        advanced_tools_menu.add_command(label="Test Efficiency", command=lambda: time_test())
        advanced_tools_menu.add_command(label="K-Means Clustering", command=None)
        advanced_tools_menu.add_command(label="t-SNE", command=None)
        advanced_tools_menu.add_command(label="DBSCAN", command=None)
        advanced_tools_menu.add_command(label="PCA", command=None)

        # ############################################### Entry Fields ################################################
        # Frame for MEA parameters (e.g. plotted electrode, min peak distance, min peak amplitude, prominence, etc)
        self.mea_parameters_frame = tk.Frame(self, width=1620, height=100, bg="white")
        self.mea_parameters_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        self.mea_parameters_frame.grid_propagate(False)

        # self.file_name = tk.StringVar()
        # self.file_name.set("No file")
        self.file_name_label = tk.Label(self.mea_parameters_frame, text="No file", bg="white", wraplength=200)
        self.file_name_label.grid(row=0, column=9, columnspan=4, padx=5, pady=5)

        # Min peak distance label, entry field, trace and positioning.
        self.min_peak_dist_label = tk.Label(self.mea_parameters_frame, text="Min Peak Distance", bg="white", wraplength=80)
        self.min_peak_dist_label.grid(row=0, column=1, padx=5, pady=2)
        self.min_peak_dist_val = tk.StringVar()
        self.min_peak_dist_val.trace_add("write", self.min_peak_dist_callback)
        self.min_peak_dist_val.set("1000")
        self.min_peak_dist_entry = tk.Entry(self.mea_parameters_frame, text=self.min_peak_dist_val, width=8)
        self.min_peak_dist_entry.grid(row=1, column=1, padx=5, pady=2)

        # Min peak height label, entry field, trace and positioning.
        self.min_peak_height_label = tk.Label(self.mea_parameters_frame, text="Min Peak Height", bg="white", wraplength=80)
        self.min_peak_height_label.grid(row=0, column=2, padx=5, pady=2)
        self.min_peak_height_val = tk.StringVar()
        self.min_peak_height_val.trace_add("write", self.min_peak_height_callback)
        self.min_peak_height_val.set("100")
        self.min_peak_height_entry = tk.Entry(self.mea_parameters_frame, text=self.min_peak_height_val, width=8)
        self.min_peak_height_entry.grid(row=1, column=2, padx=5, pady=2)

        # Peak prominence label, entry field, trace and positioning.
        self.parameter_prominence_label = tk.Label(self.mea_parameters_frame, text="Peak Prominence", bg="white", wraplength=100)
        self.parameter_prominence_label.grid(row=0, column=3, padx=5, pady=2)
        self.parameter_prominence_val = tk.StringVar()
        self.parameter_prominence_val.trace_add("write", self.parameter_prominence_callback)
        self.parameter_prominence_val.set("100")
        self.parameter_prominence_entry = tk.Entry(self.mea_parameters_frame, text=self.parameter_prominence_val, width=8)
        self.parameter_prominence_entry.grid(row=1, column=3, padx=5, pady=2)

        # Peak width label, entry field, trace and positioning.
        self.parameter_width_label = tk.Label(self.mea_parameters_frame, text="Peak Width", bg="white", wraplength=100)
        self.parameter_width_label.grid(row=0, column=4, padx=5, pady=2)
        self.parameter_width_val = tk.StringVar()
        self.parameter_width_val.trace_add("write", self.parameter_width_callback)
        self.parameter_width_val.set("3")
        self.parameter_width_entry = tk.Entry(self.mea_parameters_frame, text=self.parameter_width_val, width=8)
        self.parameter_width_entry.grid(row=1, column=4, padx=5, pady=2)

        # Peak threshold label, entry field, trace and positioning.
        self.parameter_thresh_label = tk.Label(self.mea_parameters_frame, text="Peak Threshold", bg="white", wraplength=100)
        self.parameter_thresh_label.grid(row=0, column=5, padx=5, pady=2)
        self.parameter_thresh_val = tk.StringVar()
        self.parameter_thresh_val.trace_add("write", self.parameter_thresh_callback)
        self.parameter_thresh_val.set("50")
        self.parameter_thresh_entry = tk.Entry(self.mea_parameters_frame, text=self.parameter_thresh_val, width=8)
        self.parameter_thresh_entry.grid(row=1, column=5, padx=5, pady=2)

        # ################################################# Heatmap ####################################################
        # Frame and elements for pacemaker heat map plot.
        self.mea_heatmap_frame = tk.Frame(self, width=1620, height=800, bg="white")
        self.mea_heatmap_frame.grid(row=1, column=0, padx=5, pady=5)
        self.mea_heatmap_frame.grid_propagate(False)
        self.gen_all_heatmap = FigureCanvasTkAgg(heat_map.curr_plot, self.mea_heatmap_frame)
        self.gen_all_heatmap.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
        # Beat select slider, belongs to different frame.
        self.mea_beat_select = tk.Scale(self.mea_parameters_frame, length=200, width=15, from_=1, to=20,
                                        orient="horizontal", bg="white", label="Current Beat Number")
        self.mea_beat_select.grid(row=0, column=7, rowspan=2, padx=100, pady=5)
        self.mea_beat_select.bind("<ButtonRelease-1>",
                                  lambda event: graph_all(self, heat_map, pace_maker, upstroke_vel,
                                                          local_act_time, conduction_vel, input_param))
        self.toolbar_all_heatmap_frame = tk.Frame(self.mea_parameters_frame)
        self.toolbar_all_heatmap_frame.grid(row=0, column=8, rowspan=2, padx=50, pady=5)
        self.toolbar_all_heatmap = NavigationToolbar2Tk(self.gen_all_heatmap, self.toolbar_all_heatmap_frame)

        # The following lines are for the GUI controls found in child windows when doing beat detect or the individual,
        # or solo, calculations for the respective parameter.
        self.elec_to_plot_val = tk.StringVar()
        self.elec_to_plot_val.set("1")
        self.elec_to_plot_val.trace_add("write", self.col_sel_callback)
        self.pm_solo_beat_select = None
        self.dvdt_solo_beat_select = None
        self.lat_solo_beat_select = None
        self.cv_solo_beat_select = None
        self.param_vs_dist_beat_select = None

        # # print(dir(self))

    def beat_detect_window(self, cm_beats, input_param):
        beat_detect = tk.Toplevel(self)
        beat_detect.title('Beat Detect Window')
        beat_detect.geometry('1250x850')
        beat_detect_frame = tk.Frame(beat_detect, width=1200, height=850, bg="white")
        beat_detect_frame.grid(row=0, column=0, padx=5, pady=5)
        beat_detect_frame.grid_propagate(False)
        gen_beats_fig = FigureCanvasTkAgg(cm_beats.comp_plot, beat_detect_frame)
        gen_beats_fig.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        # NavigationToolbar2Tk calls pack internally, conflicts with grid.  Workaround: establish in own frame,
        # use grid to place that frame in_side of the chosen parent frame.  This works because the parent frame is still
        # a descent of "root", which is the overarching parent of all of these GUI elements.
        gen_beats_toolbar_frame = tk.Frame(beat_detect)
        gen_beats_toolbar_frame.grid(row=4, column=0, columnspan=2, in_=beat_detect_frame)
        gen_beats_toolbar = NavigationToolbar2Tk(gen_beats_fig, gen_beats_toolbar_frame)

        # Electrode entry field to change display for plot shown.
        elec_to_plot_label = tk.Label(beat_detect_frame, text="Electrode Plotted", bg="white", wraplength=80)
        elec_to_plot_label.grid(row=0, column=0, padx=5, pady=2)
        elec_to_plot_entry = tk.Entry(beat_detect_frame, text=self.elec_to_plot_val, width=8)
        elec_to_plot_entry.grid(row=1, column=0, padx=5, pady=2)

        # Invoke graph_peaks function for plotting only.  Meant to be used after find peaks, after switching columns.
        graph_beats_button = tk.Button(beat_detect_frame, text="Graph Beats", width=15, height=3, bg="red2",
                                            command=lambda: graph_beats(self, cm_beats, input_param))
        graph_beats_button.grid(row=0, rowspan=2, column=1, padx=2, pady=2)

    def pacemaker_heatmap_window(self, cm_beats, pace_maker, heat_map, input_param):
        pm_heatmap = tk.Toplevel(self)
        pm_heatmap.title("Pacemaker Heatmap")
        pm_heatmap_frame = tk.Frame(pm_heatmap, width=1400, height=900, bg="white")
        pm_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        pm_heatmap_frame.grid_propagate(False)
        pm_heatmap_fig = FigureCanvasTkAgg(heat_map.pm_solo_plot, pm_heatmap_frame)
        pm_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.pm_solo_beat_select = tk.Scale(pm_heatmap_frame, length=200, width=15, from_=1,
                                            to=int(cm_beats.beat_count_dist_mode[0]),
                                        orient="horizontal", bg="white", label="Current Beat Number")
        self.pm_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.pm_solo_beat_select.bind("<ButtonRelease-1>",
                                  lambda event: graph_pacemaker(self, heat_map, pace_maker, input_param))

    def dvdt_heatmap_window(self, cm_beats, upstroke_vel, heat_map, input_param):
        dvdt_heatmap = tk.Toplevel(self)
        dvdt_heatmap.title("Upstroke Velocity Heatmap")
        dvdt_heatmap_frame = tk.Frame(dvdt_heatmap, width=1400, height=900, bg="white")
        dvdt_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        dvdt_heatmap_frame.grid_propagate(False)
        dvdt_heatmap_fig = FigureCanvasTkAgg(heat_map.dvdt_solo_plot, dvdt_heatmap_frame)
        dvdt_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.dvdt_solo_beat_select = tk.Scale(dvdt_heatmap_frame, length=200, width=15, from_=1,
                                              to=int(cm_beats.beat_count_dist_mode[0]),
                                              orient="horizontal", bg="white", label="Current Beat Number")
        self.dvdt_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.dvdt_solo_beat_select.bind("<ButtonRelease-1>",
                                        lambda event: graph_upstroke(self, heat_map, upstroke_vel, input_param))

    def lat_heatmap_window(self, cm_beats, local_act_time, heat_map, input_param):
        lat_heatmap = tk.Toplevel(self)
        lat_heatmap.title("Local Activation Time Heatmap")
        lat_heatmap_frame = tk.Frame(lat_heatmap, width=1400, height=900, bg="white")
        lat_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        lat_heatmap_frame.grid_propagate(False)
        lat_heatmap_fig = FigureCanvasTkAgg(heat_map.lat_solo_plot, lat_heatmap_frame)
        lat_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.lat_solo_beat_select = tk.Scale(lat_heatmap_frame, length=200, width=15, from_=1,
                                              to=int(cm_beats.beat_count_dist_mode[0]),
                                              orient="horizontal", bg="white", label="Current Beat Number")
        self.lat_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.lat_solo_beat_select.bind("<ButtonRelease-1>",
                                        lambda event: graph_local_act_time(self, heat_map, local_act_time, input_param))

    def cv_heatmap_window(self, cm_beats, local_act_time, conduction_vel, heat_map, input_param):
        cv_heatmap = tk.Toplevel(self)
        cv_heatmap.title("Conduction Velocity Heatmap")
        cv_heatmap_frame = tk.Frame(cv_heatmap, width=1400, height=900, bg="white")
        cv_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        cv_heatmap_frame.grid_propagate(False)
        cv_heatmap_fig = FigureCanvasTkAgg(heat_map.cv_solo_plot, cv_heatmap_frame)
        cv_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.cv_solo_beat_select = tk.Scale(cv_heatmap_frame, length=200, width=15, from_=1,
                                            to=int(cm_beats.beat_count_dist_mode[0]),
                                            orient="horizontal", bg="white", label="Current Beat Number")
        self.cv_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.cv_solo_beat_select.bind("<ButtonRelease-1>",
                                      lambda event: graph_conduction_vel(self, heat_map, local_act_time, conduction_vel, input_param))

    def param_vs_dist_stats_window(self, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats):
        param_vs_dist= tk.Toplevel(self)
        param_vs_dist.title("Parameter vs Distance Plot w/ R-Square")
        param_vs_dist_options_frame = tk.Frame(param_vs_dist, width=1300, height=100, bg="white")
        param_vs_dist_options_frame.grid(row=0, column=0, padx=5, pady=5)
        param_vs_dist_frame = tk.Frame(param_vs_dist, width=1300, height=800, bg="white")
        param_vs_dist_frame.grid(row=1, column=0, padx=5, pady=5)
        param_vs_dist_frame.grid_propagate(False)
        param_vs_dist_fig = FigureCanvasTkAgg(cm_stats.param_vs_dist_plot, param_vs_dist_frame)
        param_vs_dist_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.param_vs_dist_beat_select = tk.Scale(param_vs_dist_frame, length=200, width=15, from_=1,
                                                  to=int(cm_beats.beat_count_dist_mode[0]), orient="horizontal", bg="white", label="Current Beat Number")
        self.param_vs_dist_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.param_vs_dist_beat_select.bind("<ButtonRelease-1>", lambda event: param_vs_distance_analysis(self, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats))

    def col_sel_callback(self, *args):
        print("You entered: \"{}\"".format(self.elec_to_plot_val.get()))
        try:
            chosen_electrode_val = int(self.elec_to_plot_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def min_peak_dist_callback(self, *args):
        print("You entered: \"{}\"".format(self.min_peak_dist_val.get()))
        try:
            chosen_electrode_val = int(self.min_peak_dist_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def min_peak_height_callback(self, *args):
        print("You entered: \"{}\"".format(self.min_peak_height_val.get()))
        try:
            chosen_electrode_val = int(self.min_peak_height_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_prominence_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_prominence_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_prominence_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_width_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_width_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_width_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_thresh_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_thresh_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_thresh_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")


def main():
    raw_data = ImportedData()
    cm_beats = BeatAmplitudes()
    pace_maker = PacemakerData()
    upstroke_vel = UpstrokeVelData()
    local_act_time = LocalATData()
    conduction_vel = CondVelData()
    input_param = InputParameters()
    heat_map = MEAHeatMaps()
    cm_stats = StatisticsData()

    # Heatmap axes for Calculate All (main window)
    heat_map.curr_plot = plt.Figure(figsize=(13, 6.5), dpi=120)
    heat_map.axis1 = heat_map.curr_plot.add_subplot(221)
    heat_map.axis2 = heat_map.curr_plot.add_subplot(222)
    heat_map.axis3 = heat_map.curr_plot.add_subplot(223)
    heat_map.axis4 = heat_map.curr_plot.add_subplot(224)

    # Heatmap axis for only Pacemaker window
    heat_map.pm_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.pm_solo_axis = heat_map.pm_solo_plot.add_subplot(111)

    # Heatmap axis for only Upstroke Velocity window
    heat_map.dvdt_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.dvdt_solo_axis = heat_map.dvdt_solo_plot.add_subplot(111)

    # Heatmap axis for only Local Activation Time window
    heat_map.lat_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.lat_solo_axis = heat_map.lat_solo_plot.add_subplot(111)

    # Heatmap axis for only Conduction Velocity window
    heat_map.cv_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.cv_solo_axis = heat_map.cv_solo_plot.add_subplot(111)

    # Subplot axes for Peak Finder (Beats) window
    cm_beats.comp_plot = plt.Figure(figsize=(10.5, 6), dpi=120)
    cm_beats.comp_plot.suptitle("Comparisons of find_peaks methodologies")
    cm_beats.axis1 = cm_beats.comp_plot.add_subplot(221)
    cm_beats.axis2 = cm_beats.comp_plot.add_subplot(222)
    cm_beats.axis3 = cm_beats.comp_plot.add_subplot(223)
    cm_beats.axis4 = cm_beats.comp_plot.add_subplot(224)

    cm_stats.param_vs_dist_plot = plt.Figure(figsize=(10.5, 6), dpi=120)
    cm_stats.param_vs_dist_axis_pm = cm_stats.param_vs_dist_plot.add_subplot(221)
    cm_stats.param_vs_dist_axis_lat = cm_stats.param_vs_dist_plot.add_subplot(223)
    cm_stats.param_vs_dist_axis_dvdt = cm_stats.param_vs_dist_plot.add_subplot(222)
    cm_stats.param_vs_dist_axis_cv = cm_stats.param_vs_dist_plot.add_subplot(224)

    root = tk.Tk()
    # Dimensions width x height, distance position from right of screen + from top of screen
    # root.geometry("2700x1000+900+900")

    # Calls class to create the GUI window. *********
    elecGUI120 = ElecGUI120(root, raw_data, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, heat_map, cm_stats)
    # print(vars(elecGUI120))
    # print(dir(elecGUI120))
    # print(hasattr(elecGUI120 "elec_to_plot_entry"))
    root.mainloop()


main()
