# Designed to run on Python 3.6 or newer.  Programmed under Python 3.8.
# Biggest known issues with versions earlier than 3.6: use of dictionary to contain electrode coordinates.
# Consider using an OrderedDict instead if running under earlier versions of Python.  Second issue: tkinter vs Tkinter

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import pandas as pd
import seaborn as seab
import os
import time
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks
from scipy import stats

# Fills in NaN values with whatever you specify
# print("NaN replaced with 0.0123456")
# print(data_three.fillna(0.123456))
#
# print(data_three.fillna(method='pad'))

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


class ElectrodeConfig:
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


# Import data files.  Files must be in .txt or .csv format.  May add toggles or checks to support more data types.
def data_import(raw_data):
    data_filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("txt files", "*.txt"), ("all files", "*.*")))

    start_time = time.process_time()

    # Checks whether data was previously imported into program.  If True, the previous data is deleted.
    if hasattr(raw_data, 'imported') is True:
        print("Raw data is not empty; clearing before reading file.")
        delattr(raw_data, 'imported')

    print("Importing data...")
    # Import electrodes for column headers from file.
    raw_data.names = pd.read_csv(data_filename, sep="\s+\t", lineterminator='\n', skiprows=[0, 1, 3], header=None,
                                 nrows=1, encoding='iso-8859-15', skipinitialspace=True, engine='python')
    # Import data from file.
    raw_data.imported = pd.read_csv(data_filename,
                                     sep='\s+', lineterminator='\n', skiprows=3, header=0,
                                     encoding='iso-8859-15', skipinitialspace=True, low_memory=False)

    new_data_size = np.shape(raw_data.imported)
    print(new_data_size)
    end_time = time.process_time()
    print(end_time - start_time)
    print("Import complete.")
    return raw_data.imported


# Finds peaks based on given input parameters.
def determine_beats(elecGUI60, raw_data, cm_beats, input_param):
    try:
        print("Finding beats...\n")
        start_time = time.process_time()

        if hasattr(cm_beats, 'x_axis') is True:
            print("Beat data are not empty; clearing before finding peaks.")
            delattr(cm_beats, 'x_axis')
            delattr(cm_beats, 'dist_beats')
            delattr(cm_beats, 'prom_beats')
            delattr(cm_beats, 'width_beats')
            delattr(cm_beats, 'thresh_beats')

        cm_beats.x_axis = raw_data.imported.iloc[0:, 0]
        # y_axis indexing ends at column -1, or second to last column, to remove the columns containing only \r
        cm_beats.y_axis = raw_data.imported.iloc[0:, 1:-1]
        print("Y-axis data type is:: " + str(type(cm_beats.y_axis)) + "\n")
        print("Number of columns in cm_beats.y_axis: " + str(len(cm_beats.y_axis.columns)) + "\n")

        input_param.elec_choice = int(elecGUI60.elec_to_plot_val.get()) - 1
        input_param.min_peak_dist = float(elecGUI60.min_peak_dist_val.get())
        input_param.min_peak_height = float(elecGUI60.min_peak_height_val.get())
        input_param.parameter_prominence = float(elecGUI60.parameter_prominence_val.get())
        input_param.parameter_width = float(elecGUI60.parameter_width_val.get())
        input_param.parameter_thresh = float(elecGUI60.parameter_thresh_val.get())

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
        # Not currently equipped to handle datasets with dual-recorded MEAs (e.g. dual MEA60s)
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
        # print(cm_beats.dist_beats)
        # print("Data type of cm_beats.dist_beats: " + str(type(cm_beats.dist_beats)))

        prom_beats_size = np.shape(cm_beats.prom_beats)
        print("Shape of cm_beats.prom_beats: " + str(prom_beats_size))
        # print(cm_beats.prom_beats)
        # print("Data type of cm_beats.prom_beats: " + str(type(cm_beats.prom_beats)) + "\n")

        width_beats_size = np.shape(cm_beats.width_beats)
        print("Shape of cm_beats.width_beats: " + str(width_beats_size) + ".\n")
        # print(cm_beats.width_beats)
        # print("Data type of cm_beats.width_beats: " + str(type(cm_beats.width_beats)) + "\n")

        thresh_beats_size = np.shape(cm_beats.thresh_beats)
        print("Shape of cm_beats.thresh_beats: " + str(thresh_beats_size) + ".\n")
        # print(cm_beats.thresh_beats)
        # print("Data type of cm_beats.thresh_beats: " + str(type(cm_beats.thresh_beats)) + "\n")

        print("Finished.")
        end_time = time.process_time()
        print(end_time - start_time)
        print("Plotting...")
        graph_beats(elecGUI60, cm_beats, input_param)
    except AttributeError:
        print("No data found. Please import data (.txt or .csv converted MCD file) first.")


# Produces 4-subplot plot of peak finder data and graphs it.  Can be called via button.
# Will throw exception of data does not exist.
def graph_beats(elecGUI60, cm_beats, input_param):
    try:
        cm_beats.axis1.cla()
        cm_beats.axis2.cla()
        cm_beats.axis3.cla()
        cm_beats.axis4.cla()

        input_param.elec_choice = int(elecGUI60.elec_to_plot_val.get()) - 1
        print("Will generate graph for electrode " + str(input_param.elec_choice + 1) + ".")
        cm_beats.comp_plot.suptitle("Comparisons of find_peaks methodologies: electrode " + (str(input_param.elec_choice + 1)) + ".")

        mask_dist = ~np.isnan(cm_beats.dist_beats.iloc[0:, input_param.elec_choice].values)
        dist_without_nan = cm_beats.dist_beats.iloc[0:, input_param.elec_choice].values[mask_dist].astype('int64')
        cm_beats.axis1.plot(dist_without_nan, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values[dist_without_nan], "xr")
        cm_beats.axis1.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
        cm_beats.axis1.legend(['distance = ' + str(elecGUI60.min_peak_dist_val.get())], loc='lower left')

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


def calculate_pacemaker(elecGUI60, cm_beats, pace_maker):
    try:
        start_time = time.process_time()
        pace_maker.param_dist_raw = pd.DataFrame()
        pace_maker.param_prom_raw = pd.DataFrame()
        pace_maker.param_width_raw = pd.DataFrame()
        pace_maker.param_thresh_raw = pd.DataFrame()

        for column in range(len(cm_beats.dist_beats.columns)):
            if cm_beats.beat_count_dist_mode[0] == len(cm_beats.dist_beats.iloc[0:, column].dropna()):
                # print(len(cm_beats.dist_beats.iloc[0:, column].dropna()))
                pace_maker_dist_raw = pd.Series(cm_beats.dist_beats.iloc[0:, column], name=column+1)
                pace_maker.param_dist_raw = pd.concat([pace_maker.param_dist_raw, pace_maker_dist_raw], axis='columns')
            else:
                # print(len(cm_beats.dist_beats.iloc[0:, column].dropna()))
                pace_maker_dist_raw = pd.Series(name=column+1)
                pace_maker.param_dist_raw = pd.concat([pace_maker.param_dist_raw, pace_maker_dist_raw], axis='columns')

        for column in range(len(cm_beats.prom_beats.columns)):
            if cm_beats.beat_count_prom_mode[0] == len(cm_beats.prom_beats.iloc[0:, column].dropna()):
                # print(len(cm_beats.prom_beats.iloc[0:, column].dropna()))
                pace_maker_prom_raw = pd.Series(cm_beats.prom_beats.iloc[0:, column], name=column+1)
                pace_maker.param_prom_raw = pd.concat([pace_maker.param_prom_raw, pace_maker_prom_raw], axis='columns')
            else:
                # print(len(cm_beats.prom_beats.iloc[0:, column].dropna()))
                pace_maker_prom_raw = pd.Series(name=column+1)
                pace_maker.param_prom_raw = pd.concat([pace_maker.param_prom_raw, pace_maker_prom_raw], axis='columns')

        for column in range(len(cm_beats.width_beats.columns)):
            if cm_beats.beat_count_prom_mode[0] == len(cm_beats.width_beats.iloc[0:, column].dropna()):
                # print(len(cm_beats.width_beats.iloc[0:, column].dropna()))
                pace_maker_width_raw = pd.Series(cm_beats.width_beats.iloc[0:, column], name=column+1)
                pace_maker.param_width_raw = pd.concat([pace_maker.param_width_raw, pace_maker_width_raw], axis='columns')
            else:
                # print(len(cm_beats.width_beats.iloc[0:, column].dropna()))
                pace_maker_width_raw = pd.Series(name=column+1)
                pace_maker.param_width_raw = pd.concat([pace_maker.param_width_raw, pace_maker_width_raw], axis='columns')

        for column in range(len(cm_beats.thresh_beats.columns)):
            if cm_beats.beat_count_thresh_mode[0] == len(cm_beats.thresh_beats.iloc[0:, column].dropna()):
                # print(len(cm_beats.width_beats.iloc[0:, column].dropna()))
                pace_maker_thresh_raw = pd.Series(cm_beats.thresh_beats.iloc[0:, column], name=column+1)
                pace_maker.param_thresh_raw = pd.concat([pace_maker.param_thresh_raw, pace_maker_thresh_raw], axis='columns')
            else:
                # print(len(cm_beats.width_beats.iloc[0:, column].dropna()))
                pace_maker_thresh_raw = pd.Series(name=column+1)
                pace_maker.param_thresh_raw = pd.concat([pace_maker.param_thresh_raw, pace_maker_thresh_raw], axis='columns')

        pace_maker.param_dist_normalized = pace_maker.param_dist_raw.sub(pace_maker.param_dist_raw.min(axis=1), axis=0)
        pace_maker.param_prom_normalized = pace_maker.param_prom_raw.sub(pace_maker.param_prom_raw.min(axis=1), axis=0)
        pace_maker.param_width_normalized = pace_maker.param_width_raw.sub(pace_maker.param_width_raw.min(axis=1), axis=0)
        pace_maker.param_thresh_normalized = pace_maker.param_thresh_raw.sub(pace_maker.param_thresh_raw.min(axis=1), axis=0)
        # print(pace_maker.param_dist_raw.min(axis=1))
        print("Done.")
        end_time = time.process_time()
        print(end_time - start_time)
    except AttributeError:
        print("Please use Find Peaks first.")


def data_print(elecGUI60, raw_data):
    # adding .iloc to a data frame allows to reference [row, column], where rows and columns can be ranges separated
    # by colons
    print(id(raw_data.imported))
    print(elecGUI60.elec_to_plot_val.get())
    print(raw_data.names)
    print(raw_data.imported.iloc[0:10, 0:15])
    print(raw_data.imported.iloc[0:10, 110:])


def graph_heatmap(vegetables, farmers, harvest, heat_map, axis1):
    # imshow() is the key heatmap function here.
    axis1.cla()
    im = axis1.imshow(harvest, interpolation="nearest", aspect="auto", cmap="jet")

    cbar = axis1.figure.colorbar(im)
    cbar.remove()

    cbar = axis1.figure.colorbar(im)
    cbar.ax.set_ylabel("Harvested Crops (t/year)", rotation=-90, va="bottom")

    axis1.set_xticks(np.arange(len(farmers)))
    axis1.set_yticks(np.arange(len(vegetables)))
    axis1.set_xticklabels(farmers)
    axis1.set_yticklabels(vegetables)
    plt.setp(axis1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = axis1.text(j, i, harvest[i, j], ha="center", va="center", color="w")
    axis1.set_title("Harvest Demo from Matplotlib Website")
    heat_map.tight_layout()
    heat_map.canvas.draw()
    return


class ElecGUI60(tk.Frame):
    def __init__(self, master, raw_data, cm_beats, pace_maker, input_param):
        tk.Frame.__init__(self, master)
        # The fun story about grid: columns and rows cannot be generated past the number of widgets you have (or at
        # least I've yet to learn the way to do so, and will update if I find out how).  It's all relative geometry,
        # where the relation is with other widgets.  If you have 3 widgets you can have 3 rows or 3 columns and
        # organize accordingly.  If you have 2 widgets you can only have 2 rows or 2 columns.
        # use .bind("<Enter>", "color") or .bind("<Leave>", "color") to change mouse-over color effects.
        self.grid()
        self.winfo_toplevel().title("Elec GUI 60 - Prototype")

        # ############################################### Buttons #####################################################
        self.file_operations = tk.Frame(self, width=100, height=800, bg="white")
        self.file_operations.grid(row=1, column=0, padx=10, pady=10, sticky="nw")
        self.import_data_button = tk.Button(self.file_operations, text="Import Data", width=15, height=3, bg="skyblue",
                                            command=lambda: data_import(raw_data))
        self.import_data_button.grid(row=0, column=0, padx=2, pady=2)

        # to save data; not implemented.
        self.save_data_button = tk.Button(self.file_operations, text="Save Data", width=15, height=3, bg="lightgreen")
        self.save_data_button.grid(row=1, column=0, padx=2, pady=2)

        # prints data from import; eventually test to specify columns and rows.
        self.print_data_button = tk.Button(self.file_operations, text="Print Data", width=15, height=3, bg="yellow",
                                           command=lambda: data_print(self, raw_data))
        self.print_data_button.grid(row=2, column=0, padx=2, pady=2)

        # Invoke peak finder (beats) for data. Calls to function determine_beats, which is external to class ElecGUI60
        self.calc_peaks_button = tk.Button(self.file_operations, text="Find Peaks", width=15, height=3, bg="orange",
                                           command=lambda: determine_beats(self, raw_data, cm_beats, input_param))
        self.calc_peaks_button.grid(row=3, column=0, padx=2, pady=2)

        # Invoke calculate pacemaker function, using data acquired from find_peaks function (contained in cm_beats)
        self.calc_pacemaker_button = tk.Button(self.file_operations, text="Calculate PM", width=15, height=3, bg="light coral",
                                               command=lambda: calculate_pacemaker(self, cm_beats, pace_maker))
        self.calc_pacemaker_button.grid(row=4, column=0, padx=2, pady=2)

        # Invoke graph_peaks function for plotting only.  Meant to be used after find peaks, after switching columns.
        self.graph_beats_button = tk.Button(self.file_operations, text="Graph Beats", width=15, height=3, bg="tomato",
                                            command=lambda: graph_beats(self, cm_beats, input_param))
        self.graph_beats_button.grid(row=5, column=0, padx=2, pady=2)

        # to generate heatmap; currently only generates for Matplotlib demo data.
        self.graph_heatmap_button = tk.Button(self.file_operations, text="Graph Heatmap", width=15, height=3, bg="red",
                                              command=lambda: graph_heatmap(vegetables, farmers, harvest, heat_map, axis1))
        self.graph_heatmap_button.grid(row=6, column=0, padx=2, pady=2)

        # Frame for MEA parameters (e.g. plotted electrode, min peak distance, min peak amplitude, prominence, etc)
        self.mea_parameters_frame = tk.Frame(self, width=2420, height=100, bg="white")
        self.mea_parameters_frame.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
        self.mea_parameters_frame.grid_propagate(False)

        # GUI elements in the mea_parameters_frame
        self.elec_to_plot_label = tk.Label(self.mea_parameters_frame, text="Electrode Plotted", bg="white", wraplength=80)
        self.elec_to_plot_label.grid(row=0, column=0, padx=5, pady=2)
        self.elec_to_plot_val = tk.StringVar()
        self.elec_to_plot_val.trace_add("write", self.col_sel_callback)
        self.elec_to_plot_val.set("1")
        self.elec_to_plot_entry = tk.Entry(self.mea_parameters_frame, text=self.elec_to_plot_val, width=8)
        self.elec_to_plot_entry.grid(row=1, column=0, padx=5, pady=2)

        # ############################################### Entry Fields ################################################
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

        # ############################################### Plot Frames #################################################
        # Frame and elements for heat map plot.
        self.mea_array_frame = tk.Frame(self, width=1200, height=800, bg="white")
        self.mea_array_frame.grid(row=1, column=1, padx=10, pady=10)
        self.mea_array_frame.grid_propagate(False)
        self.gen_figure = FigureCanvasTkAgg(heat_map, self.mea_array_frame)
        self.gen_figure.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        # Frame and elements for peak finder plots.
        self.beat_detect_frame = tk.Frame(self, width=1200, height=800, bg="white")
        self.beat_detect_frame.grid(row=1, column=2, padx=10, pady=10)
        self.beat_detect_frame.grid_propagate(False)
        self.gen_beats = FigureCanvasTkAgg(cm_beats.comp_plot, self.beat_detect_frame)
        self.gen_beats.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        # NavigationToolbar2Tk calls pack internally, conflicts with grid.  Workaround: establish in own frame,
        # use grid to place that frame in_side of the chosen parent frame.  This works because the parent frame is still
        # a descent of "root", which is the overarching parent of all of these GUI elements.
        self.gen_beats_toolbar_frame = tk.Frame(self)
        self.gen_beats_toolbar_frame.grid(row=2, column=0, in_=self.beat_detect_frame)
        self.gen_beats_toolbar = NavigationToolbar2Tk(self.gen_beats, self.gen_beats_toolbar_frame)

        # print(dir(self))
    def col_sel_callback(self, *args):
        print("You entered: \"{}\"".format(self.elec_to_plot_val.get()))
        try:
            chosen_electrode_val = int(self.elec_to_plot_val.get())
            # print(type(chosen_electrode_val))
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def min_peak_dist_callback(self, *args):
        print("You entered: \"{}\"".format(self.min_peak_dist_val.get()))
        try:
            chosen_electrode_val = int(self.min_peak_dist_val.get())
            # print(type(chosen_electrode_val))
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def min_peak_height_callback(self, *args):
        print("You entered: \"{}\"".format(self.min_peak_height_val.get()))
        try:
            chosen_electrode_val = int(self.min_peak_height_val.get())
            # print(type(chosen_electrode_val))
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_prominence_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_prominence_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_prominence_val.get())
            # print(type(chosen_electrode_val))
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_width_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_width_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_width_val.get())
            # print(type(chosen_electrode_val))
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_thresh_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_thresh_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_thresh_val.get())
            # print(type(chosen_electrode_val))
        except ValueError:
            print("Only numbers are allowed.  Please try again.")


def main():
    # Taken from matplotlib website for the sake of testing out a heatmap.  Still trying to figure out how to properly
    # integrate this into a GUI.
    global vegetables
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]

    global farmers
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    global harvest
    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    global heat_map
    heat_map = plt.Figure(figsize=(10, 6), dpi=120)
    global axis1
    axis1 = heat_map.add_subplot(111)

    raw_data = ImportedData()
    cm_beats = BeatAmplitudes()
    pace_maker = PacemakerData()
    input_param = InputParameters()

    cm_beats.comp_plot = plt.Figure(figsize=(10.5, 6), dpi=120)
    cm_beats.comp_plot.suptitle("Comparisons of find_peaks methodologies")
    cm_beats.axis1 = cm_beats.comp_plot.add_subplot(221)
    cm_beats.axis2 = cm_beats.comp_plot.add_subplot(222)
    cm_beats.axis3 = cm_beats.comp_plot.add_subplot(223)
    cm_beats.axis4 = cm_beats.comp_plot.add_subplot(224)

    # print("Import data memory reference: " + str(id(import_raw_data)))

    root = tk.Tk()
    # Dimensions width x height, distance position from right of screen + from top of screen
    root.geometry("2700x1200+900+900")

    # Calls class to create the GUI window. *********
    elecGUI60 = ElecGUI60(root, raw_data, cm_beats, pace_maker, input_param)
    # print(vars(ElecGUI60))
    # print(dir(elecGUI60))
    # print(hasattr(elecGUI60, "elec_to_plot_entry"))
    root.mainloop()


main()
