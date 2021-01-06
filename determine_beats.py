# Author: Christopher S. Dunham
# 12/4/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import time
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats


# Finds peaks based on given input parameters.
def determine_beats(elecGUI120, raw_data, cm_beats, input_param):
    try:
        print("Finding beats...\n")
        start_time = time.process_time()

        if hasattr(cm_beats, 'x_axis') is True:
            print("Beat data are not empty; clearing before finding beats.")
            delattr(cm_beats, 'x_axis')
            delattr(cm_beats, 'y_axis')
            delattr(cm_beats, 'dist_beats')
            delattr(cm_beats, 'prom_beats')
            delattr(cm_beats, 'width_beats')
            delattr(cm_beats, 'thresh_beats')

        input_param.elec_choice = int(elecGUI120.elec_to_plot_val.get()) - 1
        input_param.min_peak_dist = float(elecGUI120.min_peak_dist_val.get())
        input_param.min_peak_height = float(elecGUI120.min_peak_height_val.get())
        input_param.parameter_prominence = float(elecGUI120.parameter_prominence_val.get())
        input_param.parameter_width = float(elecGUI120.parameter_width_val.get())
        input_param.parameter_thresh = float(elecGUI120.parameter_thresh_val.get())
        input_param.sample_frequency = float(elecGUI120.sample_frequency_val.get())

        # file_length = # of rows / sample_freq --> # of seconds in data set 
        # multiply by 60s in 1 min --> number of minutes in data set (rounded)
        file_length = ((len(raw_data.imported.index) / 
            input_param.sample_frequency) / 60)
        elecGUI120.file_length_label.configure(text=str(round(file_length, 2)) +
            " minutes")

        if elecGUI120.trunc_toggle_on_off.get() == False:
            print("Calculating using full data set.")
            cm_beats.x_axis = raw_data.imported.iloc[0:, 0]
            # y_axis indexing ends at column -1, or second to last column, to 
            # remove the columns containing only \r
            if '\r' in raw_data.imported.columns:
                cm_beats.y_axis = raw_data.imported.iloc[0:, 1:-1]
            else:
                cm_beats.y_axis = raw_data.imported.iloc[0:, 1:]
        elif elecGUI120.trunc_toggle_on_off.get() == True:
            trunc_start = int(elecGUI120.trunc_start_value.get())
            trunc_end = int(elecGUI120.trunc_end_value.get())
            print("Truncating between {} and {} minutes.".format(
                trunc_start, trunc_end))
            
            start_milsec = int(trunc_start * 60 * input_param.sample_frequency)
            end_milsec = int(trunc_end * 60 * input_param.sample_frequency)
            print("Start (ms) = {}, End (ms) = {}".format(
                start_milsec, end_milsec))
            cm_beats.x_axis = raw_data.imported.iloc[start_milsec:end_milsec, 0]
            # y_axis indexing ends at column -1, or second to last column, to 
            # remove the columns containing only \r
            if '\r' in raw_data.imported.columns:
                cm_beats.y_axis = raw_data.imported.iloc[start_milsec:end_milsec, 
                1:-1]
            else:
                cm_beats.y_axis = raw_data.imported.iloc[start_milsec:end_milsec, 
                1:]

        # print("Y-axis data type is:: " + str(type(cm_beats.y_axis)) + "\n")
        print("Number of columns in cm_beats.y_axis: " + str(len(cm_beats.y_axis.columns)) + "\n")

        # Establish "strucs" as dataframes for subsequent operations.
        cm_beats.dist_beats = pd.DataFrame()
        cm_beats.prom_beats = pd.DataFrame()
        cm_beats.width_beats = pd.DataFrame()
        cm_beats.thresh_beats = pd.DataFrame()

        print("Summary of parameters: " + str(input_param.min_peak_dist) + ", " + str(input_param.min_peak_height) +
              ", " + str(input_param.parameter_prominence) + ", " + str(input_param.parameter_width) + ", " +
              str(input_param.parameter_thresh) + ".\n")

        # For loops for finding beats (peaks) in each channel (electrode).  
        # Suitable for any given MCD-converted file in which only one MEA is 
        # recorded (i.e. works for a single 120 electrode MEA)
        # Caveat 1: have not tested for singular MEA60 data.
        # Disclaimer: Not currently equipped to handle datasets with 
        # dual-recorded MEAs (e.g. dual MEA60s)
        for column in range(len(cm_beats.y_axis.columns)):
            dist_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist)[0], name=column+1)
            cm_beats.dist_beats = pd.concat([cm_beats.dist_beats, dist_beats], 
                axis='columns')

            prom_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist, 
                prominence=input_param.parameter_prominence)[0], name=column+1)
            cm_beats.prom_beats = pd.concat([cm_beats.prom_beats, prom_beats], 
                axis='columns')

            width_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist, 
                width=input_param.parameter_width)[0], name=column+1)
            cm_beats.width_beats = pd.concat([cm_beats.width_beats, width_beats], 
                axis='columns')

            thresh_beats = pd.Series(find_peaks(cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist, 
                threshold=input_param.parameter_thresh)[0], name=column+1)
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
        print("Shape of cm_beats.width_beats: " + str(width_beats_size))
        thresh_beats_size = np.shape(cm_beats.thresh_beats)
        print("Shape of cm_beats.thresh_beats: " + str(thresh_beats_size) + ".\n")

        print("Finished.")
        end_time = time.process_time()
        print(end_time - start_time)
        print("Plotting...")
    except AttributeError:
        print("No data found. Please import data (.txt or .csv converted MCD file) first.")

# Produces 4-subplot plot of peak finder data and graphs it.  Can be called via 
# button. Will throw exception of data does not exist.
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