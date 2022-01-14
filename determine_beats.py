# Author: Christopher S. Dunham
# 12/4/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import time
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy import stats


# Finds peaks based on given input parameters.
def determine_beats(analysisGUI, raw_data, cm_beats, input_param, 
electrode_config, batch_data):
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

        # Check whether batch analysis is being performed.
        # If not (i.e. normal mode/single file), pull parameters from GUI
        if batch_data.batch_config == False:
            input_param.min_peak_dist = float(
                analysisGUI.pkDistEdit.text())
            input_param.min_peak_height = float(
                analysisGUI.pkHeightEdit.text())
            input_param.parameter_prominence = float(
                analysisGUI.pkPromEdit.text())
            input_param.parameter_width = float(
                analysisGUI.pkWidthEdit.text())
            input_param.parameter_thresh = float(
                analysisGUI.pkThreshEdit.text())
            input_param.sample_frequency = float(
                analysisGUI.sampleFreqEdit.currentText())

        # file_length = # of rows / sample_freq --> # of seconds in data set 
        # multiply by 60s in 1 min --> number of minutes in data set (rounded)
        analysisGUI.file_length = ((len(raw_data.imported.index) / 
            input_param.sample_frequency) / 60)

        analysisGUI.fileLength.setText(
            str(round(analysisGUI.file_length, 2)) + " minutes")

        # Check if using batch processing.
        # If not, get truncation status & endpoints from GUI
        if batch_data.batch_config == False:
            if analysisGUI.truncCheckbox.isChecked() == False:
                print("Calculating using full data set.")
                cm_beats.x_axis = raw_data.imported.iloc[0:, 0]
                # y_axis indexing ends at column -1, or second to last column, to 
                # remove the columns containing only \r
                if '\r' in raw_data.imported.columns:
                    cm_beats.y_axis = raw_data.imported.iloc[0:, 1:-1]
                else:
                    cm_beats.y_axis = raw_data.imported.iloc[0:, 1:]
            elif analysisGUI.truncCheckbox.isChecked() == True:
                trunc_start = float(analysisGUI.truncStartEdit.text())
                trunc_end = float(analysisGUI.truncEndEdit.text())
                print("Truncating between {} and {} minutes.".format(
                    trunc_start, trunc_end))
                
                start_milsec = int(
                    trunc_start * 60 * input_param.sample_frequency)
                end_milsec = int(
                    trunc_end * 60 * input_param.sample_frequency)
                print("Start (ms) = {}, End (ms) = {}".format(
                    start_milsec, end_milsec))
                cm_beats.x_axis = (
                    raw_data.imported.iloc[start_milsec:end_milsec, 0].values 
                    - raw_data.imported.iloc[
                        start_milsec:end_milsec, 0].min().min())
                # y_axis indexing ends at column -1, or second to last column, to 
                # remove the columns containing only \r
                if '\r' in raw_data.imported.columns:
                    cm_beats.y_axis = raw_data.imported.iloc[
                    start_milsec:end_milsec, 
                    1:-1]
                    cm_beats.y_axis.reset_index(drop=True, inplace=True)
                else:
                    cm_beats.y_axis = raw_data.imported.iloc[
                    start_milsec:end_milsec, 
                    1:]
                    cm_beats.y_axis.reset_index(drop=True, inplace=True)
        
        # If using batch, use batch file values for trunc status & endpoints
        elif batch_data.batch_config == True:
            if input_param.toggle_trunc == False:
                print("Batch processing. Calculating using full data set.")
                cm_beats.x_axis = raw_data.imported.iloc[0:, 0]
                # y_axis indexing ends at column -1, or second to last column, to 
                # remove the columns containing only \r
                if '\r' in raw_data.imported.columns:
                    cm_beats.y_axis = raw_data.imported.iloc[0:, 1:-1]
                else:
                    cm_beats.y_axis = raw_data.imported.iloc[0:, 1:]

            elif input_param.toggle_trunc == True:
                trunc_start = float(input_param.trunc_start)
                trunc_end = float(input_param.trunc_end)
                print("Batch processing. " + 
                    "Truncating between {} and {} minutes.".format(
                    trunc_start, trunc_end))
                
                start_milsec = int(
                    trunc_start * 60 * input_param.sample_frequency)
                end_milsec = int(
                    trunc_end * 60 * input_param.sample_frequency)
                print("Start (ms) = {}, End (ms) = {}".format(
                    start_milsec, end_milsec))
                cm_beats.x_axis = (
                    raw_data.imported.iloc[start_milsec:end_milsec, 0].values 
                    - raw_data.imported.iloc[
                        start_milsec:end_milsec, 0].min().min())
                # y_axis indexing ends at column -1, or second to last column,
                # to remove the columns containing only \r
                if '\r' in raw_data.imported.columns:
                    cm_beats.y_axis = raw_data.imported.iloc[
                        start_milsec:end_milsec, 1:-1]
                    cm_beats.y_axis.reset_index(drop=True, inplace=True)
                else:
                    cm_beats.y_axis = raw_data.imported.iloc[
                        start_milsec:end_milsec, 1:]
            
        # Checks for whether Butterworth filter is selected.  If so, runs the
        # appropriate operations for the given selection.  Needs to filter per
        # column in cm_beats.y_axis
        if batch_data.batch_config == False:
            if analysisGUI.beatsWindow.filterTypeEdit.currentText() == "No filter":
                print("No filter.")

            elif analysisGUI.beatsWindow.filterTypeEdit.currentText() == "Low-pass Only":
                bworth_ord = int(
                    analysisGUI.beatsWindow.butterOrderEdit.text())
                low_cutoff_freq = float(
                    analysisGUI.beatsWindow.lowPassFreqEdit.text())
                print("Low-pass filter." + 
                    " Order = {}, Low Cutoff Freq. = {}".format(
                    bworth_ord, low_cutoff_freq))

                sos_low = butter(
                    bworth_ord, 
                    low_cutoff_freq, 
                    btype='low', 
                    output='sos', 
                    fs=input_param.sample_frequency)
                filtered_low = np.zeros(
                    (len(cm_beats.y_axis.index), len(cm_beats.y_axis.columns)))
                for col, column in enumerate(cm_beats.y_axis.columns):
                    filtered_low[:, col] = sosfilt(sos_low, 
                        cm_beats.y_axis[column])
                cm_beats.y_axis = pd.DataFrame(filtered_low)
            
            elif analysisGUI.beatsWindow.filterTypeEdit.currentText() == "High-pass Only":
                bworth_ord = int(
                    analysisGUI.beatsWindow.butterOrderEdit.text())
                high_cutoff_freq = float(
                    analysisGUI.beatsWindow.highPassFreqEdit.text())
                print("High-pass filter." + 
                    " Order = {}, High Cutoff Freq = {}".format(
                    bworth_ord, high_cutoff_freq))

                sos_high = butter(
                    bworth_ord,
                    high_cutoff_freq, 
                    btype='high', 
                    output='sos', 
                    fs=input_param.sample_frequency)
                filtered_high = np.zeros(
                    (len(cm_beats.y_axis.index), len(cm_beats.y_axis.columns)))
                for col, column in enumerate(cm_beats.y_axis.columns):
                    filtered_high[:, col] = sosfilt(sos_high, 
                        cm_beats.y_axis[column])
                cm_beats.y_axis = pd.DataFrame(filtered_high)
            
            elif analysisGUI.beatsWindow.filterTypeEdit.currentText() == "Bandpass":
                bworth_ord = int(
                    analysisGUI.beatsWindow.butterOrderEdit.text())
                low_cutoff_freq = float(
                    analysisGUI.beatsWindow.lowPassFreqEdit.text())
                high_cutoff_freq = float(
                    analysisGUI.beatsWindow.highPassFreqEdit.text())
                print("Bandpass filter." +  
                    " Order = {}, Low cutoff = {}, High cutoff = {}".format(
                    bworth_ord, low_cutoff_freq, high_cutoff_freq))
                
                sos_bp = butter(
                    bworth_ord, 
                    [high_cutoff_freq, low_cutoff_freq], 
                    btype='bandpass', 
                    output='sos',
                    fs = input_param.sample_frequency)
                filtered_bp = np.zeros(
                    (len(cm_beats.y_axis.index), len(cm_beats.y_axis.columns)))
                for col, column in enumerate(cm_beats.y_axis.columns):
                    filtered_bp[:, col] = sosfilt(sos_bp, 
                        cm_beats.y_axis[column])
                cm_beats.y_axis = pd.DataFrame(filtered_bp)

        # print("Y-axis data type is:: " + str(type(cm_beats.y_axis)) + "\n")
        print("Number of columns in cm_beats.y_axis: " + 
            str(len(cm_beats.y_axis.columns)))
        print("Number of rows in cm_beats.y_axis: " + 
            str(len(cm_beats.y_axis)) + "\n")

        if batch_data.batch_config == False:
            analysisGUI.beatsWindow.paramSlider1a.setMaximum(
                len(cm_beats.y_axis.columns) - 1)

        # Establish "fields" as dataframes for subsequent operations.
        cm_beats.dist_beats = pd.DataFrame()
        cm_beats.prom_beats = pd.DataFrame()
        cm_beats.width_beats = pd.DataFrame()
        cm_beats.thresh_beats = pd.DataFrame()
        # Dataframe for negative amplitudes
        # Applied only for dist_beats as it is the primary detection method
        cm_beats.negative_dist_beats = pd.DataFrame()

        print("Summary of parameters: " + str(input_param.min_peak_height) 
            + ", " + str(input_param.min_peak_dist) +
            ", " + str(input_param.parameter_prominence) + ", " + 
            str(input_param.parameter_width) + ", " +
            str(input_param.parameter_thresh) + ", " + 
            str(input_param.sample_frequency) + "\n")

        # Assign electrode labels to cm_beats.y_axis columns
        cm_beats.y_axis.columns = electrode_config.electrode_names

        # Manually silence selected electrodes if toggle silence is checked.
        if batch_data.batch_config == False:
            if analysisGUI.toggleSilence.isChecked() == True:
                silencedElecs = analysisGUI.elecCombobox.currentData()
                for elec in silencedElecs:
                    cm_beats.y_axis.loc[:, elec] = 0
                print("Silenced electrodes: " + str(silencedElecs))

        elif batch_data.batch_config == True:
            if input_param.toggle_silence == True:
                silencedElecs = input_param.silenced_elecs
                for elec in silencedElecs:
                    cm_beats.y_axis.loc[:, elec] = 0
                print("Batch processing. Silenced electrodes: " 
                    + str(silencedElecs))

        # Negative version of beat data, to find full beat amplitude.
        cm_beats.neg_y_axis = -1 * cm_beats.y_axis

        # For loop for finding beats (peaks) in each channel (electrode).  
        # Suitable for any given MCD-converted file in which only one MEA is 
        # recorded (i.e. works for a single 120 or 60 electrode MEA).
        # Disclaimer: Not currently equipped to handle datasets with 
        # dual-recorded MEAs (e.g. dual MEA60s)
        for column in range(len(cm_beats.y_axis.columns)):
            dist_beats = pd.Series(find_peaks(
                cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist)[0], 
                name=column+1)
            cm_beats.dist_beats = pd.concat(
                [cm_beats.dist_beats, dist_beats], 
                axis='columns')

            # Applying a different standard for negative beat detection.
            # In many MEA samples, the negative peaks can be largely lost.
            # Using a fairly generous parameter set here may help.
            # 9/28/21: Updating the previous comment, unable to find
            # a consistent, fair parameter set for this purpose.
            # Leaving in the code for negative peaks, but it's functionally
            # useless at this stage; data are not robust enough.
            neg_dist_beats = pd.Series(find_peaks(
                cm_beats.neg_y_axis.iloc[0:, column], 
                height=25,
                distance=400)[0], 
                name=column+1)
            cm_beats.negative_dist_beats = pd.concat(
                [cm_beats.negative_dist_beats, neg_dist_beats], 
                axis='columns')

            prom_beats = pd.Series(find_peaks(
                cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist, 
                prominence=input_param.parameter_prominence)[0], 
                name=column+1)
            cm_beats.prom_beats = pd.concat(
                [cm_beats.prom_beats, prom_beats], 
                axis='columns')

            width_beats = pd.Series(find_peaks(
                cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist, 
                width=input_param.parameter_width)[0], 
                name=column+1)
            cm_beats.width_beats = pd.concat(
                [cm_beats.width_beats, width_beats], 
                axis='columns')

            thresh_beats = pd.Series(find_peaks(
                cm_beats.y_axis.iloc[0:, column], 
                height=input_param.min_peak_height,
                distance=input_param.min_peak_dist, 
                threshold=input_param.parameter_thresh)[0], 
                name=column+1)
            cm_beats.thresh_beats = pd.concat(
                [cm_beats.thresh_beats, thresh_beats], 
                axis='columns')

        # Assign column name identifiers to columns of dist_beats
        cm_beats.dist_beats.columns = electrode_config.electrode_names

        # Data designation to ensure NaN values are properly handled by 
        # subsequent calculations.
        cm_beats.dist_beats.astype('float64')
        cm_beats.prom_beats.astype('float64')
        cm_beats.width_beats.astype('float64')
        cm_beats.thresh_beats.astype('float64')
        cm_beats.negative_dist_beats.astype('float64')

        # Generate beat counts for the different peakfinder methods by finding 
        # the length of each electrode (column).
        cm_beats.beat_count_dist = np.zeros(
            len(cm_beats.dist_beats.columns))
        for column in range(len(cm_beats.dist_beats.columns)):
            cm_beats.beat_count_dist[column] = len(
                cm_beats.dist_beats.iloc[0:, column].dropna(axis='index'))
        
        cm_beats.neg_beat_count_dist = np.zeros(
            len(cm_beats.negative_dist_beats.columns))
        for column in range(len(cm_beats.negative_dist_beats.columns)):
            cm_beats.neg_beat_count_dist[column] = len(
                cm_beats.negative_dist_beats.iloc[0:, column].dropna(
                    axis='index'))

        cm_beats.beat_count_prom = np.zeros(
            len(cm_beats.prom_beats.columns))
        for column in range(len(cm_beats.prom_beats.columns)):
            cm_beats.beat_count_prom[column] = len(
                cm_beats.prom_beats.iloc[0:, column].dropna(axis='index'))

        cm_beats.beat_count_width = np.zeros(
            len(cm_beats.width_beats.columns))
        for column in range(len(cm_beats.width_beats.columns)):
            cm_beats.beat_count_width[column] = len(
                cm_beats.width_beats.iloc[0:, column].dropna(axis='index'))

        cm_beats.beat_count_thresh = np.zeros(
            len(cm_beats.thresh_beats.columns))
        for column in range(len(cm_beats.thresh_beats.columns)):
            cm_beats.beat_count_thresh[column] = len(
                cm_beats.thresh_beats.iloc[0:, column].dropna(axis='index'))

        # Finds the mode of beats across the dataset for each peakfinder 
        # parameter set.
        cm_beats.beat_count_dist_mode = stats.mode(
            cm_beats.beat_count_dist)
        cm_beats.neg_beat_count_dist_mode = stats.mode(
            cm_beats.neg_beat_count_dist)
        cm_beats.beat_count_prom_mode = stats.mode(
            cm_beats.beat_count_prom)
        cm_beats.beat_count_width_mode = stats.mode(
            cm_beats.beat_count_width)
        cm_beats.beat_count_thresh_mode = stats.mode(
            cm_beats.beat_count_thresh)

        # Prints the output from the preceding operations.
        print("Mode of beats using distance parameter: " + str(
            cm_beats.beat_count_dist_mode[0]))
        print("Mode of (negative) beats using distance parameter: " + str(
            cm_beats.neg_beat_count_dist_mode[0]))
        print("Mode of beats using prominence parameter: " + str(
            cm_beats.beat_count_prom_mode[0]))
        print("Mode of beats using width parameter: " + str(
            cm_beats.beat_count_width_mode[0]))
        print("Mode of beats using threshold parameter: " + str(
            cm_beats.beat_count_thresh_mode[0]) + "\n")

        dist_beats_size = np.shape(cm_beats.dist_beats)
        print("Shape of cm_beats.dist_beats: " + 
            str(dist_beats_size))
        neg_dist_beats_size = np.shape(cm_beats.negative_dist_beats)
        print("Shape of cm_beats.negative_dist_beats: " + 
            str(neg_dist_beats_size))
        prom_beats_size = np.shape(cm_beats.prom_beats)
        print("Shape of cm_beats.prom_beats: " + 
            str(prom_beats_size))
        width_beats_size = np.shape(cm_beats.width_beats)
        print("Shape of cm_beats.width_beats: " + 
            str(width_beats_size))
        thresh_beats_size = np.shape(cm_beats.thresh_beats)
        print("Shape of cm_beats.thresh_beats: " + 
            str(thresh_beats_size) + ".\n")

        print("Finished.")
        end_time = time.process_time()
        print(end_time - start_time)
        if batch_data.batch_config == False:
            print("Plotting...")
            graph_beats(analysisGUI, cm_beats, input_param, electrode_config)
    except AttributeError:
        print("No data found." + 
            "Please import data (.txt or .csv converted MCD file) first.")
    except ValueError:
        print("Be sure to use numerical" + 
            " values for the start and end interval.")

# Produces 4-subplot plot of peak finder data and graphs it.  Can be called via 
# button. Will throw exception of data does not exist.
def graph_beats(analysisGUI, cm_beats, input_param, electrode_config):
    try:
        analysisGUI.beatsWindow.paramPlot1.axis1.cla()
        # analysisGUI.beatsWindow.paramPlot.axis2.cla()
        # analysisGUI.beatsWindow.paramPlot.axis3.cla()
        # analysisGUI.beatsWindow.paramPlot.axis4.cla()

        input_param.elec_choice = analysisGUI.beatsWindow.paramSlider1a.value()
        print("Generating graph for electrode " + 
            str(input_param.elec_choice + 1) + ".")
        analysisGUI.beatsWindow.paramPlot1.fig.suptitle(
            "Comparisons of find_peaks methodologies: electrode " + 
            (str(input_param.elec_choice + 1)) + ".")

        mask_dist = ~np.isnan(
            cm_beats.dist_beats.iloc[0:, input_param.elec_choice].values)
        dist_without_nan = cm_beats.dist_beats.iloc[
            0:, input_param.elec_choice].values[mask_dist].astype('int64')
        analysisGUI.beatsWindow.paramPlot1.axis1.plot(
            cm_beats.x_axis[dist_without_nan], 
            cm_beats.y_axis[electrode_config.electrode_names[
                input_param.elec_choice]].values[dist_without_nan], 
            "xr")
        analysisGUI.beatsWindow.paramPlot1.axis1.plot(
            cm_beats.x_axis, 
            cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
        analysisGUI.beatsWindow.paramPlot1.axis1.legend(
            ['distance = ' + str(input_param.min_peak_dist)], 
            loc='lower left')

        # mask_prom = ~np.isnan(
        #     cm_beats.prom_beats.iloc[0:, input_param.elec_choice].values)
        # prom_without_nan = cm_beats.prom_beats.iloc[
        #     0:, input_param.elec_choice].values[mask_prom].astype('int64')
        # analysisGUI.beatsWindow.paramPlot.axis2.plot(
        #     cm_beats.x_axis[prom_without_nan], 
        #     cm_beats.y_axis[electrode_config.electrode_names[
        #         input_param.elec_choice]].values[prom_without_nan], 
        #     "ob")
        # analysisGUI.beatsWindow.paramPlot.axis2.plot(
        #     cm_beats.x_axis,
        #     cm_beats.y_axis[electrode_config.electrode_names[
        #         input_param.elec_choice]].values)
        # analysisGUI.beatsWindow.paramPlot.axis2.legend(
        #     ['prominence = ' + str(input_param.parameter_prominence)], 
        #     loc='lower left')

        # mask_width = ~np.isnan(
        #     cm_beats.width_beats.iloc[0:, input_param.elec_choice].values)
        # width_without_nan = cm_beats.width_beats.iloc[
        #     0:, input_param.elec_choice].values[mask_width].astype('int64')
        # analysisGUI.beatsWindow.paramPlot.axis3.plot(
        #     cm_beats.x_axis[width_without_nan], 
        #     cm_beats.y_axis[electrode_config.electrode_names[
        #         input_param.elec_choice]].values[width_without_nan], 
        #     "vg")
        # analysisGUI.beatsWindow.paramPlot.axis3.plot(
        #     cm_beats.x_axis, 
        #     cm_beats.y_axis.iloc[0:, input_param.elec_choice].values)
        # analysisGUI.beatsWindow.paramPlot.axis3.legend(
        #     ['width = ' + str(input_param.parameter_width)], 
        #     loc='lower left')

        # mask_thresh = ~np.isnan(
        #     cm_beats.thresh_beats.iloc[0:, input_param.elec_choice].values)
        # thresh_without_nan = cm_beats.thresh_beats.iloc[
        #     0:, input_param.elec_choice].values[mask_thresh].astype('int64')
        # analysisGUI.beatsWindow.paramPlot.axis4.plot(
        #     cm_beats.x_axis[thresh_without_nan], 
        #     cm_beats.y_axis[electrode_config.electrode_names[
        #         input_param.elec_choice]].values[thresh_without_nan], 
        #     "xk")
        # analysisGUI.beatsWindow.paramPlot.axis4.plot(
        #     cm_beats.x_axis, 
        #     cm_beats.y_axis[electrode_config.electrode_names[
        #         input_param.elec_choice]].values)
        # analysisGUI.beatsWindow.paramPlot.axis4.legend(
        #     ['threshold = ' + str(input_param.parameter_thresh)], 
        #     loc='lower left')

        analysisGUI.beatsWindow.paramPlot1.draw()
        print("Plotting complete.")
    except AttributeError:
        print("Please use Find Peaks first.")
