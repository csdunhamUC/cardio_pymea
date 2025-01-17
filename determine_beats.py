# Author: Christopher S. Dunham
# Date: 12/4/2020
# Principal Investigator: James K. Gimzewski
# Organization: University of California, Los Angeles
# Department of Chemistry and Biochemistry
# Original work by CSD

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
            delattr(cm_beats, 'negative_dist_beats')

        # Check whether batch analysis is being performed.
        # If not (i.e. normal mode/single file), pull parameters from GUI
        if batch_data.batch_config == False:
            input_param.min_peak_dist = float(
                analysisGUI.pkDistEdit.text())
            input_param.min_peak_height = float(
                analysisGUI.pkHeightEdit.text())
            input_param.sample_frequency = float(
                analysisGUI.sampleFreqEdit.currentText())

        # file_length = # of rows / sample_freq --> # of seconds in data set 
        # multiply by 60s in 1 min --> number of minutes in data set (rounded)
        analysisGUI.file_length = ((len(raw_data.imported.index) / 
            input_param.sample_frequency) / 60)

        rounded_file_length = str(round(analysisGUI.file_length, 2))

        analysisGUI.fileLength.setText(
            f"Recording length: {rounded_file_length} minutes")

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

        print("Number of columns in cm_beats.y_axis: " + 
            str(len(cm_beats.y_axis.columns)))
        print("Number of rows in cm_beats.y_axis: " + 
            str(len(cm_beats.y_axis)) + "\n")
 
        # Establish "fields" as dataframes for subsequent operations.
        cm_beats.dist_beats = pd.DataFrame()

        # Dataframe for negative amplitudes
        # Applied only for dist_beats as it is the primary detection method
        cm_beats.negative_dist_beats = pd.DataFrame()

        print("Summary of parameters: " 
            + str(input_param.min_peak_height) + ", "
            + str(input_param.min_peak_dist) + ", " 
            + str(input_param.sample_frequency) + "\n")

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
            # Optimization of parameters here may help.
            neg_dist_beats = pd.Series(find_peaks(
                cm_beats.neg_y_axis.iloc[0:, column], 
                height=25,
                distance=400)[0], 
                name=column+1)
            cm_beats.negative_dist_beats = pd.concat(
                [cm_beats.negative_dist_beats, neg_dist_beats], 
                axis='columns')


        # Assign column name identifiers to columns of dist_beats
        cm_beats.dist_beats.columns = electrode_config.electrode_names

        # Data designation to ensure NaN values are properly handled by 
        # subsequent calculations.
        cm_beats.dist_beats.astype('float64')
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
 
        # Finds the mode of beats across the dataset for each peakfinder 
        # parameter set.
        cm_beats.beat_count_dist_mode = stats.mode(
            cm_beats.beat_count_dist)
        cm_beats.neg_beat_count_dist_mode = stats.mode(
            cm_beats.neg_beat_count_dist)

        # Prints the output from the preceding operations.
        print("Mode of beat count: " + str(
            cm_beats.beat_count_dist_mode[0]))

        dist_beats_size = np.shape(cm_beats.dist_beats)
        print("Shape of cm_beats.dist_beats: " + 
            str(dist_beats_size))
        neg_dist_beats_size = np.shape(cm_beats.negative_dist_beats)
        print("Shape of cm_beats.negative_dist_beats: " + 
            str(neg_dist_beats_size))

        print("Finished.")
        end_time = time.process_time()
        print(end_time - start_time)
        
        # If not doing batch analysis, set slider maxima and generate plots.
        if batch_data.batch_config == False:
            analysisGUI.beatsWindow.paramSlider1b.setMaximum(
                len(cm_beats.y_axis.columns) - 1)
            analysisGUI.beatsWindow.paramSlider1a.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1) 
            graph_beats(analysisGUI, cm_beats, electrode_config)
            full_mea_plot(analysisGUI, cm_beats, electrode_config)

    except AttributeError:
        print("No data found." + 
            "Please import data (.txt or .csv converted MCD file) first.")
    except ValueError:
        print("Be sure to use numerical" + 
            " values for the start and end interval.")

# Produces 4-subplot plot of peak finder data and graphs it.  Can be called via 
# button. Will throw exception of data does not exist.
def graph_beats(analysisGUI, cm_beats, electrode_config):
    try:
        analysisGUI.beatsWindow.paramPlot1.axis1.cla()
        analysisGUI.beatsWindow.paramPlot2.axis1.cla()
        
        # Get beat choice from slider 1, use to get beat occurrence time.
        beat_choice = analysisGUI.beatsWindow.paramSlider1a.value()

        # Get electrode choice from slider 2, use to get electrode name
        elec_choice = analysisGUI.beatsWindow.paramSlider1b.value()
        all_elecs = cm_beats.dist_beats.columns
        curr_elec = all_elecs[elec_choice]
        
        if np.isnan(cm_beats.dist_beats.loc[beat_choice, curr_elec]):
            x_low_lim = np.nanmedian(cm_beats.dist_beats.loc[
                beat_choice, :]) - 500
            x_high_lim = np.nanmedian(cm_beats.dist_beats.loc[
                beat_choice, :]) + 500
            analysisGUI.beatsWindow.paramPlot1.fig.suptitle( 
                f"Signal recorded by (excluded) electrode {curr_elec}")
        else:
            x_low_lim = cm_beats.dist_beats.loc[
                beat_choice, curr_elec] - 500
            x_high_lim = cm_beats.dist_beats.loc[
                beat_choice, curr_elec] + 500
            analysisGUI.beatsWindow.paramPlot1.fig.suptitle(
                f"Beat {beat_choice+1} " + 
                f"field potentials recorded by electrode {curr_elec}")

        # Left-panel plot
        mask_dist = ~np.isnan(
            cm_beats.dist_beats.iloc[0:, elec_choice].values)
        dist_without_nan = cm_beats.dist_beats.iloc[
            0:, elec_choice].values[mask_dist].astype('int64')

        analysisGUI.beatsWindow.paramPlot1.axis1.plot(
            cm_beats.x_axis[dist_without_nan], 
            cm_beats.y_axis[electrode_config.electrode_names[
                elec_choice]].values[dist_without_nan], 
            "xr")
        
        analysisGUI.beatsWindow.paramPlot1.axis1.plot(
            cm_beats.x_axis, 
            cm_beats.y_axis.iloc[0:, elec_choice].values)
        
        analysisGUI.beatsWindow.paramPlot1.axis1.set(
            xlim=(x_low_lim, x_high_lim),
            xlabel="Time (ms)",
            ylabel="Voltage (μV)")

        analysisGUI.beatsWindow.paramPlot1.draw()
        # End left-panel plot

    except AttributeError:
        print("Please use Find Peaks first.") 
    except KeyError:
        # See next occurrence of KeyError exception in full_mea_plot.
        print("Minimum peak height value is too high. Check beat count mode. " + 
            "If beat count mode = 0, lower minimum peak height parameter " + 
            "and try again.")


def full_mea_plot(analysisGUI, cm_beats, electrode_config):
    try:
        print()
        # Right-panel plot
        tpose_df = cm_beats.y_axis.T

        tpose_df.insert(0, "Electrode", electrode_config.electrode_names)
        tpose_df.insert(1, "X", electrode_config.electrode_coords_x)
        tpose_df.insert(2, "Y", electrode_config.electrode_coords_y)

        pivot_tpose_df = tpose_df.pivot(
            index='Y', 
            columns='X', 
            values='Electrode')
        
        K_max = 12
        K_min = 0
        L_max = 12
        L_min = 0

        # Get beat choice from slider 1, use to get beat occurrence time.
        beat_choice = analysisGUI.beatsWindow.paramSlider1a.value()
        
        x_low_lim = np.nanmedian(cm_beats.dist_beats.loc[
            beat_choice, :]) - 500
        # Check for negative x_low_lim
        # If x_low_lim is negative, reset to 0 to avoid plotting error
        if x_low_lim < 0:
            x_low_lim = 0
        x_high_lim = np.nanmedian(cm_beats.dist_beats.loc[
            beat_choice, :]) + 500

        int_xlow_lim = int(x_low_lim)
        int_xhigh_lim = int(x_high_lim)
        print(f"X-low: {int_xlow_lim}, X-high: {int_xhigh_lim}")
        int_xmid = (int_xlow_lim + int_xhigh_lim)/2

        # If the resulting plot is not centered on the axis, tune these 
        # parameters for the x and y offset. 
        # Remember: x and y here refer to geometric coordinates for the array!
        x_offset = 1200 # tune these
        y_offset = 1200 # tune these
        
        # Assign shorthand name for the GUI axis/plot.
        ax = analysisGUI.beatsWindow.paramPlot2.axis1
        ax.axis("off")
        # Set plot limits based on the array configuration.
        # Set y limit
        ax.set_ylim([0, 
            (K_max-K_min +1)*y_offset])
        # Set x limit
        ax.set_xlim([int_xlow_lim-100, 
            (L_max - L_min+1)*x_offset + int_xhigh_lim+100])
        ax.set_xticks([])
        ax.set_yticks([])
        # Turn off grid axes
        ax.grid('off')

        # Iterate through the 12x12 grid for each electrode to plot the signal.
        for col_pos, col in enumerate(pivot_tpose_df.columns):
            # val = values (i.e. electrode) from pivot table
            for row_pos, val in enumerate(pivot_tpose_df[col]):
                if val != val:
                    ax.plot(np.arange(1000) + row_pos*x_offset, 
                        5+np.random.rand(1000) + col_pos*y_offset, 
                        color='white',
                        alpha=0)
                else:
                    # Placeholder variable, get y values for electrode
                    temp_y = -1*cm_beats.y_axis.loc[0:, val]
                    # Limit y values (amplitudes) shown to the provided x lim
                    y_axis_vals = temp_y[int_xlow_lim:int_xhigh_lim]
                    ax.plot(
                        (cm_beats.x_axis[
                            int_xlow_lim:int_xhigh_lim] + col_pos*x_offset), 
                        (1000 + y_axis_vals + row_pos*y_offset))
                    # Generate MEA electrode labels
                    ax.annotate(
                        f"{val}",
                        (500 + int_xlow_lim + (col_pos*x_offset), 
                            500 + (row_pos*y_offset)),
                        size=8,
                        ha="center",
                        color=ax.lines[-1].get_color())
        
        # Adjust y-axis so that it fits our coordinate system
        ax.invert_yaxis()
        ax.set_title("All Electrodes View", x=0.425, y=1, pad=20)

        # Assign title and tighten up layout before drawing the figure.
        analysisGUI.beatsWindow.paramPlot2.fig.tight_layout()
        analysisGUI.beatsWindow.paramPlot2.draw()
        # End right-panel plot
    except AttributeError:
        print("Please use Find Peaks first.")
    except KeyError:
        print("Minimum peak height value is too high. Check beat count mode. " + 
            "If beat count mode = 0, lower minimum peak height parameter " + 
            "and try again.")

