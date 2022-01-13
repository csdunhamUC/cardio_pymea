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
from scipy import optimize
from matplotlib import pyplot as plt

# Comment out init() if using Python on Windows.
init()

# Function that calculates the pacemaker (time lag).  
# Performs this calculation for all electrodes, and filters
# electrodes based on mismatched beat counts relative to the mode of 
# the beat count.
def calculate_pacemaker(analysisGUI, cm_beats, pace_maker, heat_map, 
input_param, electrode_config):
    try:
        if hasattr(pace_maker, 'param_dist_raw') is True:
            print("Clearing old pacemaker data before new calculation.")
            delattr(pace_maker, 'param_dist_raw')
            delattr(pace_maker, 'param_prom_raw')
            delattr(pace_maker, 'param_width_raw')
            delattr(pace_maker, 'param_thresh_raw')

        # Clock the time it takes to run the calculation.
        start_time = time.process_time()
        print("Calculating pacemaker intervals per beat.")

        # Establishing these attributes of the pace_maker class as DataFrames.
        pace_maker.param_dist_raw = pd.DataFrame()
        pace_maker.param_negative_dist_raw = pd.DataFrame()
        pace_maker.param_prom_raw = pd.DataFrame()
        pace_maker.param_width_raw = pd.DataFrame()
        pace_maker.param_thresh_raw = pd.DataFrame()

        # Performs PM calculation for each detection parameter 
        # (peak distance, negative distance, prominence, width, threshold)
        for column in range(len(cm_beats.dist_beats.columns)):
            if cm_beats.beat_count_dist_mode[0] == len(
            cm_beats.dist_beats.iloc[0:, column].dropna()):
                pace_maker_dist_raw = pd.Series(
                    cm_beats.dist_beats.iloc[0:, column].dropna(), 
                    name=column+1)
                pace_maker.param_dist_raw = pd.concat(
                    [pace_maker.param_dist_raw, pace_maker_dist_raw], 
                    axis='columns')
            else:
                pace_maker_dist_raw = pd.Series(
                    name=column+1, 
                    dtype='float64')
                pace_maker.param_dist_raw = pd.concat(
                    [pace_maker.param_dist_raw, pace_maker_dist_raw], 
                    axis='columns')

        for column in range(len(cm_beats.negative_dist_beats.columns)):
            if cm_beats.neg_beat_count_dist_mode[0] == len(
            cm_beats.negative_dist_beats.iloc[0:, column].dropna()):
                pace_maker_neg_dist_raw = pd.Series(
                    cm_beats.negative_dist_beats.iloc[0:, column].dropna(), 
                    name=column+1)
                pace_maker.param_negative_dist_raw = pd.concat(
                    [pace_maker.param_negative_dist_raw, 
                        pace_maker_neg_dist_raw], 
                    axis='columns')
            else:
                pace_maker_neg_dist_raw = pd.Series(
                    name=column+1, 
                    dtype='float64')
                pace_maker.param_negative_dist_raw = pd.concat(
                    [pace_maker.param_negative_dist_raw, 
                        pace_maker_neg_dist_raw], 
                    axis='columns')

        for column in range(len(cm_beats.prom_beats.columns)):
            if cm_beats.beat_count_prom_mode[0] == len(
            cm_beats.prom_beats.iloc[0:, column].dropna()):
                pace_maker_prom_raw = pd.Series(
                    cm_beats.prom_beats.iloc[0:, column].dropna(), 
                    name=column+1)
                pace_maker.param_prom_raw = pd.concat(
                    [pace_maker.param_prom_raw, pace_maker_prom_raw], 
                    axis='columns')
            else:
                pace_maker_prom_raw = pd.Series(
                    name=column+1, 
                    dtype='float64')
                pace_maker.param_prom_raw = pd.concat(
                    [pace_maker.param_prom_raw, pace_maker_prom_raw], 
                    axis='columns')

        for column in range(len(cm_beats.width_beats.columns)):
            if cm_beats.beat_count_prom_mode[0] == len(
            cm_beats.width_beats.iloc[0:, column].dropna()):
                pace_maker_width_raw = pd.Series(
                    cm_beats.width_beats.iloc[0:, column].dropna(), 
                    name=column+1)
                pace_maker.param_width_raw = pd.concat(
                    [pace_maker.param_width_raw, pace_maker_width_raw], 
                    axis='columns')
            else:
                pace_maker_width_raw = pd.Series(
                    name=column+1, 
                    dtype='float64')
                pace_maker.param_width_raw = pd.concat(
                    [pace_maker.param_width_raw, pace_maker_width_raw], 
                    axis='columns')

        for column in range(len(cm_beats.thresh_beats.columns)):
            if cm_beats.beat_count_thresh_mode[0] == len(
            cm_beats.thresh_beats.iloc[0:, column].dropna()):
                pace_maker_thresh_raw = pd.Series(
                    cm_beats.thresh_beats.iloc[0:, column].dropna(), 
                    name=column+1)
                pace_maker.param_thresh_raw = pd.concat(
                    [pace_maker.param_thresh_raw, pace_maker_thresh_raw], 
                    axis='columns')
            else:
                pace_maker_thresh_raw = pd.Series(
                    name=column+1, 
                    dtype='float64')
                pace_maker.param_thresh_raw = pd.concat(
                    [pace_maker.param_thresh_raw, pace_maker_thresh_raw], 
                    axis='columns')

        # Normalizes the values for each beat by subtracting the minimum time 
        # of a given beat from all other electrodes
        if 1000 / input_param.sample_frequency == 1.0:
            pace_maker.param_dist_normalized = pace_maker.param_dist_raw.sub(
                pace_maker.param_dist_raw.min(axis=1), axis=0)
            pace_maker.param_neg_dist_normalized = pace_maker.param_negative_dist_raw.sub(
                pace_maker.param_negative_dist_raw.min(axis=1), axis=0)
            pace_maker.param_prom_normalized = pace_maker.param_prom_raw.sub(
                pace_maker.param_prom_raw.min(axis=1), axis=0)
            pace_maker.param_width_normalized = pace_maker.param_width_raw.sub(
                pace_maker.param_width_raw.min(axis=1), axis=0)
            pace_maker.param_thresh_normalized = pace_maker.param_thresh_raw.sub(
                pace_maker.param_thresh_raw.min(axis=1), axis=0)
        elif 1000 / input_param.sample_frequency == 0.1:
            pace_maker.param_dist_normalized = pace_maker.param_dist_raw.sub(
                pace_maker.param_dist_raw.min(axis=1), axis=0).div(10)
            pace_maker.param_neg_dist_normalized = pace_maker.param_negative_dist_raw.sub(
                pace_maker.param_negative_dist_raw.min(axis=1), axis=0).div(10)
            pace_maker.param_prom_normalized = pace_maker.param_prom_raw.sub(
                pace_maker.param_prom_raw.min(axis=1), axis=0).div(10)
            pace_maker.param_width_normalized = pace_maker.param_width_raw.sub(
                pace_maker.param_width_raw.min(axis=1), axis=0).div(10)
            pace_maker.param_thresh_normalized = pace_maker.param_thresh_raw.sub(
                pace_maker.param_thresh_raw.min(axis=1), axis=0).div(10)

        # Set slider values to maximum number of beats
        analysisGUI.mainSlider.setMaximum(
            int(cm_beats.beat_count_dist_mode[0]) - 1)

        # Find the number of excluded electrodes (removed for noise, etc)
        pace_maker.excluded_elec = np.count_nonzero(
            pace_maker.param_dist_normalized.count() == 0)
        print("{}Excluded electrodes: {} {}".format(
            Fore.YELLOW, pace_maker.excluded_elec, Style.RESET_ALL))

        # Assigns column headers (names) using the naming convention provided 
        # in the electrode_config class.
        pace_maker.param_dist_normalized.columns = electrode_config.electrode_names
        pace_maker.param_neg_dist_normalized.columns = electrode_config.electrode_names
        pace_maker.param_prom_normalized.columns = electrode_config.electrode_names
        pace_maker.param_width_normalized.columns = electrode_config.electrode_names
        pace_maker.param_thresh_normalized.columns = electrode_config.electrode_names

        pace_maker.param_dist_raw.columns = electrode_config.electrode_names
        pace_maker.param_negative_dist_raw.columns = electrode_config.electrode_names

        # Generate index (row) labels, as a list, in order to access chosen 
        # beat heatmaps in subsequent function.
        pace_maker.final_dist_beat_count = []
        for beat in range(int(cm_beats.beat_count_dist_mode[0])):
            pace_maker.final_dist_beat_count.append('Beat ' + str(beat+1))

        # Generate index (row) labels, as a list, for assignment to dataframe, 
        # prior to transpose.
        dist_new_index = []
        for row in pace_maker.param_dist_normalized.index:
            dist_new_index.append('Beat ' + str(row+1))

        neg_dist_new_index = []
        for row in pace_maker.param_neg_dist_normalized.index:
            neg_dist_new_index.append('Beat ' + str(row+1))

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
        pace_maker.param_neg_dist_normalized.index = neg_dist_new_index
        pace_maker.param_prom_normalized.index = prom_new_index
        pace_maker.param_width_normalized.index = width_new_index
        pace_maker.param_thresh_normalized.index = thresh_new_index

        pace_maker.param_dist_raw.index = dist_new_index
        pace_maker.param_negative_dist_raw.index = neg_dist_new_index

        # Transpose dataframe to make future plotting easier.  
        # Makes rows = electrodes and columns = beat number.
        pace_maker.param_dist_normalized = pace_maker.param_dist_normalized.transpose()
        pace_maker.param_prom_normalized = pace_maker.param_prom_normalized.transpose()
        pace_maker.param_width_normalized = pace_maker.param_width_normalized.transpose()
        pace_maker.param_thresh_normalized = pace_maker.param_thresh_normalized.transpose()

        pace_maker.param_dist_raw = pace_maker.param_dist_raw.transpose()
        pace_maker.param_negative_dist_raw = pace_maker.param_negative_dist_raw.transpose()

        # Find the maximum time lag per beat.
        pace_maker.param_dist_normalized_per_beat_max = pace_maker.param_dist_normalized.max()

        # Find maximum time lag (interval) in data set
        pace_maker.param_dist_normalized_max = pace_maker.param_dist_normalized.max().max()

        # Find the mean interval time.
        pace_maker.param_dist_normalized_mean = np.nanmean(
            pace_maker.param_dist_normalized.max())

        # Insert electrode name as column to make future plotting easier when 
        # attempting to use pivot table.
        pace_maker.param_dist_normalized.insert(
            0, 'Electrode', electrode_config.electrode_names)
        pace_maker.param_prom_normalized.insert(
            0, 'Electrode', electrode_config.electrode_names)
        pace_maker.param_width_normalized.insert(
            0, 'Electrode', electrode_config.electrode_names)
        pace_maker.param_thresh_normalized.insert(
            0, 'Electrode', electrode_config.electrode_names)

        # Insert electrode coordinates X,Y (in micrometers) as columns after 
        # electrode identifier.
        pace_maker.param_dist_normalized.insert(
            1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_dist_normalized.insert(
            2, 'Y', electrode_config.electrode_coords_y)
        # Repeat for prominence parameter.
        pace_maker.param_prom_normalized.insert(
            1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_prom_normalized.insert(
            2, 'Y', electrode_config.electrode_coords_y)
        # Repeat for width parameter.
        pace_maker.param_width_normalized.insert(
            1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_width_normalized.insert(
            2, 'Y', electrode_config.electrode_coords_y)
        # Repeat for threshold parameter.
        pace_maker.param_thresh_normalized.insert(
            1, 'X', electrode_config.electrode_coords_x)
        pace_maker.param_thresh_normalized.insert(
            2, 'Y', electrode_config.electrode_coords_y)

        pace_maker.param_dist_normalized.name = 'Pacemaker (Normalized)'

        print("Done.")
        # Finishes tabulating time for the calculation and prints the time.
        end_time = time.process_time()
        # print(end_time - start_time)
        print("{}Maximum time lag in data set: {} {}".format(
            Fore.MAGENTA, pace_maker.param_dist_normalized_max, 
            Style.RESET_ALL))
        deinit()
    except AttributeError:
        print("Please use Find Peaks first.")


def graph_pacemaker(analysisGUI, heat_map, pace_maker, input_param):
    try:
        if hasattr(heat_map, 'pm_solo_cbar') is True:
            heat_map.pm_solo_cbar.remove()
            delattr(heat_map, 'pm_solo_cbar')

        analysisGUI.pmWindow.paramPlot.axes.cla()
        input_param.pm_solo_beat_choice = analysisGUI.pmWindow.paramSlider.value()

        electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table = pace_maker.param_dist_normalized.pivot(index='Y', 
            columns='X', values=pace_maker.final_dist_beat_count[
                input_param.pm_solo_beat_choice])

        pm_solo_temp = sns.heatmap(heatmap_pivot_table, cmap="jet", 
            annot=electrode_names, fmt="", 
            ax=analysisGUI.pmWindow.paramPlot.axes, vmin=0, 
            vmax=pace_maker.param_dist_normalized_max, cbar=False)
        mappable = pm_solo_temp.get_children()[0]
        heat_map.pm_solo_cbar = (
            analysisGUI.pmWindow.paramPlot.axes.figure.colorbar(mappable, 
            ax=analysisGUI.pmWindow.paramPlot.axes))
        heat_map.pm_solo_cbar.ax.set_title("Time Lag (ms)", fontsize=10)

        analysisGUI.pmWindow.paramPlot.axes.set(title="Pacemaker, Beat " + 
            str(input_param.pm_solo_beat_choice+1), xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")
        analysisGUI.pmWindow.paramPlot.fig.tight_layout()
        analysisGUI.pmWindow.paramPlot.draw()

    except AttributeError:
        print("Please calculate PM first.")
    except IndexError:
        print("You entered a beat that does not exist.")


# Much appreciation to the author of this Stack Exchange post, whose code I
# adapted (with modifications) for the purpose of the circle fitting.
# (https://stackoverflow.com/questions/44647239/
# how-to-fit-a-circle-to-a-set-of-points-with-a-constrained-radius)
def estimate_pm_origin(analysisGUI, pace_maker, input_param):
    try:
        analysisGUI.circFitWindow.paramPlot.axes.cla()
        # pace_maker.param_dist_normalized
        curr_beat = pace_maker.final_dist_beat_count[
            analysisGUI.circFitWindow.paramSlider.value()]
        temp_without_nan = pace_maker.param_dist_normalized[
            ['X', 'Y', curr_beat]].dropna()
        contX_uniq = np.sort(temp_without_nan.X.unique())
        contY_uniq = np.sort(temp_without_nan.Y.unique())
        contX, contY = np.meshgrid(contX_uniq, contY_uniq)
        contPM = temp_without_nan.pivot_table(index='Y', columns='X', 
            values=temp_without_nan).values

        # Get values of wave front from contour plot of PM
        pm_contour = plt.contour(contX, contY, 
            contPM, cmap='jet')

        # Figure out which contour lines to used.  Currently favors the lines
        # containing the most points (x,y).
        best_vals = False
        if best_vals is False:
            for i in range(len(pm_contour.allsegs)):
                if len(pm_contour.allsegs[i]) > 0:
                    if len(pm_contour.allsegs[i][0]) >= 11:
                        temp_array = pm_contour.allsegs[i][0][0:]
                        print("Using position " + str(i) + " from allsegs.")
                        best_vals = True
                        break

        if best_vals is False:
            for i in range(len(pm_contour.allsegs)):
                if len(pm_contour.allsegs[i]) > 0:
                    if len(pm_contour.allsegs[i][0]) >= 9:
                        temp_array = pm_contour.allsegs[i][0][0:]
                        print("No segments with more than 11 points.Using 9.")
                        print("Using position " + str(i) + " from allsegs.")
                        best_vals = True
                        break

        if best_vals is False:
            for i in range(len(pm_contour.allsegs)):
                if len(pm_contour.allsegs[i]) > 0:
                    if len(pm_contour.allsegs[i][0]) >= 7:
                        temp_array = pm_contour.allsegs[i][0][0:]
                        print("No segments with more than 9 points. Using 7.")
                        print("Using position " + str(i) + " from allsegs.")
                        best_vals = True
                        break
        
        if best_vals is False:
            for i in range(len(pm_contour.allsegs)):
                if len(pm_contour.allsegs[i]) > 0:
                    if len(pm_contour.allsegs[i][0]) >= 4:
                        temp_array = pm_contour.allsegs[i][0][0:]
                        print("No segments with more than 7 points. Using 4.")
                        print("Using position " + str(i) + " from allsegs.")
                        best_vals = True
                        break

        print(np.shape(temp_array))
        x_test = temp_array[0:, 0]
        y_test = temp_array[0:, 1]
        cont_data = np.array([x_test, y_test]).T

        x_guess = np.mean(x_test)
        y_guess = np.mean(y_test)
        print(x_guess)
        print(y_guess)

        # Estimated h, k and initial radius Ri of circle for 200x30um spacing 
        # and electrode size, 120 electrode MEA w/ PM @ center
        # h, k, R_pred, r_min, r_max
        low_bounds = np.array([-9500, -9500, 75, 75, 150])
        up_bounds = np.array([9500, 9500, 9500, 150, 9500])
        estimates = [x_guess, y_guess, 1000, 100, 9500]

        circle_fit_pts = optimize.least_squares(circle_residual, 
            estimates, bounds=(low_bounds, up_bounds), args=([cont_data])).x
        print(circle_fit_pts)

        phi_vals = np.linspace(0, 2*np.pi, 500)
        pm_estimate=np.array(
            [gen_circle(phi,*circle_fit_pts) for phi in phi_vals])

        analysisGUI.circFitWindow.paramPlot.axes.plot(pm_estimate[:,0], 
            pm_estimate[:,1])
        analysisGUI.circFitWindow.paramPlot.axes.contourf(contX, contY, contPM, 
            cmap='jet')
        analysisGUI.circFitWindow.paramPlot.axes.scatter(x_test, y_test)
        analysisGUI.circFitWindow.paramPlot.axes.scatter(circle_fit_pts[0], 
            circle_fit_pts[1], c="orange")
        analysisGUI.circFitWindow.paramPlot.axes.invert_yaxis()
        analysisGUI.circFitWindow.paramPlot.axes.set(
            title="Estimated Origin of Pacemaker During " + curr_beat, 
            xlabel="Coordinates (μm)", ylabel="Coordinates (μm)")
        analysisGUI.circFitWindow.paramPlot.draw()

    except TypeError:
        print("Issue with chosen allseg.  Select a new band.")
    except IndexError:
        print("Early allseg entries are empty.")
    except AttributeError:
        print("No data.")


# Calculate residuals for least_squares, with bounds on R
def circle_residual(parameters, data_points):
    h, k, R_pred, r_min, r_max = parameters
    # r = sqrt((x-h)^2 + (y-k)^2)
    Ri = r_min + (r_max - r_min) * 0.5 * (1+np.tanh(R_pred))
    radius = [np.sqrt((x - h)**2 + (y - k)**2) for x, y in data_points]
    residual = np.array([(Ri - rad)**2 for rad in radius])
    return residual


def gen_circle(phi, h, k, R_pred, r_min, r_max):
    r_limited = (r_min + (r_max - r_min) * 0.5 * (1 + np.tanh(r_max)))
    return [h + r_limited * np.cos(phi), k + r_limited * np.sin(phi)]
