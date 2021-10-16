# Author:
# Christopher S. Dunham
# Date:
# 9/22/2021
# Principal Investigator:
# James K. Gimzewski
# Organization:
# University of California, Los Angeles
# Department of Chemistry and Biochemistry
# Original work by CSD.

import time
import pandas as pd
import numpy as np
import seaborn as sns

from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import sosfilt
import scipy.stats as spstats


def calc_fpd(analysisGUI, cm_beats, field_potential, local_act_time, heat_map, 
input_param):
    # Find indices for T-wave peak locations.
    field_potential.T_wave_indices = find_T_wave(cm_beats, field_potential, 
        local_act_time, input_param)

    # Adjust second slider (electrode slider) to maximum number of elecs.
    analysisGUI.fpdWindow.paramSlider1b.setMaximum(
        len(field_potential.T_wave_indices.index) - 1)
    
    # Plot T-wave calculation results.
    graph_T_wave(analysisGUI, cm_beats, field_potential, input_param)
    

    # field_potential.T_wave_indices.to_excel("Twave_output.xlsx")


def find_T_wave(cm_beats, field_potential, local_act_time, input_param):
    fpd_full_dict = {}

    for beat in local_act_time.param_dist_raw.columns[3:]:
        # print(beat)
        temp_dict = {}
        for elec in local_act_time.param_dist_raw.index:
            # print(elec)
            temp_idx = local_act_time.param_dist_raw.loc[elec, beat]
            # print(temp_idx)
            if np.isnan(temp_idx) == True:
                temp_idx = None
            elif np.isnan(temp_idx) == False:
                temp_idx = int(temp_idx) + 30
                idx_end = temp_idx + 430
                temp_volt_trace = cm_beats.y_axis.loc[temp_idx:idx_end, elec]
                temp_pos_T_wave = find_peaks(
                    temp_volt_trace, 
                    height=15,
                    # width=4,
                    # rel_height=0.5,
                    prominence=2,
                    distance=50)
                temp_neg_T_wave = find_peaks(
                    -1*temp_volt_trace,
                    height=15,
                    # width=4,
                    # rel_height=0.5,
                    prominence=2,
                    distance=50)

                check_pos = np.any(temp_pos_T_wave[0])
                if check_pos == True:
                    max_pos_T_wave = max(temp_pos_T_wave[1]["peak_heights"])
                    max_pos_T_idx = np.where(
                        temp_volt_trace == max_pos_T_wave)[0]
                    real_pos_T_idx = temp_volt_trace.index[
                        max_pos_T_idx].values[0]
                else:
                    max_pos_T_wave = None

                check_neg = np.any(temp_neg_T_wave[0])
                if check_neg == True:
                    max_neg_T_wave = max(temp_neg_T_wave[1]["peak_heights"])
                    max_neg_T_idx = np.where(
                        temp_volt_trace == -1*max_neg_T_wave)[0]
                    real_neg_T_idx = temp_volt_trace.index[
                        max_neg_T_idx].values[0]
                else:
                    max_neg_T_wave = None

                # Check if positive and negative amplitudes detected
                if check_pos and check_neg == True:
                    # Check whether T-wave might be biphasic
                    if (max_pos_T_wave - max_neg_T_wave) <= 10:
                        if real_pos_T_idx > real_neg_T_idx:
                            temp_dict[elec] = real_pos_T_idx
                        elif real_pos_T_idx < real_neg_T_idx:
                            temp_dict[elec] = real_neg_T_idx
                    # If not biphasic (assuming roughly equal Twave+, Twave-
                    # peak amplitudes), choose most reasonable index.
                    else:
                        if max_pos_T_wave > max_neg_T_wave:
                            temp_dict[elec] = real_pos_T_idx
                        elif max_pos_T_wave < max_neg_T_wave:
                            temp_dict[elec] = real_neg_T_idx
                # If not, proceed with whichever detection range worked.
                elif check_pos == True and check_neg == False:
                    temp_dict[elec] = real_pos_T_idx
                elif check_neg == True and check_pos == False:
                    temp_dict[elec] = real_neg_T_idx

        # Discriminate against empty keys before adding to fictionary.
        if any(temp_dict.keys()) == False:
            continue
        else:
            fpd_full_dict[beat] = temp_dict
    
    # Return dataframe.
    return pd.DataFrame(fpd_full_dict)


def calc_trapezium(cm_beats, x_i, x_m, x_r, y_i, y_m, y_r):
    # x_m, x_r, y_m, y_r are immobile points
    # Variable pairs: (x_m, y_m), (x_r, y_m), (x_r, y_i), (x_i, y_i)
    # T_end corresponds to (x_i, y_i) where Area, A, is maximum.

    # x_m, y_m = coordinate of location of largest absolute first derivative
    # inside the T-wave, after the last peak in the T-wave.
    # This means we will want to perform peak detection within a given window
    
    # x_r, y_r = coordinate of location somewhere after T_end, with a first
    # derivative value somewhere near zero. Exact location is unimportant;
    # what is important is that this point must be after T_end!

    A_trap = 0.5 * (y_m - y_i) * (2*x_r - x_i - x_m)
    return A_trap


# Find x_m, y_m by first locating either the positive or negative T-wave peak
# Determine whether T-wave is positive or negative
def calc_Xm_Ym(cm_beats, field_potential):
    all_beats = field_potential.T_wave_indices.columns
    all_elecs = field_potential.T_wave_indices.index

    for beat in all_beats:
        
        for elec in all_elecs:

            Twave_idx = field_potential.T_wave_indices.loc[elec, beat]

            Twave_idx_end = T_wave_idx + 150

            x_vals = cm_beats.x_axis[Twave_idx:Twave_idx_end]
            y_vals = cm_beats.y_axis[elec].values[x_vals]

            derivs = np.diff(y_vals)/np.diff(x_vals)
            abs_derivs = abs(derivs)
            max_deriv = max(abs_derivs)
            max_deriv_loc = np.argmax(abs_derivs)

    return (x_m, y_m)


def calc_Xr_Yr(cm_beats, field_potential, x_m, y_m):

    return (x_r, y_r)


# Function that graphs T-wave data on left-side plot (paramPlot1)
# (Soon) designed to plot on a per-beat, per-electrode basis using two sliders.
# Top slider: choose beat.  Bottom slider: choose electrode.
def graph_T_wave(analysisGUI, cm_beats, field_potential, input_param):
    # Get beat, electrode from slider 1, slider 2
    beat_choice = analysisGUI.fpdWindow.paramSlider1a.value()
    elec_choice = analysisGUI.fpdWindow.paramSlider1b.value()
    
    # Get all available beats, electrodes that are NaN from dataset.
    all_beats = field_potential.T_wave_indices.columns
    all_elecs = field_potential.T_wave_indices.index
    
    # Use slider value as index for available beats, elecs to yield current one.
    curr_elec = all_elecs[elec_choice]
    curr_beat = all_beats[beat_choice]
    
    # Clear axis for new plot.
    analysisGUI.fpdWindow.paramPlot1.axes.cla()
    
    # Assign figure title.
    analysisGUI.fpdWindow.paramPlot1.fig.suptitle(
        f"From FPD plot, full signal of {curr_elec}")

    # Generate mask for marking peak locations.
    mask_dist = ~np.isnan(
        cm_beats.dist_beats[curr_elec].values)
    dist_without_nan = cm_beats.dist_beats[curr_elec].values[
        mask_dist].astype("int64")
    
    # Mark peak locations.
    analysisGUI.fpdWindow.paramPlot1.axes.plot(
        cm_beats.x_axis[dist_without_nan], 
        cm_beats.y_axis[curr_elec].values[dist_without_nan], 
        "xr", 
        label="R wave")

    # Generate mask for marking T-wave locations.
    mask_Twave = ~np.isnan(
        field_potential.T_wave_indices.loc[curr_elec].values)
    Twave_sans_nan = field_potential.T_wave_indices.loc[curr_elec].values[
        mask_Twave].astype("int64")
    
    # Mark T-wave locations.
    analysisGUI.fpdWindow.paramPlot1.axes.plot(
        cm_beats.x_axis[Twave_sans_nan],
        cm_beats.y_axis[curr_elec].values[Twave_sans_nan],
        "Dm", 
        label="T wave")

    # Original, full plot
    analysisGUI.fpdWindow.paramPlot1.axes.plot(
        cm_beats.x_axis, 
        cm_beats.y_axis[curr_elec].values)
    
    # Show legend.
    analysisGUI.fpdWindow.paramPlot1.axes.legend(loc='lower left')

    # Update the canvas by drawing the plot.
    analysisGUI.fpdWindow.paramPlot1.draw()


def heatmap_fpd(analysisGUI, cm_beats, field_potential, heat_map, input_param):
    
    if hasattr(heat_map, 'fpd_solo_cbar') is True:
        heat_map.fpd_solo_cbar.remove()
        delattr(heat_map, 'fpd_solo_cbar')

    analysisGUI.fpdWindow.paramPlot.axes.cla()
    selected_beat = analysisGUI.fpdWindow.paramSlider.value()

    electrode_names = to_plot.bp_filt_y.pivot(index='Y', columns='X',
        values='Electrode')
    heatmap_pivot_table = to_plot.bp_filt_y.pivot(index='Y', columns='X',
        values=field_potential.final_beat_count[selected_beat])

    fpd_solo_temp = sns.heatmap(heatmap_pivot_table, cmap="jet",
        annot=electrode_names, fmt="",
        ax=analysisGUI.fpdWindow.paramPlot.axes, vmin=0,
        vmax=field_potential.fpd_max, cbar=False)
    mappable = fpd_solo_temp.get_children()[0]
    heat_map.fpd_solo_cbar = (
        analysisGUI.fpdWindow.paramPlot.axes.figure.colorbar(mappable, 
            ax=analysisGUI.fpdWindow.paramPlot.axes))
    heat_map.fpd_solo_cbar.ax.set_title("FPD (ms)", fontsize=10)

    analysisGUI.fpdWindow.paramPlot.axes.set(
        title=f"Field Potential Duration, Beat {selected_beat+1}",
        xlabel="X coordinate (μm)",
        ylabel="Y coordinate (μm)")
    analysisGUI.fpdWindow.paramPlot.fig.tight_layout()
    analysisGUI.fpdWindow.paramPlot.draw()


def bandpass_filter(cm_beats, input_param, bworth_ord=4, low_cutoff_freq=0.5, 
high_cutoff_freq=30):
    
    print("Using bandpass filter.\n" + 
       f"Order = {bworth_ord}\n" +
       f"Low cutoff = {low_cutoff_freq}Hz\n" + 
       f"High cutoff = {high_cutoff_freq}Hz\n")
    
    sos_bp = butter(bworth_ord, [low_cutoff_freq, high_cutoff_freq], 
        btype='bandpass', output='sos', fs = input_param.sample_frequency)
    
    filtered_bp = np.zeros(
        (len(cm_beats.y_axis.index), len(cm_beats.y_axis.columns)))
    
    for col, column in enumerate(cm_beats.y_axis.columns):
        filtered_bp[:, col] = sosfilt(sos_bp, cm_beats.y_axis[column])
        
    filtered_bp_df = pd.DataFrame(filtered_bp)
    
    return filtered_bp_df

#     elif analysisGUI.beatsWindow.filterTypeEdit.currentText() == "Low-pass Only":
#         bworth_ord = int(analysisGUI.beatsWindow.butterOrderEdit.text())
#         low_cutoff_freq = float(
#             analysisGUI.beatsWindow.lowPassFreqEdit.text())
#         print("Low-pass filter. Order = {}, Low Cutoff Freq. = {}".format(
#             bworth_ord, low_cutoff_freq))

#         sos = butter(bworth_ord, low_cutoff_freq, btype='low', 
#             output='sos', fs=input_param.sample_frequency)
#         filtered_low = np.zeros(
#             (len(cm_beats.y_axis.index), len(cm_beats.y_axis.columns)))
#         for col, column in enumerate(cm_beats.y_axis.columns):
#             filtered_low[:, col] = sosfilt(sos, cm_beats.y_axis[column])
#         cm_beats.y_axis = pd.DataFrame(filtered_low)
    
#     elif analysisGUI.beatsWindow.filterTypeEdit.currentText() == "High-pass Only":
#         bworth_ord = int(analysisGUI.beatsWindow.butterOrderEdit.text())
#         high_cutoff_freq = float(
#             analysisGUI.beatsWindow.highPassFreqEdit.text())
#         print("High-pass filter. Order = {}, High Cutoff Freq = {}".format(
#             bworth_ord, high_cutoff_freq))

#         sos = butter(bworth_ord, high_cutoff_freq, btype='high', 
#             output='sos', fs=input_param.sample_frequency)
#         filtered_high = np.zeros(
#             (len(cm_beats.y_axis.index), len(cm_beats.y_axis.columns)))
#         for col, column in enumerate(cm_beats.y_axis.columns):
#             filtered_high[:, col] = sosfilt(sos, cm_beats.y_axis[column])
#         cm_beats.y_axis = pd.DataFrame(filtered_high)
    
    
