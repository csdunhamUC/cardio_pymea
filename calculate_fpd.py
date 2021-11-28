# Author:
# Christopher S. Dunham
# Date:
# 9/22/2021
# Principal Investigator:
# James K. Gimzewski
# Organization:
# University of California, Los Angeles
# Department of Chemistry and Biochemistry
# Original work by CSD unless otherwise noted.
# Utilizes Trapezium's Area algorithm from Vasquez-Seisdedos et al 2011
# "New approach for T-wave end detection on electrocardiogram: Performance
# in noisy conditions"

import time
import pandas as pd
import numpy as np
import seaborn as sns

from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.optimize import minimize


def calc_fpd(analysisGUI, cm_beats, field_potential, local_act_time, heat_map, 
input_param):
    # Filter signals.
    # preprocess_fpd()

    # Find indices for T-wave peak locations.
    field_potential.T_wave_indices = find_T_wave(cm_beats, field_potential, 
        local_act_time, input_param)

    # Adjust second slider (electrode slider) to maximum number of elecs.
    analysisGUI.fpdWindow.paramSlider1b.setMaximum(
        len(field_potential.T_wave_indices.index) - 1)
    
    # field_potential.T_wave_indices.to_excel("Twave_output3.xlsx")

    # Calculate x_m, y_m
    field_potential.x_m, field_potential.y_m = calc_Xm_Ym(cm_beats, 
        field_potential)
    # print(f"x_m: {field_potential.x_m}")
    # print(f"y_m: {field_potential.y_m}")

    field_potential.Tend = calc_Tend(cm_beats, field_potential)
    print(f"Tend df:\n{field_potential.Tend}")

    # Plot T-wave calculation results.
    graph_T_wave(analysisGUI, cm_beats, local_act_time, field_potential, 
        input_param)
    

def preprocess_fpd(cm_beats, field_potential):
    return 5

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
                temp_idx = int(temp_idx) + 100
                idx_end = temp_idx + 400
                temp_volt_trace = cm_beats.y_axis.loc[temp_idx:idx_end, elec]
                temp_pos_T_wave = find_peaks(
                    temp_volt_trace, 
                    height=15,
                    # width=4,
                    # rel_height=0.5,
                    prominence=1,
                    distance=50)
                temp_neg_T_wave = find_peaks(
                    -1*temp_volt_trace,
                    height=15,
                    # width=4,
                    # rel_height=0.5,
                    prominence=1,
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


def calc_Tend(cm_beats, field_potential):
    all_beats = field_potential.T_wave_indices.columns
    all_elecs = field_potential.T_wave_indices.index

    Tend_dict = {}

    for col, beat in enumerate(all_beats):
        temp_dict = {}
        for row, elec in enumerate(all_elecs):
            # x_m, x_r, y_m, y_r are immobile points for a given beat, 
            # elec combo
            print(f"Currently processing {beat}, {elec}.")
            x_m = int(field_potential.x_m[row, col])
            y_m = field_potential.y_m[row, col]
            x_r = x_m + 150
            y_r = y_m + 150
            x_i = cm_beats.x_axis[x_m:x_r].values.astype("int64")
            y_i = cm_beats.y_axis[elec].values[x_i]
            y_i_min = min(y_i)
            y_i_max = max(y_i)

            Tend_guess = [x_m + 30, y_m - 10]

            Tend_bounds = ((x_m, x_r), (y_i_min, y_i_max))

            min_trap_area = minimize(
                fun=trapezium_area, 
                x0=Tend_guess,
                args=(x_m, y_m, x_r, y_r),
                bounds=Tend_bounds)

            real_x_i, real_y_i = min_trap_area.x
            # print(min_trap_area.x)

            temp_dict[elec] = int(real_x_i)
        
        Tend_dict[beat] = temp_dict
    # Variable pairs: (x_m, y_m), (x_r, y_m), (x_r, y_i), (x_i, y_i) 
    # T_end corresponds to (x_i, y_i) where Area, A, is maximum.

    # x_m, y_m = coordinate of location of largest absolute first derivative
    # inside the T-wave, after the last peak in the T-wave.
    # This means we will want to perform peak detection within a given window
    
    # x_r, y_r = coordinate of location somewhere after T_end, with a first
    # derivative value somewhere near zero. Exact location is unimportant;
    # what is important is that this point must be after T_end!
    # print(f"Finished.\nTend_dict: {Tend_dict}")

    Tend = pd.DataFrame(Tend_dict)
    return Tend


def trapezium_area(Tend: tuple, x_m, y_m, x_r, y_r):
    x_i, y_i = Tend
    # Need the maximum through minimization, so minimize the negative function.
    A_trap = -1*(0.5 * (y_m - y_i) * (2*x_r - x_i - x_m))
    return A_trap

# Find x_m, y_m by first locating either the positive or negative T-wave peak
# Determine whether T-wave is positive or negative
def calc_Xm_Ym(cm_beats, field_potential):
    all_beats = field_potential.T_wave_indices.columns
    all_elecs = field_potential.T_wave_indices.index

    x_m = np.zeros((len(all_elecs), len(all_beats)))
    y_m = np.zeros((len(all_elecs), len(all_beats)))

    for col, beat in enumerate(all_beats):
        for row, elec in enumerate(all_elecs):
            if np.isnan(field_potential.T_wave_indices.loc[elec, beat]):
                continue
            else:
                Twave_idx = int(field_potential.T_wave_indices.loc[elec, beat])

                Twave_idx_end = Twave_idx + 200

                # print(f"Row: {row}, Column: {col}")
                x_vals = cm_beats.x_axis[
                    Twave_idx:Twave_idx_end].values.astype('int64')
                y_vals = cm_beats.y_axis[elec].values[x_vals]

                derivs = np.diff(y_vals)/np.diff(x_vals)
                abs_derivs = abs(derivs)
                max_deriv = max(abs_derivs)
                max_deriv_loc = np.argmax(abs_derivs)

                # 10/16/21 @ 19:29
                # Performance is still spotty... both in T-wave detect and
                # subsequent derivative mark. Likely related. Need to consider
                # applying various pre-processing steps before going forward...
                # May also just try to finish implementing the algorithm to
                # see how it performs under existent circumstances.

                # Once we find the index of the maximum abs(deriv), we need
                # the x-value and y-value from cm_beats that corresponds to
                # this (derivative) point.
                real_x = x_vals[max_deriv_loc+1]
                real_y = cm_beats.y_axis[elec].values[real_x]

                x_m[row, col] = real_x
                # print(f"Derivs: {derivs}")
                # print(f"Max derivative: {max_deriv}")
                # print(f"Deriv loc: {max_deriv_loc}")
                # print(f"Where: {np.where(abs(derivs) == max_deriv)}")
                y_m[row, col] = real_y

    return (x_m, y_m)


def calc_Xr_Yr(cm_beats, field_potential):
    x_m = field_potential.x_m
    y_m = field_potential.y_m
    return (x_r, y_r)


# Function that graphs T-wave data on left-side plot (paramPlot1)
# (Soon) designed to plot on a per-beat, per-electrode basis using two sliders.
# Top slider: choose beat.  Bottom slider: choose electrode.
def graph_T_wave(analysisGUI, cm_beats, local_act_time, field_potential, 
input_param):
    # Get beat, electrode from slider 1, slider 2
    beat_choice = analysisGUI.fpdWindow.paramSlider1a.value()
    elec_choice = analysisGUI.fpdWindow.paramSlider1b.value()
    
    # Get all available beats, electrodes that are NaN from dataset.
    all_beats = field_potential.T_wave_indices.columns
    all_elecs = field_potential.T_wave_indices.index
    
    # Use slider value as index for available beats, elecs to yield current one.
    curr_elec = all_elecs[elec_choice]
    curr_beat = all_beats[beat_choice]
    
    print(f"Beat: {curr_beat}")

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
    
    # # Mark T-wave locations.
    # analysisGUI.fpdWindow.paramPlot1.axes.plot(
    #     cm_beats.x_axis[Twave_sans_nan],
    #     cm_beats.y_axis[curr_elec].values[Twave_sans_nan],
    #     "Dm", 
    #     label="T wave")

    # # Mark T-wave derivatives.
    # analysisGUI.fpdWindow.paramPlot1.axes.plot(
    #     field_potential.x_m[elec_choice, :],
    #     field_potential.y_m[elec_choice, :],
    #     "om",
    #     label="Derivative")

    # # Mark T-wave endpoint
    # analysisGUI.fpdWindow.paramPlot1.axes.plot(
    #         field_potential.Tend.loc[curr_elec, all_beats[:-1]],
    #     cm_beats.y_axis[curr_elec].values[
    #         field_potential.Tend.loc[curr_elec, all_beats[:-1]]],
    #     "Py",
    #     label="T-wave End")

    # Original, full plot
    analysisGUI.fpdWindow.paramPlot1.axes.plot(
        cm_beats.x_axis, 
        cm_beats.y_axis[curr_elec].values)
    
    #
    # print(local_act_time.param_dist_raw.loc[curr_elec, curr_beat])
    x_low_lim = local_act_time.param_dist_raw.loc[curr_elec, curr_beat] - 500
    x_high_lim = local_act_time.param_dist_raw.loc[curr_elec, curr_beat] + 500

    # Set axis units.
    analysisGUI.fpdWindow.paramPlot1.axes.set(
        xlabel="Time (ms)",
        ylabel=r"Voltage ($\mu$V)",
        xlim=(x_low_lim, x_high_lim))

    # Increase DPI to 300.
    # analysisGUI.fpdWindow.paramPlot1.fig.set_dpi(300)

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
    
    
