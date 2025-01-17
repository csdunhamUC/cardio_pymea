# Author: Christopher S. Dunham
# Date: 9/22/2021
# Principal Investigator: James K. Gimzewski
# Organization: University of California, Los Angeles
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
input_param, electrode_config):
    try:
        # Filter signals. Currenly not utilized here directly.
        # Same effect should be achieved by applying filters during Find Beats.
        # preprocess_fpd()
        print("Field potential duration calculation in progress.")
        # Find indices for T-wave peak locations.
        field_potential.T_wave_pre = find_T_wave(cm_beats, field_potential, 
            local_act_time, input_param)

        # Adjust second slider (electrode slider) to maximum number of elecs.
        analysisGUI.fpdWindow.paramSlider1b.setMaximum(
            len(field_potential.T_wave_pre.index) - 1)
        
        inserted_x_coords = [0]*len(field_potential.T_wave_pre.index)
        inserted_y_coords = [0]*len(field_potential.T_wave_pre.index)
        
        # Define electrode coordinates for either MEA120 or MEA60 systems.
        # Note: this logic will need revision if you add a different
        # MEA configuration (i.e. not 120 or 60 electrodes).
        for num, elec in enumerate(field_potential.T_wave_pre.index):
            if elec in electrode_config.mea_120_coordinates:
                temp_x = electrode_config.mea_120_coordinates[elec][0]
                temp_y = electrode_config.mea_120_coordinates[elec][1]
                inserted_x_coords[num] = temp_x
                inserted_y_coords[num] = temp_y
            elif elec in electrode_config.mea_60_coordinates:
                temp_x = electrode_config.mea_60_coordinates[elec][0]
                temp_y = electrode_config.mea_60_coordinates[elec][1]
                inserted_x_coords[num] = temp_x
                inserted_y_coords[num] = temp_y

        # Insert x and y coordinates for 'interpolation' of missing FPD values
        field_potential.T_wave_pre.insert(0, "X", inserted_x_coords)
        field_potential.T_wave_pre.insert(1, "Y", inserted_y_coords)

        field_potential.T_wave_indices = interpolate_Twave_idx(
            field_potential.T_wave_pre)

        # Calculate x_m, y_m
        field_potential.x_m, field_potential.y_m = calc_Xm_Ym(
            cm_beats, 
            field_potential)

        # Calculate T-wave endpoint
        field_potential.Tend = calc_Tend(
            cm_beats, 
            field_potential)

        # For brevity, assign local_act_time and field_potential to
        # shorter variable names, for use in calculating FPD.
        lat_data = local_act_time.param_dist_raw
        fpd_data = field_potential.Tend

        # FPD = delta in time between R-wave peak and T-wave endpoint
        # FPD is positive value, so take absolute value here.
        difference = abs(lat_data[lat_data.columns[3:]] - fpd_data)
        # Restore index to retain electrode coordinate information
        # This is necessary for the heatmap.
        reidx_difference = difference.reindex_like(lat_data)
        reidx_difference[["Electrode", "X", "Y"]] = lat_data[
            ["Electrode", "X", "Y"]]
        # Assign FPD to struc using the correctly reindexed values.
        field_potential.FPD = reidx_difference

        # Plot T-wave calculation results.
        graph_T_wave(analysisGUI, cm_beats, local_act_time, field_potential, 
            input_param)
        heatmap_fpd(analysisGUI, cm_beats, field_potential, heat_map, 
            input_param)

        print("Finished.")
    except (AttributeError):
        print("Please use Find Beats first.")
    

# def preprocess_fpd(cm_beats, field_potential):
#    return None


def find_T_wave(cm_beats, field_potential, local_act_time, input_param):
    try:
        fpd_full_dict = {}

        for beat in local_act_time.param_dist_raw.columns[3:]:
            temp_dict = {}
            for elec in local_act_time.param_dist_raw.index:
                temp_idx = local_act_time.param_dist_raw.loc[elec, beat]
                if np.isnan(temp_idx) == True:
                    temp_idx = None
                elif np.isnan(temp_idx) == False:
                    temp_idx = int(temp_idx) + 50
                    idx_end = temp_idx + 400
                    temp_volt_trace = cm_beats.y_axis.loc[
                        temp_idx:idx_end, elec]
                    temp_pos_T_wave = find_peaks(
                        temp_volt_trace, 
                        height=12,
                        width=10,
                        # rel_height=0.5,
                        # prominence=30,
                        # distance=20
                        )
                    temp_neg_T_wave = find_peaks(
                        -1*temp_volt_trace,
                        height=12,
                        width=10,
                        # rel_height=0.5,
                        # prominence=30,
                        # distance=20
                        )

 
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
    except (AttributeError):
        print("")


def interpolate_Twave_idx(data: pd.DataFrame):
    nan_idx = {}
    nan_elecs = {}

    # For each beat,
    # Find and store the location of each NaN electrode
    for beat in data.columns[2:]:
        if data[beat].isna().any():
            idx = np.where(data[beat].isna())[0]
            nan_idx[beat] = idx
            nan_elecs[beat] = (data.index[idx].values)

    nan_elec_neighbors = {}
    # For each NaN electrode,
    # Get the coordinates of the NaN electrode
    # Compare these coordinates to all other coordinates
    # Find the coordinates within 230 micrometers in x and y directions 
    # (ignores diagonal)
    for elecs, (beat, idx) in zip(nan_elecs.values(), nan_idx.items()):
        temp_df = data[[beat, "X", "Y"]]

        temp_neighbors = []
        if len(elecs) > 1:
            for elec in elecs:
                nan_x = temp_df.loc[elec, "X"]
                nan_y = temp_df.loc[elec, "Y"]
                for elec, x, y in zip(temp_df.index, data["X"], data["Y"]):
                    coord_dist = np.sqrt( (x - nan_x)**2 + (y - nan_y)**2 )
                    if (coord_dist <= 230) & (elec not in nan_elecs[beat]):
                        temp_neighbors.append(elec)
        else:
            nan_x = temp_df.iloc[idx, 1].values[0]
            nan_y = temp_df.iloc[idx, 2].values[0]
            for elec, x, y in zip(temp_df.index, data["X"], data["Y"]):
                coord_dist = np.sqrt( (x - nan_x)**2 + (y - nan_y)**2 )
                if (coord_dist <= 230) & (elec not in nan_elecs[beat]):
                    temp_neighbors.append(elec)
        nan_elec_neighbors[beat] = temp_neighbors


    # New dataframe to return values for missing FPD T-waves.
    new_df = data.copy()

    # For each beat, get the FPD value of each neighbor electrode
    # Average these values
    # Sub in for the nan electrode.
    for (beat, nan_neighbor), nan_elec in zip(
    nan_elec_neighbors.items(), nan_elecs.values()):
        mean_of_neighbors = np.floor(np.mean(data.loc[nan_neighbor, beat]))
        new_df.loc[nan_elec, beat] = mean_of_neighbors

    new_df.drop(columns=["X", "Y"], inplace=True)
    print("Finished nearest-neighbor estimation of NaN FPD T-wave indices.")
    return new_df


def calc_Tend(cm_beats, field_potential):
    all_beats = field_potential.T_wave_indices.columns
    all_elecs = field_potential.T_wave_indices.index

    max_x = max(cm_beats.x_axis)

    Tend_dict = {}

    for col, beat in enumerate(all_beats):
        temp_dict = {}
        for row, elec in enumerate(all_elecs):
            # Get Twave peak location
            Twave_idx = int(field_potential.T_wave_indices.loc[elec, beat])

            # x_m, x_r, y_m, y_r are immobile points for a given beat, 
            # elec combo
            print(f"Currently processing {beat}, {elec}.")
            x_m = int(field_potential.x_m[row, col])
            y_m = field_potential.y_m[row, col]
            
            # Check whether x_r is being chosen outside of the bounds of the
            # recorded data.
            if (Twave_idx + 400) < max_x:
                x_r = Twave_idx + 400
                x_i_guess = x_m + 30
            elif (Twave_idx + 400) > max_x:
                x_r = int(max_x)
                x_i_guess = int(max_x - 1)
                # print(f"Xi Guess: {x_i_guess}")

            y_r = cm_beats.y_axis[elec].values[x_r]
            y_r = y_m + 100
            x_i = cm_beats.x_axis[x_m:x_r].values.astype("int64")
            # y_i = cm_beats.y_axis[elec].values[x_i]
            y_i_min = -50 # min(y_i)
            y_i_max = 50 # max(y_i)

            # x_i_guess = x_m + 30
            y_i_guess = cm_beats.y_axis[elec].values[x_i_guess]

            Tend_guess = [x_i_guess, y_i_guess]
            # print(f"Tend Guess: {Tend_guess}")

            # print(f"x_m: {x_m}\nx_r: {x_r}")
            Tend_bounds = ((x_m, x_r), (y_i_min, y_i_max))

            # print(f"Tend bounds: {Tend_bounds}")

            min_trap_area = minimize(
                fun=trapezium_area, 
                x0=Tend_guess,
                args=(x_m, y_m, x_r, y_r),
                method="Nelder-Mead",
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
    # Need the maximum area. 
    # We're finding the maximum through minimization
    # Therefore, we must minimize the negative function.
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

                # Twave window is the interval between:
                # (Twave_idx, Twave_idx_end)
                # This value (200 ms)  is in line with Vasquez et al 2011.
                Twave_idx_end = Twave_idx + 200

                # Get x, y values for use in derivative 
                # (change in y / change in x)
                x_vals = cm_beats.x_axis[
                    Twave_idx:Twave_idx_end].values.astype('int64')
                y_vals = cm_beats.y_axis[elec].values[x_vals]

                # Calculative derivatives along T-waves over the window
                # specified by Twave_idx and Twave_idx_end
                derivs = np.diff(y_vals)/np.diff(x_vals)
                
                # Find the maximum absolute derivative in the T-wave
                abs_derivs = abs(derivs)
                max_deriv = max(abs_derivs)
                # Obtain the point in time, corresponding to x_m,
                # of the maximum absolute derivative.
                max_deriv_loc = np.argmax(abs_derivs)


                # Once we find the index of the maximum abs(deriv), we need
                # the x-value and y-value from cm_beats that corresponds to
                # this (maximum absolute derivative) point.
                real_x = x_vals[max_deriv_loc]
                real_y = cm_beats.y_axis[elec].values[real_x]

                x_m[row, col] = real_x
                y_m[row, col] = real_y

    # Return (x_m, y_m) coordinate pair.
    return (x_m, y_m)
# End of calc_Xm_Ym function


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

    # Clear axis for new plot.
    analysisGUI.fpdWindow.paramPlot1.axis1.cla()
    
    # Assign figure title.
    analysisGUI.fpdWindow.paramPlot1.fig.suptitle(
        f"{curr_beat} of electrode {curr_elec}")

    # Generate mask for marking peak locations.
    mask_dist = ~np.isnan(
        cm_beats.dist_beats[curr_elec].values)
    dist_without_nan = cm_beats.dist_beats[curr_elec].values[
        mask_dist].astype("int64")
    
    # Mark peak locations.
    analysisGUI.fpdWindow.paramPlot1.axis1.plot(
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
    analysisGUI.fpdWindow.paramPlot1.axis1.plot(
        cm_beats.x_axis[Twave_sans_nan],
        cm_beats.y_axis[curr_elec].values[Twave_sans_nan],
        "Dm", 
        label="T wave")

    # Mark T-wave derivatives.
    analysisGUI.fpdWindow.paramPlot1.axis1.plot(
        field_potential.x_m[elec_choice, :],
        field_potential.y_m[elec_choice, :],
        "om",
        label="Derivative")

    # Mark T-wave endpoint
    analysisGUI.fpdWindow.paramPlot1.axis1.plot(
        field_potential.Tend.loc[curr_elec, all_beats[:-1]],
        cm_beats.y_axis[curr_elec].values[
            field_potential.Tend.loc[curr_elec, all_beats[:-1]]],
        "Py",
        label="T-wave End")

    # Original, full plot
    analysisGUI.fpdWindow.paramPlot1.axis1.plot(
        cm_beats.x_axis, 
        cm_beats.y_axis[curr_elec].values)
    
    # print(local_act_time.param_dist_raw.loc[curr_elec, curr_beat])
    x_low_lim = local_act_time.param_dist_raw.loc[curr_elec, curr_beat] - 500
    x_high_lim = local_act_time.param_dist_raw.loc[curr_elec, curr_beat] + 500
    rwave_time = local_act_time.param_dist_raw.loc[curr_elec, curr_beat]
    
    # Set axis units.
    analysisGUI.fpdWindow.paramPlot1.axis1.set(
        xlabel="Time (ms)",
        ylabel=r"Voltage ($\mu$V)",
        xlim=(x_low_lim, x_high_lim))

    # Change figure size, increase DPI to 300.
    # analysisGUI.fpdWindow.paramPlot1.fig.set_size_inches(13, 2)
    # analysisGUI.fpdWindow.paramPlot1.fig.set_dpi(300)
    
    # Show legend.
    analysisGUI.fpdWindow.paramPlot1.axis1.legend(loc='lower left')

    # Update the canvas by drawing the plot.
    analysisGUI.fpdWindow.paramPlot1.draw()


def heatmap_fpd(analysisGUI, cm_beats, field_potential, heat_map, input_param):
    
    if hasattr(heat_map, 'fpd_cbar') is True:
        heat_map.fpd_cbar.remove()
        delattr(heat_map, 'fpd_cbar')

    analysisGUI.fpdWindow.paramPlot2.axis1.cla()
    beat_choice = analysisGUI.fpdWindow.paramSlider1a.value()

    # Get beats from data.
    all_beats = field_potential.T_wave_indices.columns

    # Use slider value as index for available beats to yield current beat.
    curr_beat = all_beats[beat_choice]

    fpd_max = field_potential.FPD[field_potential.FPD.columns[3:]].max().max()
    fpd_min = field_potential.FPD[field_potential.FPD.columns[3:]].min().min()

    electrode_names = field_potential.FPD.pivot(
        index='Y', 
        columns='X',
        values='Electrode'
        )
    
    heatmap_pivot_table = field_potential.FPD.pivot(
        index='Y', 
        columns='X',
        values=curr_beat
        )

    fpd_temp = sns.heatmap(
        heatmap_pivot_table, 
        cmap="jet",
        annot=electrode_names, 
        fmt="",
        ax=analysisGUI.fpdWindow.paramPlot2.axis1, 
        vmin=fpd_min,
        vmax=fpd_max,
        cbar=False
        )

    mappable = fpd_temp.get_children()[0]
    heat_map.fpd_cbar = (
        analysisGUI.fpdWindow.paramPlot2.axis1.figure.colorbar(
            mappable, 
            ax=analysisGUI.fpdWindow.paramPlot2.axis1))
    heat_map.fpd_cbar.ax.set_title(
        "FPD (ms)", 
        fontsize=10
        )

    analysisGUI.fpdWindow.paramPlot2.axis1.set(
        title=f"Field Potential Duration, {curr_beat}",
        xlabel="X coordinate (μm)",
        ylabel="Y coordinate (μm)")
    analysisGUI.fpdWindow.paramPlot2.fig.tight_layout()
    analysisGUI.fpdWindow.paramPlot2.draw()


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
    
    
