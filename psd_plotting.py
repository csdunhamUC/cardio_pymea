# Author: Christopher S. Dunham
# 1/23/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

# Log vs Log and PSD (also Log vs Log) plotting functions for operation using 
# mea-analysis program.

import numpy as np
import scipy as sp
from scipy import signal
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import time

"""
    Variables of likely use for this module:
    psd_data.loglog_before_ax = psd_data.psd_plots.add_subplot(321)
    psd_data.loglog_during_ax = psd_data.psd_plots.add_subplot(323)
    psd_data.loglog_after_ax = psd_data.psd_plots.add_subplot(325)
    psd_data.psd_before_ax = psd_data.psd_plots.add_subplot(322)
    psd_data.psd_during_ax = psd_data.psd_plots.add_subplot(324)
    psd_data.psd_during_ax = psd_data.psd_plots.add_subplot(326)
    cm_beats.x_axis
    cm_beats.y_axis
    input_param.sample_frequency
    input_param.psd_plot_slider = int(
        analysisGUI.psd_electrode_select.get()) - 1
    local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
        input_param.psd_plot_slider]]
"""

def psd_plotting(analysisGUI, cm_beats, electrode_config, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data):
    # Update slider max.
    analysisGUI.psd_electrode_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))
    # Get slider value.
    input_param.psd_plot_slider = (analysisGUI.psd_electrode_select.get() - 1)
    
    # Update entry boxes for PSD Plotting window.
    analysisGUI.psd_start_beat_value['values'] = pace_maker.final_dist_beat_count
    analysisGUI.psd_end_beat_value['values'] = pace_maker.final_dist_beat_count
    analysisGUI.psd_electrode_choice['values'] = electrode_config.electrode_names

    start_beat = analysisGUI.psd_start_beat_value.get()
    end_beat = analysisGUI.psd_end_beat_value.get()
    elec_choice = analysisGUI.psd_electrode_choice.get()

    param_choices = {"Orig. Signal": cm_beats.y_axis, 
        "Cond. Vel.": conduction_vel.param_dist_raw,
        "Up. Vel.": upstroke_vel.param_dist_normalized, 
        "Pacemaker": pace_maker.param_dist_normalized, 
        "Local AT": local_act_time.param_dist_normalized}

    if len(psd_data.loglog_during_ax.lines) >= 12:
        psd_data.psd_during_ax.cla()
        psd_data.loglog_during_ax.cla()

    plot_log_vs_log(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
        local_act_time, conduction_vel, input_param, psd_data, start_beat, 
        end_beat, elec_choice, param_choices)
    
    plot_psd_welch(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
        local_act_time, conduction_vel, input_param, psd_data, start_beat, 
        end_beat, elec_choice, param_choices, electrode_config)


def plot_log_vs_log(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data, start_beat, end_beat,
elec_choice, param_choices):

    # Only here for testing, will remove later.
    print(pace_maker.param_dist_raw.loc[elec_choice, start_beat])
    print(pace_maker.param_dist_raw.loc[elec_choice, end_beat])

    # Get slider value.
    input_param.psd_plot_slider = (analysisGUI.psd_electrode_select.get() - 1)

    check_param = analysisGUI.psd_param_choice.get()
    if check_param == "Orig. Signal":
        print()

    # Produce log vs log plot of parameter vs distance.
    psd_data.loglog_during_ax.loglog(local_act_time.distance_from_min[
        pace_maker.final_dist_beat_count[input_param.psd_plot_slider]],
        conduction_vel.param_dist_raw[pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider]], '.', base=10, 
            label=pace_maker.final_dist_beat_count[input_param.psd_plot_slider])
    psd_data.loglog_during_ax.set(title="Log CV vs Log Distance", 
        xlabel="Log μm", ylabel="Log CV")
    psd_data.loglog_during_ax.legend(loc="lower left", ncol=6)

    print(pace_maker.final_dist_beat_count[input_param.psd_plot_slider])


def plot_psd_welch(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data, start_beat, end_beat,
elec_choice, param_choices, electrode_config):
    # x = np.ndarray((2,3))

    # print(conduction_vel.param_dist_raw.drop(
    #     columns=['Electrode', 'X', 'Y']).dropna().values.flatten())
    # print(len(conduction_vel.param_dist_raw.drop(
    #     columns=['Electrode', 'X', 'Y']).dropna().values.flatten()))

    check_param = analysisGUI.psd_param_choice.get()
    if check_param == "Orig. Signal":
        param_plotted = param_choices.get(check_param)
        start_index = int(pace_maker.param_dist_raw.loc[elec_choice, start_beat])
        end_index = int(pace_maker.param_dist_raw.loc[elec_choice, end_beat])
        elec_index = electrode_config.electrode_names.index(elec_choice)
        # print("Electrode: " + str(elec_choice) + "\n" + "Index: " + 
        #     str(electrode_config.electrode_names.index(elec_choice)))
        freq, Pxx = signal.welch(param_plotted.iloc[start_index:end_index, 
            elec_index], fs=1.0, window='hann')
        
        psd_data.psd_during_ax.loglog(freq, Pxx, 
            label=elec_choice)
        psd_data.psd_during_ax.set(title="Welch PSD of Signal", 
            xlabel="Log of Frequency (Hz)", ylabel="Log of PSD of Signal")
        psd_data.psd_during_ax.legend(loc="lower left", ncol=6)
    
    elif check_param == "Cond. Vel.":
        param_plotted = param_choices.get(check_param)
        freq, Pxx = signal.welch(param_plotted[pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider]].dropna(), fs=1.0, window='hann')
        
        psd_data.psd_during_ax.loglog(freq, Pxx, 
            label=pace_maker.final_dist_beat_count[input_param.psd_plot_slider])
        psd_data.psd_during_ax.set(title="Welch PSD of CV", 
            xlabel="Log of Frequency (Hz)", ylabel="Log of PSD of CV")
        psd_data.psd_during_ax.legend(loc="lower left", ncol=6)
    
    psd_data.psd_plots.tight_layout()
    psd_data.psd_plots.canvas.draw()

    # freq, Pxx = signal.welch(conduction_vel.param_dist_raw.drop(
    #     columns=['Electrode', 'X', 'Y']).dropna().values.flatten(), fs=1.0, window='hann')
    # x must be a 1d array or series, fs = sampling frequency
    # freq, Pxx = signal.welch(x, fs=1.0)
    # print(freq)
    # print(Pxx_period)