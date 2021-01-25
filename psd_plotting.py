# Author: Christopher S. Dunham
# 1/23/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

# PSD and log-log plotting functions for operation using mea-analysis program.

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
    pace_maker.param_dist_normalized
    local_act_time.param_dist_normalized
    upstroke_vel.param_dist_normalized
    conduction_vel.param_dist_raw
    input_param.sample_frequency
    analysisGUI.psd_start_beat.cget()
    analysisGUI.psd_end_beat.cget()
    analysisGUI.psd_start_beat_value.cget()
    analysisGUI.psd_end_beat_value.cget()
    input_param.psd_plot_slider = int(
        analysisGUI.psd_electrode_select.get()) - 1
    local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider]]
"""

def psd_plotting(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data):
    # Update slider max.
    analysisGUI.psd_electrode_select.configure(to=int(cm_beats.beat_count_dist_mode[0]))
    # Get slider value.
    input_param.psd_plot_slider = (analysisGUI.psd_electrode_select.get() - 1)
    
    # Update entry boxes for PSD Plotting window.
    analysisGUI.psd_start_beat_value['values'] = pace_maker.final_dist_beat_count
    analysisGUI.psd_end_beat_value['values'] = pace_maker.final_dist_beat_count

    plot_log_vs_log(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
        local_act_time, conduction_vel, input_param, psd_data)
    plot_psd_welch(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
        local_act_time, conduction_vel, input_param, psd_data)


def plot_log_vs_log(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data):
    start_beat = analysisGUI.psd_start_beat_value.get()
    print(start_beat)
    end_beat = analysisGUI.psd_end_beat_value.get()
    print(end_beat)

    # Get slider value.
    input_param.psd_plot_slider = (analysisGUI.psd_electrode_select.get() - 1)

    psd_data.loglog_during_ax.cla()

    psd_data.loglog_during_ax.loglog(local_act_time.distance_from_min[
        pace_maker.final_dist_beat_count[input_param.psd_plot_slider]],
        conduction_vel.param_dist_raw[pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider]], '.', base=10)
    
    print(pace_maker.final_dist_beat_count[input_param.psd_plot_slider])

    psd_data.psd_plots.tight_layout()
    psd_data.psd_plots.canvas.draw()


def plot_psd_welch(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data):
    # psd_data.psd_during_ax.cla()
    # x = np.ndarray((2,3))
    freq, Pxx = signal.welch(conduction_vel.param_dist_raw[pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider]].dropna(), fs=1.0, window='hann')
    # print(conduction_vel.param_dist_raw.drop(
    #     columns=['Electrode', 'X', 'Y']).dropna().values.flatten())
    # print(len(conduction_vel.param_dist_raw.drop(
    #     columns=['Electrode', 'X', 'Y']).dropna().values.flatten()))

    # freq, Pxx = signal.welch(conduction_vel.param_dist_raw.drop(
    #     columns=['Electrode', 'X', 'Y']).dropna().values.flatten(), fs=1.0, window='hann')
    # x must be a 1d array or series, fs = sampling frequency
    # freq, Pxx = signal.welch(x, fs=1.0)
    # print(freq)
    # print(Pxx_period)

    psd_data.psd_during_ax.loglog(freq, Pxx, label=pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider])
    psd_data.psd_during_ax.legend(loc="lower left", ncol=3)
    psd_data.psd_plots.tight_layout()
    psd_data.psd_plots.canvas.draw()
    # Welch may be the more robust method to use.