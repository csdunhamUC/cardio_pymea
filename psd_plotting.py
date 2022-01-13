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


def psd_plotting(analysisGUI, cm_beats, electrode_config, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data):
    try:
        # Get slider value.
        input_param.psd_plot_slider = analysisGUI.psdWindow.paramSlider.value()
        # Set file name.
        # analysisGUI.psd_file_name.set(analysisGUI.file_name_label.cget("text"))
        
        # Update entry boxes for PSD Plotting window.
        if analysisGUI.psdWindow.startBeat.count() != len(
        local_act_time.final_dist_beat_count):
            analysisGUI.psdWindow.startBeat.clear()
            analysisGUI.psdWindow.endBeat.clear()
            analysisGUI.psdWindow.elecSelect.clear()

        if analysisGUI.psdWindow.startBeat.count() < 2:
            analysisGUI.psdWindow.startBeat.addItems(
                local_act_time.final_dist_beat_count)
            analysisGUI.psdWindow.endBeat.addItems(
                local_act_time.final_dist_beat_count)
            analysisGUI.psdWindow.elecSelect.addItems(
                electrode_config.electrode_names)

        start_beat = analysisGUI.psdWindow.startBeat.currentText()
        end_beat = analysisGUI.psdWindow.endBeat.currentText()
        elec_choice = analysisGUI.psdWindow.elecSelect.currentText()

        param_choices = {"Orig. Signal": cm_beats.y_axis, 
            "Cond. Vel.": conduction_vel.param_dist_raw,
            "Up. Vel.": upstroke_vel.param_dist_normalized, 
            "Pacemaker": pace_maker.param_dist_normalized, 
            "Local AT": local_act_time.param_dist_normalized}

        if len(analysisGUI.psdWindow.paramPlot.axis1.lines) >= 12:
            analysisGUI.psdWindow.paramPlot.axis1.cla()
            analysisGUI.psdWindow.paramPlot.axis2.cla()

        plot_log_vs_log(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
            local_act_time, conduction_vel, input_param, psd_data, start_beat, 
            end_beat, elec_choice, param_choices, electrode_config)
        
        plot_psd_welch(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
            local_act_time, conduction_vel, input_param, psd_data, start_beat, 
            end_beat, elec_choice, param_choices, electrode_config)
    except AttributeError:
        print("No data.")


def plot_log_vs_log(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data, start_beat, end_beat,
elec_choice, param_choices, electrode_config):
    analysisGUI.psdWindow.paramPlot.axis1.cla()

    check_param = analysisGUI.psdWindow.paramSelect.currentText()
    if check_param == "Orig. Signal":
        param_plotted = param_choices.get(check_param)
        start_index = int(pace_maker.param_dist_raw.loc[elec_choice, start_beat])
        end_index = int(pace_maker.param_dist_raw.loc[elec_choice, end_beat])
        elec_index = electrode_config.electrode_names.index(elec_choice)
        analysisGUI.psdWindow.paramPlot.axis1.loglog(param_plotted.iloc[
            start_index:end_index, elec_index], label=elec_choice)
        analysisGUI.psdWindow.paramPlot.axis1.legend(loc="lower left", ncol=6)
    
    elif check_param == "Cond. Vel.":
        param_plotted = param_choices.get(check_param)
        # Produce log vs log plot of parameter vs distance.
        analysisGUI.psdWindow.paramPlot.axis1.loglog(local_act_time.distance_from_min[
            pace_maker.final_dist_beat_count[input_param.psd_plot_slider]],
            param_plotted[pace_maker.final_dist_beat_count[
                input_param.psd_plot_slider]], '.', base=10, 
                label=pace_maker.final_dist_beat_count[
                    input_param.psd_plot_slider])
        analysisGUI.psdWindow.paramPlot.axis1.set(title="Log CV vs Log Distance", 
            xlabel="Log μm", ylabel="Log CV")
        analysisGUI.psdWindow.paramPlot.axis1.legend(loc="lower left", ncol=6)
    
    elif check_param == "Pacemaker":
        param_plotted = param_choices.get(check_param)
        # Produce log vs log plot of parameter vs distance.
        analysisGUI.psdWindow.paramPlot.axis1.loglog(local_act_time.distance_from_min[
            pace_maker.final_dist_beat_count[input_param.psd_plot_slider]],
            param_plotted[pace_maker.final_dist_beat_count[
                input_param.psd_plot_slider]], '.', base=10, 
                label=pace_maker.final_dist_beat_count[
                    input_param.psd_plot_slider])
        analysisGUI.psdWindow.paramPlot.axis1.set(title="Log PM vs Log Distance",
            xlabel="Log μm", ylabel="Log PM")
        analysisGUI.psdWindow.paramPlot.axis1.legend(loc="lower left", ncol=6)


def plot_psd_welch(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, psd_data, start_beat, end_beat,
elec_choice, param_choices, electrode_config):
    analysisGUI.psdWindow.paramPlot.axis2.cla()

    check_param = analysisGUI.psdWindow.paramSelect.currentText()
    if check_param == "Orig. Signal":
        param_plotted = param_choices.get(check_param)
        start_index = int(pace_maker.param_dist_raw.loc[elec_choice, start_beat])
        end_index = int(pace_maker.param_dist_raw.loc[elec_choice, end_beat])
        elec_index = electrode_config.electrode_names.index(elec_choice)
        freq, Pxx = signal.periodogram(param_plotted.iloc[start_index:end_index, 
            elec_index], fs=1.0, window='hann')
        
        analysisGUI.psdWindow.paramPlot.axis2.loglog(freq, Pxx, label=elec_choice)
        analysisGUI.psdWindow.paramPlot.axis2.set(title="Periodogram of Signal", 
            xlabel="Log of Frequency (Hz)", ylabel="Log of PSD of Signal")
        analysisGUI.psdWindow.paramPlot.axis2.legend(loc="lower left", ncol=6)
    
    elif check_param == "Cond. Vel.":
        param_plotted = param_choices.get(check_param)
        freq, Pxx = signal.welch(param_plotted[pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider]].dropna(), fs=1.0, window='hann')
        
        analysisGUI.psdWindow.paramPlot.axis2.loglog(freq, Pxx, 
            label=pace_maker.final_dist_beat_count[input_param.psd_plot_slider])
        analysisGUI.psdWindow.paramPlot.axis2.set(title="Welch PSD of CV", 
            xlabel="Log of Frequency (Hz)", ylabel="Log of PSD of CV")
        analysisGUI.psdWindow.paramPlot.axis2.legend(loc="lower left", ncol=6)
    
    elif check_param == "Pacemaker":
        param_plotted = param_choices.get(check_param)
        freq, Pxx = signal.welch(param_plotted[pace_maker.final_dist_beat_count[
            input_param.psd_plot_slider]].dropna(), fs=1.0, window='hann')
        
        analysisGUI.psdWindow.paramPlot.axis2.loglog(freq, Pxx, 
            label=pace_maker.final_dist_beat_count[input_param.psd_plot_slider])
        analysisGUI.psdWindow.paramPlot.axis2.set(title="Welch PSD of PM", 
            xlabel="Log of Frequency (Hz)", ylabel="Log of PSD of PM")
        analysisGUI.psdWindow.paramPlot.axis2.legend(loc="lower left", ncol=6)

    analysisGUI.psdWindow.paramPlot.fig.tight_layout()
    analysisGUI.psdWindow.paramPlot.draw()