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

from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy import stats
import seaborn as sns

def find_T_wave(analysisGUI, cm_beats, field_potential, heat_map, input_param):
    # Step 1: Apply bandpass filter to raw data.
    # Defaults: Order = 4, Low-Pass Freq = 30Hz, High-Pass Freq = 0.5Hz
    cm_beats.bp_filt_y = bandpass_filter(cm_beats, input_param)
    print("Done.")


def graph_fpd(analysisGUI, cm_beats, field_potential, heat_map, input_param):
    
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

def bandpass_filter(cm_beats, input_param, bworth_ord=4, low_cutoff_freq=30, 
high_cutoff_freq=0.5):
    
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
    
    
