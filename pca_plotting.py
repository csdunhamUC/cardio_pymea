# Author: Christopher S. Dunham
# 3/18/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

# PCA module for cardiomyocyte-MEA analysis

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd


def pca_data_prep(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    analysisGUI.pcaWindow.paramPlot.axes.cla()
    
    # Variables of note:
    # beat_amp_int.beat_amp (goes up to # of beats + elec label/coords)
    # beat_amp_int.beat_interval (row vector, goes up to # of beats - 1)
    # beat_amp_int.delta_beat_amp (row vector, goes up to # of beats - 1)
    # pace_maker.param_dist_normalized_per_beat_max (row vector, goes up to # of beats)
    # pace_maker.param_dist_normalized (goes up to # of beats + elec label/coords)
    # analysisGUI.pcaWindow.paramPlot.axes, and paramPlot.fig

    pre_norm = np.array([beat_amp_int.beat_interval, beat_amp_int.delta_beat_amp]).T
    # print(pace_maker.param_dist_normalized_per_beat_max.drop(
    #     [pace_maker.final_dist_beat_count[-1]]))

    pre_norm_df = pd.DataFrame(pre_norm, columns=['Beat Interval', 'Delta Beat Amp'])

    norm_array = StandardScaler().fit_transform(pre_norm_df.values)
    norm_df = pd.DataFrame(norm_array, columns=["Normalized Beat Interval", "Normalized Delta Beat Amp"])
    print(np.mean(norm_df))
    print(np.std(norm_df))

    pca_execute = PCA(n_components=2)

    pcaAmpInt = pd.DataFrame(pca_execute.fit_transform(norm_df), 
        columns=["Principal Component 1", "Principal Component 2"])

    print("Explained variation per principal component: {}".format(
        pca_execute.explained_variance_ratio_))

    analysisGUI.pcaWindow.paramPlot.axes.scatter(
        pcaAmpInt.loc[:, "Principal Component 1"], 
        pcaAmpInt.loc[:, "Principal Component 2"])
    analysisGUI.pcaWindow.paramPlot.axes.set(
        title="Principal Component Analysis of Beat Interval and ΔBeat Amp",
        xlabel="Principal Component 1", ylabel="Principal Component 2")

    analysisGUI.pcaWindow.paramPlot.draw()


def pca_plot(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    print("Plotting placeholder.")