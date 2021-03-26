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
    # beat_amp_int.raw_beat_interval
    # beat_amp_int.raw_delta_beat_amp
    # pace_maker.param_dist_normalized_per_beat_max (row vector, goes up to # of beats)
    # pace_maker.param_dist_normalized (goes up to # of beats + elec label/coords)
    # analysisGUI.pcaWindow.paramPlot.axes, and paramPlot.fig

    # Maximum time lag with end beat removed, done so in order to match dimensions with
    # beat interval and delta beat amplitude.
    temp_time_lag = pace_maker.param_dist_normalized_per_beat_max.drop(
        [pace_maker.final_dist_beat_count[-1]])
    interval_trans = beat_amp_int.raw_beat_interval.T
    amplitude_trans = beat_amp_int.raw_delta_beat_amp.T

    interval_labels = ["Interval" for idx in interval_trans.index]
    amp_labels = ["Amplitude" for idx in amplitude_trans.index]
    interval_trans["Label"] = interval_labels
    amplitude_trans["Label"] = amp_labels

    test_frame = pd.concat([interval_trans, amplitude_trans], ignore_index=True)

    # pre_norm = np.array([beat_amp_int.beat_interval, 
    #     beat_amp_int.delta_beat_amp]).T

    # pre_norm_df = pd.DataFrame(pre_norm, columns=['Beat Interval', 
    #     'Delta Beat Amp'])

    # norm_array = StandardScaler().fit_transform(pre_norm_df.values)
    norm_array = StandardScaler().fit_transform(
        test_frame.drop(columns=["Label"]).values)

    norm_df = pd.DataFrame(norm_array)
    norm_df["Label"] = test_frame["Label"].values
    norm_df.columns = test_frame.columns

    # norm_df = pd.DataFrame(norm_array, columns=["Normalized Beat Interval", 
    #     "Normalized Delta Beat Amp"])
    print(np.mean(norm_df))
    print(np.std(norm_df))
    print(norm_df.head())
    print(norm_df.tail())

    pca_execute = PCA(n_components=2)
    norm_without_label = norm_df.drop(columns=["Label"])
    # pcaAmpInt = pd.DataFrame(pca_execute.fit_transform(norm_df), 
    #     columns=["Principal Component 1", "Principal Component 2"])
    pcaAmpInt = pd.DataFrame(pca_execute.fit_transform(norm_without_label), 
        columns=["Principal Component 1", "Principal Component 2"])

    print("Explained variation per principal component: {}".format(
        pca_execute.explained_variance_ratio_))

    targets = ["Interval", "Amplitude"]
    colors = ["b", "r"]
    for target, color in zip(targets, colors):
        indices_to_keep = test_frame["Label"] == target
        analysisGUI.pcaWindow.paramPlot.axes.scatter(
            pcaAmpInt.loc[indices_to_keep, "Principal Component 1"], 
            pcaAmpInt.loc[indices_to_keep, "Principal Component 2"],
            c = color)
    
    analysisGUI.pcaWindow.paramPlot.axes.legend(targets)
    analysisGUI.pcaWindow.paramPlot.axes.set(
        title="Principal Component Analysis of Beat Interval and ΔBeat Amp",
        xlabel="Principal Component 1", ylabel="Principal Component 2")

    analysisGUI.pcaWindow.paramPlot.draw()


def pca_plot(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    print("Plotting placeholder.")

# Previous version.
    # pre_norm = np.array([beat_amp_int.beat_interval, beat_amp_int.delta_beat_amp]).T
    # # print(pace_maker.param_dist_normalized_per_beat_max.drop(
    # #     [pace_maker.final_dist_beat_count[-1]]))

    # pre_norm_df = pd.DataFrame(pre_norm, columns=['Beat Interval', 'Delta Beat Amp'])

    # norm_array = StandardScaler().fit_transform(pre_norm_df.values)
    # norm_df = pd.DataFrame(norm_array, columns=["Normalized Beat Interval", "Normalized Delta Beat Amp"])
    # print(np.mean(norm_df))
    # print(np.std(norm_df))

    # pca_execute = PCA(n_components=2)

    # pcaAmpInt = pd.DataFrame(pca_execute.fit_transform(norm_df), 
    #     columns=["Principal Component 1", "Principal Component 2"])

    # print("Explained variation per principal component: {}".format(
    #     pca_execute.explained_variance_ratio_))

    # analysisGUI.pcaWindow.paramPlot.axes.scatter(
    #     pcaAmpInt.loc[:, "Principal Component 1"], 
    #     pcaAmpInt.loc[:, "Principal Component 2"])
    # analysisGUI.pcaWindow.paramPlot.axes.set(
    #     title="Principal Component Analysis of Beat Interval and ΔBeat Amp",
    #     xlabel="Principal Component 1", ylabel="Principal Component 2")

    # analysisGUI.pcaWindow.paramPlot.draw()