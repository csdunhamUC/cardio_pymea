# Author: Christopher S. Dunham
# Date: 3/18/2021
# Principal Investigator: James K. Gimzewski
# Organization: University of California, Los Angeles
# Department of Chemistry and Biochemistry
# Original work by CSD
# PCA module for cardiomyocyte MEA analysis

###############################################################################
# Please note:
# This is unfinished work as of 3/23/2022.
###############################################################################


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd


def pca_data_prep(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    try:
        analysisGUI.pcaWindow.paramPlot.axis1.cla()

        # Maximum time lag with end beat removed, done so in order to match dimensions with
        # beat interval and delta beat amplitude.
        temp_time_lag = pace_maker.param_dist_normalized_per_beat_max.drop(
            [pace_maker.final_dist_beat_count[-1]])
        interval_trans = beat_amp_int.raw_beat_interval.T
        amplitude_trans = beat_amp_int.raw_delta_beat_amp.T

        interval_labels = ["Interval" for idx in interval_trans.index]
        amp_labels = ["Amplitude" for idx in amplitude_trans.index]
        sample_labels = ["Sample" for idx in amplitude_trans.index]
        interval_trans["Label"] = interval_labels
        amplitude_trans["Label"] = amp_labels
        print(len(sample_labels))

        pre_norm = np.array([beat_amp_int.beat_interval, 
            beat_amp_int.delta_beat_amp]).T

        pre_norm_df = pd.DataFrame(pre_norm, columns=['Beat Interval', 
            'Delta Beat Amp'])

        pre_norm = np.array([beat_amp_int.beat_interval, beat_amp_int.delta_beat_amp]).T

        pre_norm_df = pd.DataFrame(pre_norm, columns=['Beat Interval', 'Delta Beat Amp'])

        norm_array = StandardScaler().fit_transform(pre_norm_df.values)
        norm_df = pd.DataFrame(norm_array, columns=["Normalized Beat Interval", "Normalized Delta Beat Amp"])
        print(np.mean(norm_df))
        print(np.std(norm_df))
        print(np.shape(norm_df))

        pca_execute = PCA(n_components=2)

        pcaAmpInt = pd.DataFrame(pca_execute.fit_transform(norm_df), 
            columns=["Principal Component 1", "Principal Component 2"])

        print("Explained variation per principal component: {}".format(
            pca_execute.explained_variance_ratio_))

        analysisGUI.pcaWindow.paramPlot.axis1.scatter(
            pcaAmpInt.loc[:, "Principal Component 1"], 
            pcaAmpInt.loc[:, "Principal Component 2"])
        analysisGUI.pcaWindow.paramPlot.axis1.set(
            title="Principal Component Analysis of Beat Interval and ΔBeat Amp",
            xlabel="Principal Component 1", ylabel="Principal Component 2")
        
        analysisGUI.pcaWindow.paramPlot.draw()
    except AttributeError:
        print("No data.")


def pca_plot(analysisGUI, cm_beats, beat_amp_int, pace_maker, 
local_act_time, heat_map, input_param, electrode_config):
    print("Plotting placeholder.")

# Previous version w/ labeling.
# # test_frame = pd.concat([interval_trans, amplitude_trans], ignore_index=True)

#     # pre_norm = np.array([beat_amp_int.beat_interval, 
#     #     beat_amp_int.delta_beat_amp]).T

#     # pre_norm_df = pd.DataFrame(pre_norm, columns=['Beat Interval', 
#     #     'Delta Beat Amp'])

#     # norm_array = StandardScaler().fit_transform(pre_norm_df.values)
#     norm_array = StandardScaler().fit_transform(
#         test_frame.drop(columns=["Label"]).values)

#     norm_df = pd.DataFrame(norm_array)
#     norm_df["Label"] = test_frame["Label"].values
#     norm_df.columns = test_frame.columns

#     # norm_df = pd.DataFrame(norm_array, columns=["Normalized Beat Interval", 
#     #     "Normalized Delta Beat Amp"])
#     print(np.mean(norm_df))
#     print(np.std(norm_df))
#     print(norm_df.head())
#     print(norm_df.tail())

#     pca_execute = PCA(n_components=2)
#     norm_without_label = norm_df.drop(columns=["Label"])
#     # pcaAmpInt = pd.DataFrame(pca_execute.fit_transform(norm_df), 
#     #     columns=["Principal Component 1", "Principal Component 2"])
#     pcaAmpInt = pd.DataFrame(pca_execute.fit_transform(norm_without_label), 
#         columns=["Principal Component 1", "Principal Component 2"])

#     print("Explained variation per principal component: {}".format(
#         pca_execute.explained_variance_ratio_))

#     targets = ["Interval", "Amplitude"]
#     colors = ["b", "r"]
#     for target, color in zip(targets, colors):
#         indices_to_keep = test_frame["Label"] == target
#         analysisGUI.pcaWindow.paramPlot.axis1.scatter(
#             pcaAmpInt.loc[indices_to_keep, "Principal Component 1"], 
#             pcaAmpInt.loc[indices_to_keep, "Principal Component 2"],
#             c = color)

#     analysisGUI.pcaWindow.paramPlot.axis1.legend(targets)
