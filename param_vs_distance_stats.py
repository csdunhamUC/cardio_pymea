# Author: Christopher S. Dunham
# 12/30/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import numpy as np
import pandas as pd

def param_vs_distance_analysis(elecGUI120, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats):
    input_param.sigma_value = int(elecGUI120.param_vs_dist_sigma_value.get())
    print("Sigma value: " + str(input_param.sigma_value) + "\n")
    
    # Filter outliers for pacemaker.
    temp_pacemaker_pre_filtered = pace_maker.param_dist_normalized.drop(columns=['Electrode', 'X', 'Y'])
    temp_pacemaker_stddev = temp_pacemaker_pre_filtered.stack().std()
    print("Mean (Time Lag): " + str(pace_maker.param_dist_normalized_mean))
    print("Std Dev (Time Lag): " + str(temp_pacemaker_stddev))

    outlier_threshold_pm = (pace_maker.param_dist_normalized_mean + (input_param.sigma_value * temp_pacemaker_stddev))
    cm_stats.pace_maker_filtered_data = pd.DataFrame(columns=temp_pacemaker_pre_filtered.columns, index=temp_pacemaker_pre_filtered.index)
    cm_stats.pace_maker_filtered_data[temp_pacemaker_pre_filtered.columns] = np.where((temp_pacemaker_pre_filtered.values > outlier_threshold_pm), 
        np.nan, temp_pacemaker_pre_filtered.values)
    
    # Filter outliers for LAT.
    temp_lat_pre_filtered = local_act_time.param_dist_normalized.drop(columns=['Electrode', 'X', 'Y'])
    temp_lat_stddev = temp_lat_pre_filtered.stack().std()
    print("Mean (LAT): " + str(local_act_time.param_dist_normalized_mean))
    print("Std Dev (LAT): " + str(temp_lat_stddev) + "\n")

    outlier_threshold_lat = (local_act_time.param_dist_normalized_mean + (input_param.sigma_value * temp_lat_stddev))
    cm_stats.local_act_time_filtered_data = pd.DataFrame(columns=temp_lat_pre_filtered.columns, index=temp_lat_pre_filtered.index)
    cm_stats.local_act_time_filtered_data[temp_lat_pre_filtered.columns] = np.where((temp_lat_pre_filtered.values > outlier_threshold_lat), 
        np.nan, temp_lat_pre_filtered.values)
    
    # Filter outliers for dV/dt.
    temp_dvdt_pre_filtered = upstroke_vel.param_dist_normalized.drop(columns=['Electrode', 'X', 'Y'])
    temp_dvdt_stddev = temp_dvdt_pre_filtered.stack().std()
    print("Mean (dV/dt): " + str(upstroke_vel.param_dist_normalized_mean))
    print("Std Dev (dV/dt): " + str(temp_dvdt_stddev))

    outlier_threshold_dvdt = (upstroke_vel.param_dist_normalized_mean + (input_param.sigma_value * temp_dvdt_stddev))
    cm_stats.upstroke_vel_filtered_data = pd.DataFrame(columns=temp_dvdt_pre_filtered.columns, index=temp_dvdt_pre_filtered.index)
    cm_stats.upstroke_vel_filtered_data[temp_dvdt_pre_filtered.columns] = np.where((temp_dvdt_pre_filtered.values > outlier_threshold_dvdt), 
        np.nan, temp_dvdt_pre_filtered.values)

    # Filter outliers for CV.
    temp_cv_pre_filtered = conduction_vel.param_dist_raw.drop(columns=['Electrode', 'X', 'Y'])
    temp_cv_stddev = temp_cv_pre_filtered.stack().std()
    print("Mean (CV): " + str(conduction_vel.param_dist_raw_mean))
    print("Std Dev (CV): " + str(temp_cv_stddev) + "\n")

    outlier_threshold_cv = (conduction_vel.param_dist_raw_mean + (input_param.sigma_value * temp_cv_stddev))
    cm_stats.conduction_vel_filtered_data = pd.DataFrame(columns=temp_cv_pre_filtered.columns, index=temp_cv_pre_filtered.index)
    cm_stats.conduction_vel_filtered_data[temp_cv_pre_filtered.columns] = np.where((temp_cv_pre_filtered.values > outlier_threshold_cv), 
        np.nan, temp_cv_pre_filtered.values)

    
    print("Done x4")
    # x-values @: local_act_time.distance_from_min
    # y-values @: pace_maker.param_dist_normalized, upstroke_vel.param_dist_normalized,

    # Necessary operations:
    # 1) Elimination of outliers (calculate mean, stdev, remove data > mean*3 sigma)
    # 2) Calculate R^2 values, per beat, for each parameter vs distance

    # Necessary parameters:
    # 1) Sigma
    # 2) Percentile of R^2 to display/indicate

    # Necessary readouts:
    # 1) Dataset averages and standard deviation for each parameter (dV/dt, CV, PM, LAT)
    # 2) Dataset average and standard deviation of R^2 for each parameter (sorted high to low).
    # 3) Mode of PM (LAT) min & max channels.
    # 4) Mode of CV min and max channels.
    # 5) Number of unique min channels for PM (LAT)
    param_vs_distance_graphing(elecGUI120, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats)


def param_vs_distance_graphing(elecGUI120, cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats):
    input_param.stats_param_dist_slider = int(elecGUI120.param_vs_dist_beat_select.get()) - 1

    cm_stats.param_vs_dist_axis_pm.cla()
    cm_stats.param_vs_dist_axis_dvdt.cla()
    cm_stats.param_vs_dist_axis_lat.cla()
    cm_stats.param_vs_dist_axis_cv.cla()

    # mask_coords = ~np.isnan(local_act_time.distance_from_min[input_param.stats_param_dist_slider])
    cm_stats.param_vs_dist_plot.suptitle(
        "Parameter vs. Distance from Minimum.  Beat: " + str(input_param.stats_param_dist_slider + 1) + ".")
    cm_stats.param_vs_dist_axis_pm.scatter(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
        cm_stats.pace_maker_filtered_data[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
        c='red')
    cm_stats.param_vs_dist_axis_pm.set(title="Pacemaker", ylabel="Time lag (ms)")
    cm_stats.param_vs_dist_axis_dvdt.scatter(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
        cm_stats.upstroke_vel_filtered_data[upstroke_vel.final_dist_beat_count[input_param.stats_param_dist_slider]],
        c='green')
    cm_stats.param_vs_dist_axis_dvdt.set(title="Upstroke Velocity", ylabel="μV/ms")
    cm_stats.param_vs_dist_axis_lat.scatter(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
        cm_stats.local_act_time_filtered_data[local_act_time.final_dist_beat_count[input_param.stats_param_dist_slider]],
        c='orange')
    cm_stats.param_vs_dist_axis_lat.set(title="Local Activation Time", xlabel="Distance from origin (μm)", ylabel="Activation time (ms)")
    cm_stats.param_vs_dist_axis_cv.scatter(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
        cm_stats.conduction_vel_filtered_data[local_act_time.final_dist_beat_count[input_param.stats_param_dist_slider]],
        c='blue')
    cm_stats.param_vs_dist_axis_cv.set(title="Conduction Velocity", xlabel="Distance from origin (μm)", ylabel="μm/ms")

    cm_stats.param_vs_dist_plot.tight_layout()
    cm_stats.param_vs_dist_plot.canvas.draw()