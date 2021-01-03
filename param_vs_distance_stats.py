# Author: Christopher S. Dunham
# 12/30/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
# import seaborn as sns
from matplotlib import pyplot as plt

def param_vs_distance_analysis(elecGUI120, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, cm_stats):
    input_param.sigma_value = int(elecGUI120.param_vs_dist_sigma_value.get())
    print("\n" + "Sigma value: " + str(input_param.sigma_value) + "\n")
    
    # Filter outliers for pacemaker.
    temp_pacemaker_pre_filtered = pace_maker.param_dist_normalized.drop(
        columns=['Electrode', 'X', 'Y'])
    temp_pacemaker_stddev = temp_pacemaker_pre_filtered.stack().std()
    print("Dataset Mean (Time Lag): " + str(pace_maker.param_dist_normalized_mean))
    print("Std Dev (Time Lag): " + str(temp_pacemaker_stddev))

    outlier_threshold_pm = (pace_maker.param_dist_normalized_mean + 
        (input_param.sigma_value * temp_pacemaker_stddev))
    cm_stats.pace_maker_filtered_data = pd.DataFrame(
        columns=temp_pacemaker_pre_filtered.columns, 
        index=temp_pacemaker_pre_filtered.index)
    cm_stats.pace_maker_filtered_data[temp_pacemaker_pre_filtered.columns] = np.where(
        (temp_pacemaker_pre_filtered.values > outlier_threshold_pm), 
        np.nan, temp_pacemaker_pre_filtered.values)
    
    # Filter outliers for LAT.
    temp_lat_pre_filtered = local_act_time.param_dist_normalized.drop(
        columns=['Electrode', 'X', 'Y'])
    temp_lat_stddev = temp_lat_pre_filtered.stack().std()
    print("Dataset Mean (LAT): " + str(local_act_time.param_dist_normalized_mean))
    print("Std Dev (LAT): " + str(temp_lat_stddev) + "\n")

    outlier_threshold_lat = (local_act_time.param_dist_normalized_mean + 
        (input_param.sigma_value * temp_lat_stddev))
    cm_stats.local_act_time_filtered_data = pd.DataFrame(
        columns=temp_lat_pre_filtered.columns, 
        index=temp_lat_pre_filtered.index)
    cm_stats.local_act_time_filtered_data[temp_lat_pre_filtered.columns] = np.where(
        (temp_lat_pre_filtered.values > outlier_threshold_lat), 
        np.nan, temp_lat_pre_filtered.values)
    
    # Filter outliers for dV/dt.
    temp_dvdt_pre_filtered = upstroke_vel.param_dist_normalized.drop(
        columns=['Electrode', 'X', 'Y'])
    temp_dvdt_stddev = temp_dvdt_pre_filtered.stack().std()
    print("Dataset Mean (dV/dt): " + str(upstroke_vel.param_dist_normalized_mean))
    print("Std Dev (dV/dt): " + str(temp_dvdt_stddev))

    outlier_threshold_dvdt = (upstroke_vel.param_dist_normalized_mean + 
        (input_param.sigma_value * temp_dvdt_stddev))
    cm_stats.upstroke_vel_filtered_data = pd.DataFrame(
        columns=temp_dvdt_pre_filtered.columns, 
        index=temp_dvdt_pre_filtered.index)
    cm_stats.upstroke_vel_filtered_data[temp_dvdt_pre_filtered.columns] = np.where(
        (temp_dvdt_pre_filtered.values > outlier_threshold_dvdt), 
        np.nan, temp_dvdt_pre_filtered.values)

    # Filter outliers for CV.
    temp_cv_pre_filtered = conduction_vel.param_dist_raw.drop(
        columns=['Electrode', 'X', 'Y'])
    temp_cv_stddev = temp_cv_pre_filtered.stack().std()
    print("Dataset Mean (CV): " + str(conduction_vel.param_dist_raw_mean))
    print("Std Dev (CV): " + str(temp_cv_stddev) + "\n")

    outlier_threshold_cv = (conduction_vel.param_dist_raw_mean + 
        (input_param.sigma_value * temp_cv_stddev))
    cm_stats.conduction_vel_filtered_data = pd.DataFrame(
        columns=temp_cv_pre_filtered.columns, index=temp_cv_pre_filtered.index)
    cm_stats.conduction_vel_filtered_data[temp_cv_pre_filtered.columns] = np.where(
        (temp_cv_pre_filtered.values > outlier_threshold_cv), 
        np.nan, temp_cv_pre_filtered.values)

    # Calculations for R-squared values (Pacemaker).
    cm_stats.slope_pm = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.intercept_pm = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.r_value_pm = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.p_value_pm = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.std_err_pm = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    
    for num, beat in enumerate(cm_stats.pace_maker_filtered_data):
        pm_without_nan = pace_maker.param_dist_normalized[beat].dropna()
        (cm_stats.slope_pm[num], cm_stats.intercept_pm[num], 
        cm_stats.r_value_pm[num], cm_stats.p_value_pm[num], 
        cm_stats.std_err_pm[num]) = sp.stats.linregress(
            local_act_time.distance_from_min.loc[pm_without_nan.index, beat], 
            pm_without_nan)

    # Calculations for R-squared values (Upstroke Velocity)
    cm_stats.slope_dvdt = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.intercept_dvdt = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.r_value_dvdt = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.p_value_dvdt = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.std_err_dvdt = np.zeros(int(cm_beats.beat_count_dist_mode[0]))

    for num, beat in enumerate(cm_stats.upstroke_vel_filtered_data):
        dvdt_without_nan = upstroke_vel.param_dist_normalized[beat].dropna()
        (cm_stats.slope_dvdt[num], cm_stats.intercept_dvdt[num], 
        cm_stats.r_value_dvdt[num], cm_stats.p_value_dvdt[num], 
        cm_stats.std_err_dvdt[num]) = sp.stats.linregress(
            local_act_time.distance_from_min.loc[dvdt_without_nan.index, beat],
            dvdt_without_nan)

    # Calculations for R-squared values (Local Activation Time)
    cm_stats.slope_lat = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.intercept_lat = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.r_value_lat = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.p_value_lat = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.std_err_lat = np.zeros(int(cm_beats.beat_count_dist_mode[0]))

    for num, beat in enumerate(cm_stats.local_act_time_filtered_data):
        lat_without_nan = local_act_time.param_dist_normalized[beat].dropna()
        (cm_stats.slope_lat[num], cm_stats.intercept_lat[num], 
        cm_stats.r_value_lat[num], cm_stats.p_value_lat[num], 
        cm_stats.std_err_lat[num]) = sp.stats.linregress(
            local_act_time.distance_from_min.loc[lat_without_nan.index, beat], 
            lat_without_nan)

    # Curve fitting and R-squared calculations for CV.
    # cm_stats.cv_popt = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    # cm_stats.cv_pcov = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    cm_stats.cv_popt = [0]*int(cm_beats.beat_count_dist_mode[0])
    cm_stats.cv_pcov = [0]*int(cm_beats.beat_count_dist_mode[0])
    cm_stats.r_value_cv = np.zeros(int(cm_beats.beat_count_dist_mode[0]))
    
    for num, beat in enumerate(cm_stats.conduction_vel_filtered_data):
        cv_without_nan = conduction_vel.param_dist_raw[beat].dropna()
        cv_without_nan = cv_without_nan.sort_values(ascending=True)
        x_sorted = local_act_time.distance_from_min.loc[cv_without_nan.index, 
            beat].sort_values(ascending=True)
        cm_stats.cv_popt[num], cm_stats.cv_pcov[num] = curve_fit(
            fitting_func, 
            x_sorted, cv_without_nan,
            method="trf"
        )
        residuals = cv_without_nan - fitting_func(x_sorted, *cm_stats.cv_popt[num])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((cv_without_nan - np.mean(cv_without_nan))**2)
        cm_stats.r_value_cv[num] = 1 - (ss_res / ss_tot)
    
    # print("CV POPT: " + str(cm_stats.cv_popt))
    # print("CV PCOV: " + str(cm_stats.cv_pcov))
    print("Done.")

    # Necessary operations:
    # 2) Calculate R^2 values, per beat, for each parameter vs distance

    # Necessary parameters:
    # 2) Percentile of R^2 to display/indicate

    # Necessary readouts:
    # 1) Dataset averages and standard deviation for each parameter (dV/dt, CV, PM, LAT)
    # 2) Dataset average and standard deviation of R^2 for each parameter (sorted high to low).
    # 3) Mode of PM (LAT) min & max channels.
    # 4) Mode of CV min and max channels.
    # 5) Number of unique min channels for PM (LAT)
    param_vs_distance_graphing(elecGUI120, cm_beats, pace_maker, upstroke_vel, 
    local_act_time, conduction_vel, input_param, cm_stats)


def param_vs_distance_graphing(elecGUI120, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param, cm_stats):
    input_param.stats_param_dist_slider = int(
        elecGUI120.param_vs_dist_beat_select.get()) - 1

    cm_stats.param_vs_dist_axis_pm.cla()
    cm_stats.param_vs_dist_axis_dvdt.cla()
    cm_stats.param_vs_dist_axis_lat.cla()
    cm_stats.param_vs_dist_axis_cv.cla()

    # Plot for Paarameter vs Distance.  Generates scatterplot, best-fit line and error bars.
    cm_stats.param_vs_dist_plot.suptitle(
        "Parameter vs. Distance from Minimum.  Beat " + 
        str(input_param.stats_param_dist_slider + 1) + ".") 
    # cm_stats.pacemaker_reg_plot = sns.regplot(
    #     x=local_act_time.distance_from_min[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
    #     y=cm_stats.pace_maker_filtered_data[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]],
    #     ax=cm_stats.param_vs_dist_axis_pm,
    #     color="crimson",
    #     label=("R-value: {0:.3f}".format(cm_stats.r_value_pm[input_param.stats_param_dist_slider]) + 
    #         "\n" + "Std Dev: {0:.2f}".format(pace_maker.param_dist_normalized[pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider]].std()))
    # )
    
    # Pacemaker plotting.
    cm_stats.param_vs_dist_axis_pm.scatter(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.pace_maker_filtered_data[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        c='red'
    )
    cm_stats.param_vs_dist_axis_pm.plot(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.intercept_pm[input_param.stats_param_dist_slider] + 
        cm_stats.slope_pm[input_param.stats_param_dist_slider] * 
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        c='black', label=("R-value: {0:.3f}".format(cm_stats.r_value_pm[
            input_param.stats_param_dist_slider]) + 
            "\n" + "Std Dev: {0:.2f}".format(pace_maker.param_dist_normalized[
                pace_maker.final_dist_beat_count[
                    input_param.stats_param_dist_slider]].std()))
    )
    print(pace_maker.final_dist_beat_count[input_param.stats_param_dist_slider])
    cm_stats.param_vs_dist_axis_pm.errorbar(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.intercept_pm[input_param.stats_param_dist_slider] + 
        cm_stats.slope_pm[input_param.stats_param_dist_slider] * 
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        yerr=pace_maker.param_dist_normalized[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]].std(),
        ecolor='black', alpha=0.1, linewidth=0.8
    )
    cm_stats.param_vs_dist_axis_pm.set(title="Pacemaker", ylabel="Time lag (ms)")
    cm_stats.param_vs_dist_axis_pm.legend(loc='upper left')
    
    # Upstroke velocity plotting.
    cm_stats.param_vs_dist_axis_dvdt.scatter(
        local_act_time.distance_from_min[upstroke_vel.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.upstroke_vel_filtered_data[upstroke_vel.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        c='green'
    )
    cm_stats.param_vs_dist_axis_dvdt.plot(
        local_act_time.distance_from_min[upstroke_vel.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.intercept_dvdt[input_param.stats_param_dist_slider] +
        cm_stats.slope_dvdt[input_param.stats_param_dist_slider] * 
        local_act_time.distance_from_min[upstroke_vel.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        c='black', label=("R-value: {0:.3f}".format(cm_stats.r_value_dvdt[
            input_param.stats_param_dist_slider]) + 
        "\n" + "Std Dev: {0:.2f}".format(upstroke_vel.param_dist_normalized[
            upstroke_vel.final_dist_beat_count[
                input_param.stats_param_dist_slider]].std())),
    )
    cm_stats.param_vs_dist_axis_dvdt.errorbar(
        local_act_time.distance_from_min[upstroke_vel.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.intercept_dvdt[input_param.stats_param_dist_slider] +
        cm_stats.slope_dvdt[input_param.stats_param_dist_slider] * 
        local_act_time.distance_from_min[upstroke_vel.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        yerr=upstroke_vel.param_dist_normalized[upstroke_vel.final_dist_beat_count[
            input_param.stats_param_dist_slider]].std(),
        ecolor='black', alpha=0.1, linewidth=0.8
    )
    cm_stats.param_vs_dist_axis_dvdt.set(title="Upstroke Velocity", ylabel="μV/ms")
    cm_stats.param_vs_dist_axis_dvdt.legend(loc='upper left')
    
    # Local activation time plotting.
    cm_stats.param_vs_dist_axis_lat.scatter(
        local_act_time.distance_from_min[local_act_time.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.local_act_time_filtered_data[local_act_time.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        c='orange'
    )
    cm_stats.param_vs_dist_axis_lat.plot(
        local_act_time.distance_from_min[local_act_time.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.intercept_lat[input_param.stats_param_dist_slider] + 
        cm_stats.slope_lat[input_param.stats_param_dist_slider] * 
        local_act_time.distance_from_min[local_act_time.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        c='black', label=("R-value: {0:.3f}".format(cm_stats.r_value_lat[
            input_param.stats_param_dist_slider]) + 
        "\n" + "Std Dev: {0:.2f}".format(local_act_time.param_dist_normalized[
            local_act_time.final_dist_beat_count[
                input_param.stats_param_dist_slider]].std()))
    )
    cm_stats.param_vs_dist_axis_lat.errorbar(
        local_act_time.distance_from_min[local_act_time.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.intercept_lat[input_param.stats_param_dist_slider] + 
        cm_stats.slope_lat[input_param.stats_param_dist_slider] * 
        local_act_time.distance_from_min[local_act_time.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        yerr=local_act_time.param_dist_normalized[
            local_act_time.final_dist_beat_count[
                input_param.stats_param_dist_slider]].std(),
        ecolor='black', alpha=0.1, linewidth=0.8
    )
    cm_stats.param_vs_dist_axis_lat.set(title="Local Activation Time", 
        xlabel="Distance from origin (μm)", ylabel="Activation time (ms)")
    cm_stats.param_vs_dist_axis_lat.legend(loc='upper left')

    # Conduction velocity plotting.
    cm_stats.param_vs_dist_axis_cv.scatter(
        local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        cm_stats.conduction_vel_filtered_data[local_act_time.final_dist_beat_count[
            input_param.stats_param_dist_slider]],
        c='blue'
    )
    L, k, x0 = cm_stats.cv_popt[input_param.stats_param_dist_slider]
    print(L, k, x0)
    y_fit = fitting_func(local_act_time.distance_from_min[
        pace_maker.final_dist_beat_count[
            input_param.stats_param_dist_slider]].sort_values(ascending=True),
        L, k, x0)
    x_sorted = local_act_time.distance_from_min[pace_maker.final_dist_beat_count[
        input_param.stats_param_dist_slider]].sort_values(ascending=True)
    cm_stats.param_vs_dist_axis_cv.plot(
        x_sorted, y_fit, linestyle='-', c='black',
        label=("R-value: {0:.3f}".format(cm_stats.r_value_cv[
            input_param.stats_param_dist_slider]) + "\n" + 
            "Std Dev: {0:.2f}".format(conduction_vel.param_dist_raw[
                pace_maker.final_dist_beat_count[
                    input_param.stats_param_dist_slider]].std()))
    )
    cm_stats.param_vs_dist_axis_cv.errorbar(
        x_sorted, y_fit, yerr=conduction_vel.param_dist_raw[
            pace_maker.final_dist_beat_count[
                input_param.stats_param_dist_slider]].std(),
        ecolor='black', alpha=0.1, linewidth=0.8
    )
    cm_stats.param_vs_dist_axis_cv.set(title="Conduction Velocity", 
        xlabel="Distance from origin (μm)", ylabel="μm/ms")
    cm_stats.param_vs_dist_axis_cv.legend(loc='lower right')

    # Draw the plots.
    cm_stats.param_vs_dist_plot.tight_layout()
    cm_stats.param_vs_dist_plot.canvas.draw()

# Function used by curve_fit from scipy.optimize for conduction velocity.
# def fitting_func(x, a, b, c):
#     return a*x + b*x**2 + c

# Potential saturation equation
def fitting_func(x, a, b, c):
    return a*x / (b + (x/c))
# Tried to fit data to a logistic equation, e.g. logistic growth, but doesn't
# fit very well. That is what the function below is for.
# def fitting_func(x, L, k, x0):
#     return L/(1 + np.exp(-k * (x-x0)))