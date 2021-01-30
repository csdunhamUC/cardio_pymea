# Author: Christopher S. Dunham
# 1/29/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work
# New version utilizing fit to a 2D polynomial surface to calculate CV.
# Intent: eliminate artifacts caused by imposing an origin on CV.

import time
from numba import njit
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import sympy as sym


def calculate_conduction_velocity(analysisGUI, cm_beats, conduction_vel, local_act_time, heat_map, input_param, electrode_config):
    try:
        if hasattr(conduction_vel, 'param_dist_raw') is True:
            print("Clearing old CV data before running new calculation...")
            delattr(conduction_vel, 'param_dist_raw')

        start_time = time.process_time()
        # electrode_config.electrode_coords_x
        # electrode_config.electrode_coords_y
        # Y data to fit to: local_act_time.param_dist_normalized
        # "X" data consists of x, y coordinates from electrode_coords
        # print(len(electrode_config.electrode_coords_x))

        conduction_vel.cv_popt = [0]*int(cm_beats.beat_count_dist_mode[0])
        conduction_vel.cv_pcov = [0]*int(cm_beats.beat_count_dist_mode[0])
        nan_electrodes_idx = np.where(local_act_time.param_dist_normalized[
            'Beat 1'].isna())[0]
        x_elec = np.delete(electrode_config.electrode_coords_x, nan_electrodes_idx)
        y_elec = np.delete(electrode_config.electrode_coords_y, nan_electrodes_idx)
        elec_nan_removed = np.array([x_elec, y_elec])
        print(nan_electrodes_idx)
        print(nan_electrodes_idx[0])
        print(elec_nan_removed)
        print(elec_nan_removed[0])
        print(local_act_time.param_dist_normalized.iloc[4, 3])
        
        for num, beat in enumerate(local_act_time.param_dist_normalized.drop(
        columns=['Electrode', 'X', 'Y'])):
            conduction_vel.cv_popt[num], conduction_vel.cv_pcov[num] = curve_fit(
                two_dim_polynomial, elec_nan_removed, 
                local_act_time.param_dist_normalized[beat].dropna()
            )

        print(conduction_vel.cv_popt[0])
        print(conduction_vel.cv_popt[3])

        conduction_vel.param_dist_raw = local_act_time.distance_from_min.divide(
            local_act_time.param_dist_normalized.loc[
                :, local_act_time.final_dist_beat_count]).replace(
                    [np.inf, -np.inf], np.nan)
        # Need to add a placeholder value for the minimum channel; currently gives NaN as a consequence of division by zero.
        # Want/need to display the origin for heat map purposes.  Question is how to do this efficiently.

        conduction_vel.param_dist_raw_max = conduction_vel.param_dist_raw.max().max()
        conduction_vel.param_dist_raw_mean = np.nanmean(conduction_vel.param_dist_raw)

        conduction_vel.param_dist_raw.index = electrode_config.electrode_names
        conduction_vel.param_dist_raw.insert(0, 'Electrode', electrode_config.electrode_names)
        conduction_vel.param_dist_raw.insert(1, 'X', electrode_config.electrode_coords_x)
        conduction_vel.param_dist_raw.insert(2, 'Y', electrode_config.electrode_coords_y)

        end_time = time.process_time()
        calc_deriv(conduction_vel)
        print("CV calculation complete.")
        # print(end_time - start_time)
    except AttributeError:
        print("Please calculate local activation time first.")

def two_dim_polynomial(elec_nan_removed, a, b, c, d, e, f):
    # t = T(x,y) = ax**2 + by**2 + cxy + dx + ey + f
    x = elec_nan_removed[0]
    y = elec_nan_removed[1]
    # Equation from: PV Bayly et al, IEEE, 1998, doi:10.1109/10.668746
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

def calc_deriv(conduction_vel):
    # Calculate derivative w.r.t x
    x, y = sym.symbols('x, y', real=True)
    a, b, c, d, e, f = conduction_vel.cv_popt[0]
    # a = 1
    # b = 2
    # c = 3
    # d = 4
    # e = 5
    # f = 6
    t = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    for var in [x, y]:
        print("\\frac{\\partial g}{\\partial " + str(var) + "} =", 
            sym.latex(sym.simplify(t.diff(var))))

    print()