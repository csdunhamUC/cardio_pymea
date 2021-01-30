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
import sympy


def calculate_conduction_velocity(analysisGUI, conduction_vel, local_act_time, heat_map, input_param, electrode_config):
    try:
        if hasattr(conduction_vel, 'param_dist_raw') is True:
            print("Clearing old CV data before running new calculation...")
            delattr(conduction_vel, 'param_dist_raw')

        start_time = time.process_time()
        # electrode_config.electrode_coords_x
        # electrode_config.electrode_coords_y
        # Y data to fit to: local_act_time.param_dist_normalized
        # 

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
        print("CV calculation complete.")
        # print(end_time - start_time)
    except AttributeError:
        print("Please calculate local activation time first.")

def two_dim_polynomial(x, y, t, a, b, c, d, e, f):
    # T(x,y) = ax**2 + by**2 + cxy + dx + ey + f
    # Equation from: PV Bayly et al, IEEE, 1998, doi:10.1109/10.668746
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f