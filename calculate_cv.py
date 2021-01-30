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
        x_elec = np.delete(
            electrode_config.electrode_coords_x, nan_electrodes_idx)
        y_elec = np.delete(
            electrode_config.electrode_coords_y, nan_electrodes_idx)
        elec_nan_removed = np.array([x_elec, y_elec])
        # print(nan_electrodes_idx)
        # print(nan_electrodes_idx[0])
        # print(elec_nan_removed)
        # print(elec_nan_removed[0])
        # print(local_act_time.param_dist_normalized.iloc[4, 3])
        
        for num, beat in enumerate(local_act_time.param_dist_normalized.drop(
        columns=['Electrode', 'X', 'Y'])):
            conduction_vel.cv_popt[num], conduction_vel.cv_pcov[num] = curve_fit(
                two_dim_polynomial, elec_nan_removed, 
                local_act_time.param_dist_normalized[beat].dropna()
            )

        print(conduction_vel.cv_popt[0])
        print(conduction_vel.cv_popt[3])

        calc_deriv(elec_nan_removed, cm_beats, local_act_time, conduction_vel)

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
        print(end_time - start_time)
    except AttributeError:
        print("Please calculate local activation time first.")


def two_dim_polynomial(elec_nan_removed, a, b, c, d, e, f):
    # t = T(x,y) = ax**2 + by**2 + cxy + dx + ey + f
    x = elec_nan_removed[0]
    y = elec_nan_removed[1]
    # Equation from: PV Bayly et al, IEEE, 1998, doi:10.1109/10.668746
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


# Calculate derivatives w.r.t x and y for each beat.
def calc_deriv(elec_nan_removed, cm_beats, local_act_time, conduction_vel):
    x, y = sym.symbols('x, y', real=True)

    t_deriv_expr_x = [0]*int(cm_beats.beat_count_dist_mode[0])
    t_deriv_expr_y = [0]*int(cm_beats.beat_count_dist_mode[0])
    x_deriv = np.zeros(int(len(elec_nan_removed[0])))
    y_deriv = np.zeros(int(len(elec_nan_removed[1])))
    vector_mag = np.zeros((int(cm_beats.beat_count_dist_mode[0]), 
        len(elec_nan_removed[0])))
    vector_x_comp = np.zeros((int(cm_beats.beat_count_dist_mode[0]), 
        len(elec_nan_removed[0])))
    vector_y_comp = np.zeros((int(cm_beats.beat_count_dist_mode[0]), 
        len(elec_nan_removed[0])))

    for num in range(int(cm_beats.beat_count_dist_mode[0])):
        a, b, c, d, e, f = conduction_vel.cv_popt[num]
        t_xy = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        
        t_deriv_expr_x[num] = sym.lambdify([x, y], t_xy.diff(x))
        t_deriv_expr_y[num] = sym.lambdify([x, y], t_xy.diff(y))

    for num in range(int(cm_beats.beat_count_dist_mode[0])):
        for electrode in range(len(elec_nan_removed[0])):
            # x_expr = sym.lambdify([x, y], t_deriv_expr_x[num], "numpy")
            # y_expr = sym.lambdify([x, y], t_deriv_expr_y[num], "numpy")
            # From Bayly et al, the equation for the x and y velocity components
            # of the conduction velocity, Tx and Ty, are:
            # Tx / (Tx^2 + Ty^2)
            # Ty / (Tx^2 + Ty^2)
            T_part_x = t_deriv_expr_x[num](elec_nan_removed[0][electrode], 
                elec_nan_removed[1][electrode])
            T_part_y = t_deriv_expr_y[num](elec_nan_removed[1][electrode], 
                elec_nan_removed[0][electrode])

            x_deriv[electrode] = T_part_x / (T_part_x**2 + T_part_y**2)
            y_deriv[electrode] = T_part_y / (T_part_x**2 + T_part_y**2)
            # x_deriv[electrode] = t_deriv_expr_x[num](elec_nan_removed[0][electrode], 
            #     elec_nan_removed[1][electrode])
            # y_deriv[electrode] = t_deriv_expr_y[num](elec_nan_removed[1][electrode], 
            #     elec_nan_removed[0][electrode])
        
        # print(np.sqrt(np.add(np.square(x_deriv), np.square(y_deriv))))
        vector_mag[num] = np.sqrt(np.add(np.square(x_deriv), np.square(y_deriv)))
        vector_x_comp[num] = x_deriv
        vector_y_comp[num] = y_deriv
        
    conduction_vel.vector_mag = pd.DataFrame(vector_mag)
    conduction_vel.vector_x_comp = pd.DataFrame(vector_x_comp)
    conduction_vel.vector_y_comp = pd.DataFrame(vector_y_comp)

    # for var in [x, y]:
    #     print("\\frac{\\partial g}{\\partial " + str(var) + "} =", 
    #         sym.latex(sym.simplify(t_xy.diff(var))))

    print()