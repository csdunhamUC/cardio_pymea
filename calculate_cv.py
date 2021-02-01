# Author: Christopher S. Dunham
# 1/29/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work
# New version utilizing fit to a 2D polynomial surface to calculate CV.
# Intent: eliminate artifacts caused by imposing an origin on CV.

import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import sympy as sym
from lmfit import Model


def calculate_conduction_velocity(analysisGUI, cm_beats, conduction_vel, local_act_time, heat_map, input_param, electrode_config):
    # try:
        if hasattr(conduction_vel, 'param_dist_raw') is True:
            print("Clearing old CV data before running new calculation...")
            delattr(conduction_vel, 'param_dist_raw')
            delattr(conduction_vel, 'vector_mag')
            delattr(conduction_vel, 'vector_x_comp')
            delattr(conduction_vel, 'vector_y_comp')
            delattr(conduction_vel, 'cv_popt')

        start_time = time.process_time()
        print("Calculating CV per beat.")
        # electrode_config.electrode_coords_x
        # electrode_config.electrode_coords_y
        # Y data to fit to: local_act_time.param_dist_normalized
        # "X" data consists of x, y coordinates from electrode_coords

        conduction_vel.cv_popt = [0]*int(cm_beats.beat_count_dist_mode[0])
        conduction_vel.cv_pcov = [0]*int(cm_beats.beat_count_dist_mode[0])
        nan_electrodes_idx = np.where(local_act_time.param_dist_normalized[
            'Beat 1'].isna())[0]
        x_elec = np.delete(electrode_config.electrode_coords_x, nan_electrodes_idx)
        y_elec = np.delete(electrode_config.electrode_coords_y, nan_electrodes_idx)
        elec_nan_removed = np.array([x_elec, y_elec])

        print(elec_nan_removed)
        
        # Generate new list with the electrode names with NaN values removed.
        elec_to_remove = [electrode_config.electrode_names[i] for i in nan_electrodes_idx]
        elec_removed_names = [
            i for i in electrode_config.electrode_names if i not in elec_to_remove]
        
        # # Alternative: using lmfit instead of curve_fit.  Results are same.
        # twod_poly_model = Model(two_dim_polynomial, independent_vars=['x', 'y'],
        #     nan_policy='omit')
        # print(twod_poly_model.param_names, twod_poly_model.independent_vars)
        # # Calculate parameters a, b, c, d, e, f for two-dimensional polynomial
        # # for each beat.
        # model_params = twod_poly_model.make_params(
        #     a=1, b=1, c=1, d=1, e=1, f=1)
        # for num, beat in enumerate(local_act_time.param_dist_normalized.drop(
        # columns=['Electrode', 'X', 'Y'])):
        #     model_result = twod_poly_model.fit(
        #         local_act_time.param_dist_normalized[beat].dropna(), 
        #         model_params, x = x_elec, y = y_elec)
        #     conduction_vel.cv_popt[num] = list(model_result.params.values())
        #     print(model_result.fit_report())

        for num, beat in enumerate(local_act_time.param_dist_normalized.drop(
        columns=['Electrode', 'X', 'Y'])):    
            conduction_vel.cv_popt[num], conduction_vel.cv_pcov[num] = curve_fit(
                two_dim_polynomial, elec_nan_removed, 
                local_act_time.param_dist_normalized[beat].dropna(),
                method="trf")
        
        # # Calculate parameters a, b, c, d, e, f for two-dimensional polynomial
        # # for each beat.
        # for num, beat in enumerate(local_act_time.param_dist_normalized.drop(
        # columns=['Electrode', 'X', 'Y'])):
        #     conduction_vel.cv_popt[num], conduction_vel.cv_pcov[num] = curve_fit(
        #         two_dim_polynomial, elec_nan_removed, 
        #         local_act_time.param_dist_normalized[beat].dropna(),
        #         method="trf")

        # # Alternative to the preceding lines, using sorted values
        # # Gives truly nonsensical results.  Probably needs better sorting.
        # cv_without_nan = conduction_vel.param_dist_raw[beat].dropna()
        # cv_without_nan = cv_without_nan.sort_values(ascending=True)
        # x_sorted = local_act_time.distance_from_min.loc[cv_without_nan.index, 
        #     beat].sort_values(ascending=True)
        # elec_removed_sorted = np.sort(elec_nan_removed)
        # for num, beat in enumerate(local_act_time.param_dist_normalized.drop(
        # columns=['Electrode', 'X', 'Y'])):
        #     lat_sorted = local_act_time.param_dist_normalized[
        #         beat].dropna().sort_values(ascending=True)
        #     conduction_vel.cv_popt[num], conduction_vel.cv_pcov[num] = curve_fit(
        #         two_dim_polynomial, elec_removed_sorted, lat_sorted)

        # print(conduction_vel.cv_popt[0])
        # print(conduction_vel.cv_popt[3])

        # Calculate derivatives using SymPy diff and lambdify at each electrode
        # for each beat, using the parameters for the two-dimensional polynomial
        # obtained from curve_fit
        calc_deriv(elec_nan_removed, cm_beats, local_act_time, conduction_vel)

        # Assign column and index values for each calculated vector parameter.
        conduction_vel.vector_mag.columns = elec_removed_names
        conduction_vel.vector_mag.index = local_act_time.final_dist_beat_count
        conduction_vel.vector_x_comp.columns = elec_removed_names
        conduction_vel.vector_x_comp.index = local_act_time.final_dist_beat_count
        conduction_vel.vector_y_comp.columns = elec_removed_names
        conduction_vel.vector_y_comp.index = local_act_time.final_dist_beat_count

        # Fill in the void of omitted electrodes with NaN values for heatmaps.
        missing_elec_fill = [np.nan] * int(cm_beats.beat_count_dist_mode[0])
        for missing in nan_electrodes_idx:
            nan_elec = electrode_config.electrode_names[missing]
            conduction_vel.vector_mag.insert(int(missing), nan_elec, 
                missing_elec_fill)
            conduction_vel.vector_x_comp.insert(int(missing), nan_elec, 
                missing_elec_fill)
            conduction_vel.vector_y_comp.insert(int(missing), nan_elec, 
                missing_elec_fill)

        # Obtain the transpose of each vector dataframe, such that the rows are
        # electrodes and columns are beats.  Insert electrode and coordinate
        # columns.
        conduction_vel.vector_mag = conduction_vel.vector_mag.T
        conduction_vel.vector_x_comp = conduction_vel.vector_x_comp.T
        conduction_vel.vector_y_comp = conduction_vel.vector_y_comp.T
        conduction_vel.vector_mag.insert(0, 'Electrode', electrode_config.electrode_names)
        conduction_vel.vector_mag.insert(1, 'X', electrode_config.electrode_coords_x)
        conduction_vel.vector_mag.insert(2, 'Y', electrode_config.electrode_coords_y)
        conduction_vel.vector_x_comp.insert(0, 'Electrode', electrode_config.electrode_names)
        conduction_vel.vector_x_comp.insert(1, 'X', electrode_config.electrode_coords_x)
        conduction_vel.vector_x_comp.insert(2, 'Y', electrode_config.electrode_coords_y)
        conduction_vel.vector_y_comp.insert(0, 'Electrode', electrode_config.electrode_names)
        conduction_vel.vector_y_comp.insert(1, 'X', electrode_config.electrode_coords_x)
        conduction_vel.vector_y_comp.insert(2, 'Y', electrode_config.electrode_coords_y)

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
    # except AttributeError:
    #     print("Please calculate local activation time first.")


# # Function for fitting if using lmfit.
# def two_dim_polynomial(x, y, a, b, c, d, e, f):
#     # Equation from: PV Bayly et al, IEEE, 1998, doi:10.1109/10.668746
#     # t = T(x,y) = ax**2 + by**2 + cxy + dx + ey + f
#     return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


def two_dim_polynomial(elec_nan_removed, a, b, c, d, e, f):
    # Equation from: PV Bayly et al, IEEE, 1998, doi:10.1109/10.668746
    # t = T(x,y) = ax**2 + by**2 + cxy + dx + ey + f
    x = elec_nan_removed[0]
    y = elec_nan_removed[1]
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


# Calculate derivatives w.r.t x and y for each beat.
def calc_deriv(elec_nan_removed, cm_beats, local_act_time, conduction_vel):
    # Designate x and y variables for differentiation
    x, y = sym.symbols('x, y', real=True)

    # Pre-allocate lists and arrays based on # of beats or # of electrodes.
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

    # Calculate the first symbolic derivative and turn into a usable Python
    # function for each beat's set of parameters (a, b, c, d, e, f)
    for num in range(int(cm_beats.beat_count_dist_mode[0])):
        a, b, c, d, e, f = conduction_vel.cv_popt[num]
        t_xy = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

        t_deriv_expr_x[num] = sym.lambdify([x, y], t_xy.diff(x))
        t_deriv_expr_y[num] = sym.lambdify([x, y], t_xy.diff(y))

    # Evaluate the partial derivatives of function T(x,y) w.r.t x and y
    # Partial derivatives for a beat are calculated at each electrode, using the
    # appropriate (x,y) coordinates of each electrode and are then stored in
    # x_deriv or y_deriv.  
    # These values are operated upon appropriately to find the vector magnitude 
    # of all electrodes for each beat.
    for num in range(int(cm_beats.beat_count_dist_mode[0])):
        for electrode in range(len(elec_nan_removed[0])):
            # From Bayly et al, the equation for the x and y velocity components
            # of the conduction velocity, Tx and Ty, are:
            # Tx / (Tx^2 + Ty^2)
            # Ty / (Tx^2 + Ty^2)
            T_part_x = t_deriv_expr_x[num](elec_nan_removed[0][electrode], 
                elec_nan_removed[1][electrode])
            T_part_y = t_deriv_expr_y[num](elec_nan_removed[0][electrode], 
                elec_nan_removed[1][electrode])

            x_deriv[electrode] = T_part_x / (T_part_x**2 + T_part_y**2)
            y_deriv[electrode] = T_part_y / (T_part_x**2 + T_part_y**2)
        
        # Calculate vector magnitude for all electrodes in the given beat
        vector_mag[num] = np.sqrt(np.square(x_deriv) + np.square(y_deriv))
        # Store vector components for all electrodes for each beat.
        vector_x_comp[num] = x_deriv
        vector_y_comp[num] = y_deriv

    # Store magnitude, components in Pandas dataframes in conduction_vel "struc"
    conduction_vel.vector_mag = pd.DataFrame(vector_mag)
    conduction_vel.vector_x_comp = pd.DataFrame(vector_x_comp)
    conduction_vel.vector_y_comp = pd.DataFrame(vector_y_comp)

    # for var in [x, y]:
    #     print("\\frac{\\partial g}{\\partial " + str(var) + "} =", 
    #         sym.latex(sym.simplify(t_xy.diff(var))))

    print()