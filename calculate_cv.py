# Author: Christopher S. Dunham
# 1/29/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work
# New version utilizing fit to a 2D polynomial surface to calculate CV.
# Intent: eliminate artifacts caused by imposing an origin on CV.

import time
import numpy as np
import pandas as pd
# from scipy.optimize import curve_fit
import sympy as sym
from lmfit import Model
from matplotlib import pyplot as plt
import seaborn as sns


def calculate_conduction_velocity(analysisGUI, cm_beats, conduction_vel, 
local_act_time, heat_map, input_param, electrode_config):
    try:
        if hasattr(conduction_vel, 'param_dist_raw') is True:
            print("Clearing old CV data before running new calculation...")
            delattr(conduction_vel, 'param_dist_raw')
            delattr(conduction_vel, 'vector_mag')
            delattr(conduction_vel, 'vector_x_comp')
            delattr(conduction_vel, 'vector_y_comp')
            delattr(conduction_vel, 'cv_popt')

        start_time = time.process_time()
        print("Calculating CV per beat.")

        # Pre-allocate variables necessary for parameter optimization/curve fit.
        conduction_vel.cv_popt = [0]*int(cm_beats.beat_count_dist_mode[0])
        conduction_vel.cv_pcov = [0]*int(cm_beats.beat_count_dist_mode[0])
        # Find indices of electrodes with NaN values.
        nan_electrodes_idx = np.where(local_act_time.param_dist_raw[
            'Beat 1'].isna())[0]
        # Remove electrodes with NaN values for fitting modules (which cannot 
        # handle NaN values)
        x_elec = np.delete(
            electrode_config.electrode_coords_x, 
            nan_electrodes_idx)
        y_elec = np.delete(
            electrode_config.electrode_coords_y, 
            nan_electrodes_idx)
        # Generate 2xN, where N = number of non-NaN electrodes, of elec. coords.
        elec_nan_removed = np.array([x_elec, y_elec])

        # print(elec_nan_removed[0])
        # print(elec_nan_removed[1])
        
        # Generate new list with the electrode names with NaN values removed.
        elec_to_remove = [
            electrode_config.electrode_names[i] for i in nan_electrodes_idx]
        elec_removed_names = [
            i for i in electrode_config.electrode_names if i not in elec_to_remove]
        
        # Uses lmfit instead of curve_fit.  Results are same.
        twod_poly_model = Model(two_dim_polynomial, independent_vars=['x', 'y'],
            nan_policy='omit')
        print(twod_poly_model.param_names, twod_poly_model.independent_vars)
        # Calculate parameters a, b, c, d, e, f for two-dimensional polynomial
        # for each beat using lmfit's Model.fit()
        model_params = twod_poly_model.make_params(
            a=1, b=1, c=1, d=1, e=1, f=1)
        for num, beat in enumerate(local_act_time.param_dist_raw.drop(
        columns=['Electrode', 'X', 'Y'])):
            model_result = twod_poly_model.fit(
                local_act_time.param_dist_raw[beat].dropna(), 
                model_params, x = x_elec, y = y_elec)
            conduction_vel.cv_popt[num] = list(model_result.params.values())
            # print(model_result.fit_report())

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
        conduction_vel.vector_mag.insert(
            0, 
            'Electrode', 
            electrode_config.electrode_names)
        conduction_vel.vector_mag.insert(
            1, 
            'X', 
            electrode_config.electrode_coords_x)
        conduction_vel.vector_mag.insert(
            2, 
            'Y', 
            electrode_config.electrode_coords_y)
        conduction_vel.vector_x_comp.insert(
            0, 
            'Electrode', 
            electrode_config.electrode_names)
        conduction_vel.vector_x_comp.insert(
            1, 
            'X', 
            electrode_config.electrode_coords_x)
        conduction_vel.vector_x_comp.insert(
            2, 
            'Y', 
            electrode_config.electrode_coords_y)
        conduction_vel.vector_y_comp.insert(
            0, 
            'Electrode', 
            electrode_config.electrode_names)
        conduction_vel.vector_y_comp.insert(
            1, 
            'X', 
            electrode_config.electrode_coords_x)
        conduction_vel.vector_y_comp.insert(
            2, 
            'Y', 
            electrode_config.electrode_coords_y)

        # Calculate CV using simplistic finite difference methods.
        conduction_vel.param_dist_raw = local_act_time.distance_from_min.divide(
            local_act_time.param_dist_normalized.loc[
                :, local_act_time.final_dist_beat_count]).replace(
                    [np.inf, -np.inf], np.nan)

        conduction_vel.param_dist_raw_max = conduction_vel.param_dist_raw.max().max()
        conduction_vel.param_dist_raw_mean = np.nanmean(
            conduction_vel.param_dist_raw)

        # As before, insert electrode and coordinate columns into approp. fields.
        conduction_vel.param_dist_raw.index = electrode_config.electrode_names
        conduction_vel.param_dist_raw.insert(
            0, 
            'Electrode', 
            electrode_config.electrode_names)
        conduction_vel.param_dist_raw.insert(
            1, 
            'X', 
            electrode_config.electrode_coords_x)
        conduction_vel.param_dist_raw.insert(
            2, 
            'Y', 
            electrode_config.electrode_coords_y)

        end_time = time.process_time()
        print("CV calculation complete.")
        print(end_time - start_time)
    except AttributeError:
        print("Please calculate local activation time first.")
    except TypeError:
        print("Insufficient electrode count. Must have more than 6 electrodes.")
        return
    except KeyError:
        print("Insufficient Data.")


# Function for fitting if using lmfit.
def two_dim_polynomial(x, y, a, b, c, d, e, f):
    # Equation from: PV Bayly et al, IEEE, 1998, doi:10.1109/10.668746
    # t = T(x,y) = ax**2 + by**2 + cxy + dx + ey + f
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


# # Function for fitting if using curve_fit
# def two_dim_polynomial(elec_nan_removed, a, b, c, d, e, f):
#     # Equation from: PV Bayly et al, IEEE, 1998, doi:10.1109/10.668746
#     # t = T(x,y) = ax**2 + by**2 + cxy + dx + ey + f
#     x = elec_nan_removed[0]
#     y = elec_nan_removed[1]
#     return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


# Calculate derivatives w.r.t x and y for each beat.
def calc_deriv(elec_nan_removed, cm_beats, local_act_time, conduction_vel):
    # Designate x and y variables for differentiation
    x, y = sym.symbols('x, y', real=True)

    # Pre-allocate lists and arrays based on # of beats or # of electrodes.
    t_deriv_expr_x = [0]*int(cm_beats.beat_count_dist_mode[0])
    t_deriv_expr_y = [0]*int(cm_beats.beat_count_dist_mode[0])
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
        t_deriv_expr_x[num] = sym.lambdify([x, y], t_xy.diff(x), "numpy")
        t_deriv_expr_y[num] = sym.lambdify([x, y], t_xy.diff(y), "numpy")

    # Evaluate the partial derivatives of function T(x,y) w.r.t x and y
    # Partial derivatives for a beat are calculated at each electrode, using the
    # appropriate (x,y) coordinates of each electrode and are then stored in
    # x_deriv or y_deriv.  
    # These values are operated upon appropriately to find the vector magnitude 
    # of all electrodes (using vectorization via numpy) for each beat.
    for beat in range(int(cm_beats.beat_count_dist_mode[0])):
        # From Bayly et al, the equation for the x and y velocity components
        # of the conduction velocity, Tx and Ty, are:
        # Tx / (Tx^2 + Ty^2)
        # Ty / (Tx^2 + Ty^2)
        # Evaluate partial derivatives for x and y components
        T_part_x = t_deriv_expr_x[beat](elec_nan_removed[0], 
            elec_nan_removed[1])
        T_part_y = t_deriv_expr_y[beat](elec_nan_removed[0], 
            elec_nan_removed[1])
        # Complete calculation for x and y components.
        x_component = T_part_x / (T_part_x**2 + T_part_y**2)
        y_component = T_part_y / (T_part_x**2 + T_part_y**2)
        
        # Calculate vector magnitude for all electrodes in the given beat
        vector_mag[beat] = np.sqrt(
            np.square(x_component) + np.square(y_component))
        # Store vector components for all electrodes for each beat.
        vector_x_comp[beat] = x_component
        vector_y_comp[beat] = y_component

    # Store magnitude, components in Pandas dataframes in conduction_vel "struc"
    conduction_vel.vector_mag = pd.DataFrame(vector_mag)
    conduction_vel.vector_x_comp = pd.DataFrame(vector_x_comp)
    conduction_vel.vector_y_comp = pd.DataFrame(vector_y_comp)
    print()


def graph_conduction_vel(analysisGUI, heat_map, local_act_time, conduction_vel, 
input_param):
    try:
        if hasattr(heat_map, 'cv_solo_cbar') is True:
            heat_map.cv_solo_cbar.remove()
            delattr(heat_map, 'cv_solo_cbar')
        
        analysisGUI.cvWindow.paramPlot.axis1.cla()
        input_param.cv_solo_beat_choice = analysisGUI.cvWindow.paramSlider.value()

        electrode_names_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
            columns='X', values=local_act_time.final_dist_beat_count[
                input_param.cv_solo_beat_choice])

        cv_solo_temp = sns.heatmap(heatmap_pivot_table_4, cmap="jet", 
            annot=electrode_names_4, fmt="", 
            ax=analysisGUI.cvWindow.paramPlot.axis1, 
            cbar=False)
        mappable_4 = cv_solo_temp.get_children()[0]
        heat_map.cv_solo_cbar = analysisGUI.cvWindow.paramPlot.axis1.figure.colorbar(
            mappable_4, 
            ax=analysisGUI.cvWindow.paramPlot.axis1)
        heat_map.cv_solo_cbar.ax.set_title("μm/(ms)", fontsize=10)

        analysisGUI.cvWindow.paramPlot.axis1.set(
            title="Conduction Velocity, Beat " + 
            str(input_param.cv_solo_beat_choice+1), 
            xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")

        analysisGUI.cvWindow.paramPlot.fig.tight_layout()
        analysisGUI.cvWindow.paramPlot.draw()

    except AttributeError:
        print("Please make sure you've calculated Local Activation Time first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def cv_quiver_plot(analysisGUI, input_param, local_act_time, conduction_vel):
    try:
        input_param.cv_vector_beat_choice = analysisGUI.cvVectWindow.paramSlider.value() 
        analysisGUI.cvVectWindow.paramPlot.axis1.cla()
        curr_beat = local_act_time.final_dist_beat_count[
            input_param.cv_vector_beat_choice]

        cv_beat_mag = conduction_vel.vector_mag[
            ['X', 'Y', curr_beat]].dropna()
        cv_beat_raw = conduction_vel.param_dist_raw[
            ['X', 'Y', curr_beat]].dropna()
        lat_beat = local_act_time.param_dist_normalized[
            ['X', 'Y', curr_beat]].dropna()
        
        x_comp = conduction_vel.vector_x_comp[
            ['X', 'Y', curr_beat]].dropna()
        y_comp = conduction_vel.vector_y_comp[
            ['X', 'Y', curr_beat]].dropna()

        # For vector magnitude and plotting x, y coordinates in a grid
        contZ_mag = cv_beat_mag.pivot_table(index='Y', 
            columns='X', values=cv_beat_mag).values
        contZ_raw = cv_beat_raw.pivot_table(index='Y',
            columns='X', values=cv_beat_raw).values
        contZ_lat = lat_beat.pivot_table(index='Y', 
            columns='X', values=lat_beat).values
        contX_uniq = np.sort(cv_beat_mag.X.unique())
        contY_uniq = np.sort(cv_beat_mag.Y.unique())
        contX, contY = np.meshgrid(contX_uniq, contY_uniq)

        # For vector components.
        contU = x_comp.pivot_table(index='Y', columns='X', values=x_comp).values
        contV = y_comp.pivot_table(index='Y', columns='X', values=y_comp).values

        # Plot contour plots.  Change contZ_mag to contZ_raw for other contour plot.
        analysisGUI.cvVectWindow.paramPlot.axis1.contour(contX, contY, contZ_mag,
            cmap='jet')
        contf = analysisGUI.cvVectWindow.paramPlot.axis1.contourf(contX, contY, 
            contZ_mag, cmap='jet')
        # Plot streamplot.
        analysisGUI.cvVectWindow.paramPlot.axis1.streamplot(contX, contY, contU, 
            contV)
        # Plot quiver plot.
        analysisGUI.cvVectWindow.paramPlot.axis1.quiver(contX, contY, contU, 
            contV, angles='xy')
        analysisGUI.cvVectWindow.paramPlot.axis1.set(xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)", title="Quiver, Stream, Contour of CV. " + 
                str(curr_beat))

        # Add colorbar.
        cbar = plt.colorbar(contf, ax=analysisGUI.cvVectWindow.paramPlot.axis1)
        cbar.ax.set_ylabel('Conduction Velocity (μm/(ms))')

        # Invert y-axis
        analysisGUI.cvVectWindow.paramPlot.axis1.invert_yaxis()

        # Draw plot.
        analysisGUI.cvVectWindow.paramPlot.fig.tight_layout()
        analysisGUI.cvVectWindow.paramPlot.draw()

        cbar.remove()
    except AttributeError:
        print("Please calculate LAT and CV first.")


# Previous loop structure for evaluating SymPy functions.  Since switched to 
# numpy vectorized format for performance increase.
#     x_comp = np.zeros(int(len(elec_nan_removed[0])))
#     y_comp = np.zeros(int(len(elec_nan_removed[1])))
# for beat in range(int(cm_beats.beat_count_dist_mode[0])):
#     for electrode in range(len(elec_nan_removed[0])):
#         # From Bayly et al, the equation for the x and y velocity components
#         # of the conduction velocity, Tx and Ty, are:
#         # Tx / (Tx^2 + Ty^2)
#         # Ty / (Tx^2 + Ty^2)
#         # Evaluate partial derivatives for x and y components
#         T_part_x = t_deriv_expr_x[beat](elec_nan_removed[0][electrode], 
#             elec_nan_removed[1][electrode])
#         T_part_y = t_deriv_expr_y[beat](elec_nan_removed[0][electrode], 
#             elec_nan_removed[1][electrode])

#         # Complete calculation for x and y components.
#         x_comp[electrode] = T_part_x / (T_part_x**2 + T_part_y**2)
#         y_comp[electrode] = T_part_y / (T_part_x**2 + T_part_y**2)
    
#     # Calculate vector magnitude for all electrodes in the given beat
#     vector_mag[beat] = np.sqrt(np.square(x_comp) + np.square(y_comp))
#     # Store vector components for all electrodes for each beat.
#     vector_x_comp[beat] = x_comp
#     vector_y_comp[beat] = y_comp

# for var in [x, y]:
#     print("\\frac{\\partial g}{\\partial " + str(var) + "} =", 
#         sym.latex(sym.simplify(t_xy.diff(var))))

# # Calculate parameters a, b, c, d, e, f for two-dimensional polynomial
# # for each beat using curve_fit
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
