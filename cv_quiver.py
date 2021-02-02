# Author: Christopher S. Dunham
# 1/31/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import numpy as np
from numpy.lib.function_base import interp
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
import matplotlib.tri as tri
from scipy.interpolate import griddata


def cv_quiver_plot(analysisGUI, input_param, local_act_time, conduction_vel):
    # X and Y electrode coordinates
    conduction_vel.quiver_plot_axis.cla()

    z = conduction_vel.vector_x_comp[['X', 'Y', 
        local_act_time.final_dist_beat_count[
            input_param.cv_vect_beat_choice]]].dropna()
    x_arrow = z['X'].values
    y_arrow = z['Y'].values
    
    z_mag = conduction_vel.vector_mag[local_act_time.final_dist_beat_count[
            input_param.cv_vect_beat_choice]].dropna()

    input_param.cv_vect_beat_choice = int(
        analysisGUI.cv_vector_beat_select.get()) - 1
    
    uC = conduction_vel.vector_x_comp[local_act_time.final_dist_beat_count[
            input_param.cv_vect_beat_choice]].dropna().to_numpy()
    vC = conduction_vel.vector_y_comp[local_act_time.final_dist_beat_count[
            input_param.cv_vect_beat_choice]].dropna().to_numpy()
    
    xi = np.linspace(x_arrow.min(), x_arrow.max(), x_arrow.size)
    yi = np.linspace(y_arrow.min(), y_arrow.max(), y_arrow.size)

    x1, y1 = np.meshgrid(x_arrow, y_arrow)
    points = np.array((x_arrow, y_arrow)).T

    # Bicubic interpolation.
    uCi = interp2d(x_arrow, y_arrow, uC)(xi, yi)
    vCi = interp2d(x_arrow, y_arrow, vC)(xi, yi)
    grid_z_mag = interp2d(x_arrow, y_arrow, z_mag)(xi, yi)

    # uCi = griddata(points, uC, (x1, y1), method='linear')
    # vCi = griddata(points, vC, (x1, y1), method='linear')
    # grid_z_mag = griddata(points, z_mag, (x1, y1), method='linear')

    conduction_vel.quiver_plot_axis.streamplot(xi, yi, uCi, vCi)
    # conduction_vel.quiver_plot_axis.contour(x1, y1, grid_z_mag)
    # conduction_vel.quiver_plot_axis.contourf(x1, y1, grid_z_mag)

    # # Generate quiver plot using x, y arrow coordinates and magnitudes from
    # # calculate_cv function.
    # grid_z = griddata(points, z_mag, (x1, y1), method='nearest')
    # conduction_vel.quiver_plot_axis.contourf(x1, y1, grid_z, extend='both')
    # conduction_vel.quiver_plot_axis.contour(x1, y1, grid_z)

    # Generate quiver plot using x, y arrow coordinates and magnitudes from
    # calculate_cv function.
    conduction_vel.quiver_plot_axis.quiver(x_arrow, y_arrow, uC, vC)

    # Invert y-axis
    conduction_vel.quiver_plot_axis.invert_yaxis()

    # Draw plot.
    conduction_vel.quiver_plot.tight_layout()
    conduction_vel.quiver_plot.canvas.draw()