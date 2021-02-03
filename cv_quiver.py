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
from scipy.interpolate import bisplev, bisplrep


def cv_quiver_plot(analysisGUI, input_param, local_act_time, conduction_vel):
    input_param.cv_vector_beat_choice = int(
        analysisGUI.cv_vector_beat_select.get()) - 1
    
    # X and Y electrode coordinates
    conduction_vel.quiver_plot_axis.cla()

    cv_beat = conduction_vel.vector_mag[
        ['X', 'Y', local_act_time.final_dist_beat_count
            [input_param.cv_vector_beat_choice]]].dropna()
    
    x_comp = conduction_vel.vector_x_comp[
        ['X', 'Y', local_act_time.final_dist_beat_count
            [input_param.cv_vector_beat_choice]]].dropna()
    
    y_comp = conduction_vel.vector_y_comp[
        ['X', 'Y', local_act_time.final_dist_beat_count
            [input_param.cv_vector_beat_choice]]].dropna()

    # For vector mag and plotting x, y coordinates in a grid
    contZ = cv_beat.pivot_table(index='X', columns='Y', values=cv_beat).T.values
    contX_uniq = np.sort(cv_beat.X.unique())
    contY_uniq = np.sort(cv_beat.Y.unique())
    contX, contY = np.meshgrid(contX_uniq, contY_uniq)
    
    # For vector components.
    contU = x_comp.pivot_table(index='X', columns='Y', values=x_comp).T.values
    contV = y_comp.pivot_table(index='X', columns='Y', values=y_comp).T.values

    # Plot contour plots
    conduction_vel.quiver_plot_axis.contour(contX, contY, contZ,
        cmap='jet')
    contf = conduction_vel.quiver_plot_axis.contourf(contX, contY, contZ, 
        cmap='jet')
    # Plot streamplot.
    conduction_vel.quiver_plot_axis.streamplot(contX, contY, contU, contV)
    # Plot quiver plot.
    conduction_vel.quiver_plot_axis.quiver(contX, contY, contU, contV)

    # Add colorbar.
    cbar = plt.colorbar(contf, ax=conduction_vel.quiver_plot_axis)
    cbar.ax.set_ylabel('Conduction Velocity (μm/(ms))')

    # Invert y-axis
    conduction_vel.quiver_plot_axis.invert_yaxis()

    # Draw plot.
    conduction_vel.quiver_plot.tight_layout()
    conduction_vel.quiver_plot.canvas.draw()

    cbar.remove()
