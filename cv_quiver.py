# Author: Christopher S. Dunham
# 1/31/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.tri as tri


def cv_quiver_plot(analysisGUI, input_param, local_act_time, conduction_vel):
    try:
        input_param.cv_vector_beat_choice = analysisGUI.cvVectWindow.paramSlider.value()
        
        # X and Y electrode coordinates
        analysisGUI.cvVectWindow.paramPlot.axes.cla()

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

        # For vector mag and plotting x, y coordinates in a grid
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
        analysisGUI.cvVectWindow.paramPlot.axes.contour(contX, contY, contZ_lat,
            cmap='jet')
        contf = analysisGUI.cvVectWindow.paramPlot.axes.contourf(contX, contY, 
            contZ_lat, cmap='jet')
        # Plot streamplot.
        analysisGUI.cvVectWindow.paramPlot.axes.streamplot(contX, contY, contU, 
            contV)
        # Plot quiver plot.
        analysisGUI.cvVectWindow.paramPlot.axes.quiver(contX, contY, contU, 
            contV, angles='xy')
        analysisGUI.cvVectWindow.paramPlot.axes.set(xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)", title="Quiver, Stream, Contour of CV. " + 
                str(curr_beat))

        # Add colorbar.
        cbar = plt.colorbar(contf, ax=analysisGUI.cvVectWindow.paramPlot.axes)
        cbar.ax.set_ylabel('Conduction Velocity (μm/(ms))')

        # Invert y-axis
        analysisGUI.cvVectWindow.paramPlot.axes.invert_yaxis()

        # Draw plot.
        analysisGUI.cvVectWindow.paramPlot.fig.tight_layout()
        analysisGUI.cvVectWindow.paramPlot.draw()

        cbar.remove()
    except AttributeError:
        print("Please calculate LAT and CV first.")
