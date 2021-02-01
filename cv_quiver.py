# Author: Christopher S. Dunham
# 1/31/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def cv_quiver_plot(analysisGUI, input_param, local_act_time, conduction_vel):
    # X and Y electrode coordinates
    conduction_vel.quiver_plot_axis.cla()

    x_arrow = conduction_vel.vector_x_comp['X'].values
    y_arrow = conduction_vel.vector_y_comp['Y'].values

    input_param.cv_vect_beat_choice = int(
        analysisGUI.cv_vector_beat_select.get()) - 1

    # Generate quiver plot using x, y arrow coordinates and magnitudes from
    # calculate_cv function.
    conduction_vel.quiver_plot_axis.quiver(x_arrow, y_arrow,
        conduction_vel.vector_x_comp[local_act_time.final_dist_beat_count[
            input_param.cv_vect_beat_choice]], 
        conduction_vel.vector_y_comp[local_act_time.final_dist_beat_count[
            input_param.cv_vect_beat_choice]])
    
    # Invert y-axis
    conduction_vel.quiver_plot_axis.set_ylim(
        conduction_vel.quiver_plot_axis.get_ylim()[::-1])

    # Draw plot.
    conduction_vel.quiver_plot.tight_layout()
    conduction_vel.quiver_plot.canvas.draw()