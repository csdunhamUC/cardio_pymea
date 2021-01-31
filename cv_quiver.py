# Author: Christopher S. Dunham
# 1/31/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
# import seaborn as sns
from matplotlib import pyplot as plt

def cv_quiver_plot(analysisGUI, input_param, local_act_time, conduction_vel):
    x_arrow = conduction_vel.vector_x_comp['X'].values
    y_arrow = conduction_vel.vector_y_comp['Y'].values
    conduction_vel.quiver_plot_axis.quiver(x_arrow, y_arrow,
        conduction_vel.vector_x_comp[local_act_time.final_dist_beat_count[
            input_param.cv_solo_beat_choice]], 
        conduction_vel.vector_y_comp[local_act_time.final_dist_beat_count[
            input_param.cv_solo_beat_choice]])