# Author: Christopher S. Dunham
# 12/4/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import time
from numba import njit
import numpy as np
import pandas as pd


def calculate_conduction_velocity(elecGUI120, conduction_vel, local_act_time, heat_map, input_param, ElectrodeConfig):
    try:
        if hasattr(conduction_vel, 'param_dist_raw') is True:
            print("Clearing old CV data before running new calculation...")
            delattr(conduction_vel, 'param_dist_raw')
        start_time = time.process_time()
        conduction_vel.param_dist_raw = local_act_time.distance_from_min.divide(local_act_time.param_dist_normalized.loc[:, local_act_time.final_dist_beat_count])
        # Need to add a placeholder value for the minimum channel; currently gives NaN as a consequence of division by zero.
        # Want/need to display the origin for heat map purposes.  Question is how to do this efficiently.

        conduction_vel.param_dist_raw.index = ElectrodeConfig.electrode_names
        conduction_vel.param_dist_raw.insert(0, 'Electrode', ElectrodeConfig.electrode_names)
        conduction_vel.param_dist_raw.insert(1, 'X', ElectrodeConfig.electrode_coords_x)
        conduction_vel.param_dist_raw.insert(2, 'Y', ElectrodeConfig.electrode_coords_y)

        end_time = time.process_time()
        print("CV calculation complete.")
        print(end_time - start_time)
        # graph_conduction_vel(elecGUI120, heat_map, local_act_time, conduction_vel, input_param)
    except AttributeError:
        print("Please calculate local activation time first.")