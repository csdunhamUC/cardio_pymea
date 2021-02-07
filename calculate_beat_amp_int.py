# Author: Christopher S. Dunham
# 02/06/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit


def calculate_beat_amp(analysisGUI, cm_beats, pace_maker, heat_map, input_param, 
electrode_config):
    print()
    # Obtain beat amplitudes using indices from pace_maker.raw data, store in
    # variable beat_amp_int


def calculate_beat_interval():
    print()
    # Using pace_maker.raw data, calculate the time between each beat.
    # Calculation needs to take into account input_param.sample_frequency


def beat_amp_interval_graph():
    print()
    # Heatmap for beat amplitude across all electrodes, per beat.
    # Statistical plot for beat amplitude vs distance, per beat.