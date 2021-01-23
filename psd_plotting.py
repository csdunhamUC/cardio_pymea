# Author: Christopher S. Dunham
# 1/23/2021
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work

# PSD and log-log plotting functions for operation using mea-analysis program.

import numpy as np
import scipy as sp
from scipy import signal
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import time


def plot_log_vs_log(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param):
    print()


def plot_psd_welch(analysisGUI, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param):
    print()
    x = np.ndarray((2,3))
    freq, Pxx = signal.periodogram(x, fs=1.0, window='hann')
    # x must be a 1d array or series, fs = sampling frequency
    freq, Pxx = signal.welch(x, fs=1.0)
    # Welch may be the more robust method to use.