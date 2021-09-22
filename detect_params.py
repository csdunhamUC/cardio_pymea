# Author: 
# Christopher S. Dunham
# Contact Email: 
# csdunham@chem.ucla.edu, csdunham@protonmail.com
# Organization: 
# University of California, Los Angeles
# Department of Chemistry & Biochemistry
# Laboratory PI: 
# James K. Gimzewski
# This is an original work, unless otherwise noted in comments, by CSD.
# 9/14/2021

from typing import Dict, List
import numpy as np
from pandas.core.frame import DataFrame
import scipy as sp
import pandas as pd
import calculate_pacemaker
import determine_beats


def auto_tune_params(analysisGUI, pace_maker, input_param) -> Dict:
    # Ideal parameters should simultaneously minimize excluded electrodes and
    # minimize the maximum time lag.

    # Num. excluded electrodes
    excluded_elec = pace_maker.excluded_elec
    # Maximum time lag
    max_lag = pace_maker.param_dist_normalized_max

    guess_dist = 1000
    guess_height = 100
    guess_trunc = []
    guess_silenced = [] # analysisGUI.elecCombobox.currentData()

    prev_guess_dist = []
    prev_guess_height = []
    prev_guess_trunc = []
    prev_guess_silenced = []

    """
    input_param.min_peak_dist = float(analysisGUI.pkDistEdit.text())
    input_param.min_peak_height = float(analysisGUI.pkHeightEdit.text())
    input_param.parameter_prominence = float(analysisGUI.pkPromEdit.text())
    input_param.parameter_width = float(analysisGUI.pkWidthEdit.text())
    input_param.parameter_thresh = float(analysisGUI.pkThreshEdit.text())
    input_param.sample_frequency = float(analysisGUI.sampleFreqEdit.currentText())
    """

    # Simplest method:
    # Iterate through amplitude, distance values until global min for (excluded
    # electrodes, maximum time) lag achieved
    # Need to consider a reasonable step size for amplitude, distance
    # Likely step size for amplitude: 5 or 10 uV
    # Likely step size for distance: 25 or 50 @ 1,000Hz (250 or 500 @ 10,000Hz)
    # Guess should probably start low and work up.
    # Or start high and work down.
    # Also need to consider some number of iterations before/after minima to
    # determine whether true minima
    
    print()


def determine_amp():
    print()


def determine_distance():
    print()


def determine_trunc():
    print()


def determine_silenced():
    print()