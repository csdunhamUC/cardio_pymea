from typing import List
import numpy as np
from pandas.core.frame import DataFrame

def pm_translocations(analysisGUI, pace_maker, electrode_config):
    electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')
    # print(electrode_names)

    min_pm = pace_maker.param_dist_normalized.drop(
        columns=["Electrode", "X", "Y"]).min(axis=1)
    min_elecs_idx = np.where(min_pm == 0)[0]
    pm_elecs = list(
        map(electrode_config.electrode_names.__getitem__, min_elecs_idx))

    elecs_only = pace_maker.param_dist_normalized.loc[pm_elecs].drop(
        columns=pace_maker.param_dist_normalized.columns[3:])
    pm_only_all_beats = pace_maker.param_dist_normalized.loc[pm_elecs]

    print(pm_only_all_beats)

    # "Size" of event, or length in beats the pacemaker is at some electrode
    event_length = 0
    # Threshold distance in micrometers (microns)
    thresh = 800
    # Algorithm
    """
    What needs to be done for this algorithm to be successful:

    Locate PM electrode.
    Monitor PM electrode through each beat
    Stop when PM electrode is no longer the PM electrode
        Check the other electrodes
            Find distance between previous pacemaker and new pacemaker
            Does the distance exceed the threshold?
            If yes:
                Mark this as an event
                Record the number of beats previous pacemaker was THE pacemaker
                Select new PM
                Continue (or change) search with the new PM
            If no:
                Continue search for an event
    
    """
    for beat in pm_only_all_beats.columns[3:13]:
        current_pm_idx = np.where(pm_only_all_beats[beat] == 0)
        num_pm = np.shape(current_pm_idx)[1]
        
        # current_pm = []
        # Decide between pacemaker electrodes in the event there are multiple
        # for a given beat
        if num_pm > 1:
            temp_pm = []
            for idx in current_pm_idx[0]:
                temp_pm.append(pm_elecs[idx])

            current_pm = distance_calc(
                pm_only_all_beats[["Electrode", "X", "Y", beat]], 
                temp_pm, thresh=thresh,
                calc_mode="multi_min")
            print(current_pm)

        else:
            pm_idx = pm_elecs[current_pm_idx[0][0]]
            print(pm_idx)
            # current_pm = current_pm_idx[0][num]
            # print(current_pm)
            # print(current_pm_idx[0][0])
            # print(current_pm_idx[0][1])
        # When multiple minima exist, confirm their distances are under thresh
        # current_pm = pm_only_all_beats.loc[pm_only_all_beats[beat] == 0]
        # print(current_pm)


def distance_calc(pacemaker_only_df: DataFrame, pacemaker_elec: List, 
thresh: int, calc_mode=""):
    distances = []

    if calc_mode == "multi_min":
        temp_dists = []
        calc_df = pacemaker_only_df[["X", "Y"]]
        for fixed_elec in pacemaker_elec:
            pm_fixed = calc_df.loc[fixed_elec]
            # print(pm_fixed)
        
            for elec in pacemaker_elec:
                x1 = calc_df.loc[elec, "X"]
                y1 = calc_df.loc[elec, "Y"]
                x2 = pm_fixed["X"]
                y2 = pm_fixed["Y"]
                dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                temp_dists.append(dist)
        # print(temp_dists)
        
        if all(dist < thresh for dist in temp_dists):
            return pacemaker_elec[0]
        else:
            print("Problem: pacemakers for given beat are far apart.")
            print("Terminating. Consider re-evaluating your data.")


    if calc_mode == "full":
        elecs = pacemaker_only_df.index
        print()
    
    # for elec_fixed in elecs:
    #     pm_fixed_loc = elecs[elec_fixed]
    #     temp_dists = []

    #     for elec in elecs:
    #         x1 = elecs[elec][0]
    #         x2 = pm_fixed_loc[0]
    #         y1 = elecs[elec][1]
    #         y2 = pm_fixed_loc[1]

    #         dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    #         temp_dists.append(dist)
    
    # distances.append(temp_dists)

    return distances