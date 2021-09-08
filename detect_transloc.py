# Christopher S. Dunham
# 9/7/2021
# Gimzewski Laboratory
# University of California, Los Angeles
# Original work

# Module for calculating pacemaker translocations
# Uses an algorithm to determine pacemaker electrode (if multiple for a given
# beat), track changes in pacemaker electrode across beats and, if the change in
# electrode location exceeds some threshold distance, marks it as an event,
# recording the length (in beats) the electrode was in that location. Then the
# process continues with the new electrode.
from typing import List
import numpy as np
from pandas.core.frame import DataFrame

def pm_translocations(analysisGUI, pace_maker, electrode_config):
    electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')

    min_pm = pace_maker.param_dist_normalized.drop(
        columns=["Electrode", "X", "Y"]).min(axis=1)
    min_elecs_idx = np.where(min_pm == 0)[0]
    pm_elecs = list(
        map(electrode_config.electrode_names.__getitem__, min_elecs_idx))

    elecs_only = pace_maker.param_dist_normalized.loc[pm_elecs].drop(
        columns=pace_maker.param_dist_normalized.columns[3:])
    pm_only_all_beats = pace_maker.param_dist_normalized.loc[pm_elecs]

    # "Size" of event, or length in beats the pacemaker is at some electrode
    event_length = 0
    # Store event lengths in list for however many events there are.
    event_length_list = []
    # Threshold distance in micrometers (microns)
    thresh = 800
    # List to store pacemaker electrodes for checking against previous beat
    pm_electrode = []
    
    # Algorithm begins.
    """
    Pseudocode below:
    
    Identify PM electrode
        Check whether multiple PMs in given beat
            If yes:
                Calculate distance between
                If distance < threshold, pick one electrode and proceed
                If distance > threshold, terminate process
                *** The above step can use further refinement ***
            If no:
                Continue
    Monitor PM electrode through each beat
    Stop when PM electrode changes
        Compare to the new PM electrode
            Calculate distance between previous pacemaker and new pacemaker
            Does the distance exceed the threshold?
                If yes:
                    Mark this as an event
                    Record the number of beats previous electrode was THE pacemaker
                    Continue search for next event
                If no:
                    Consider electrode change meaningless
                    Continue search for an event
    """
    for num, beat in enumerate(pm_only_all_beats.columns[3:]):
        current_pm_idx = np.where(pm_only_all_beats[beat] == 0)
        num_pm = np.shape(current_pm_idx)[1]
        
        # Decide between pacemaker electrodes in the event there are multiple
        # for a given beat
        if num_pm > 1:
            temp_pm = []
            for idx in current_pm_idx[0]:
                temp_pm.append(pm_elecs[idx])

            # When multiple minima exist, confirm their distances are under 
            # the threshold.
            current_pm = distance_calc(
                pm_only_all_beats[["Electrode", "X", "Y", beat]], 
                temp_pm, thresh=thresh,
                calc_mode="multi_min")
            # print(f"Inside multi-if: {current_pm}")
            if current_pm == None:
                print("Terminating due to widely dispersed pacemakers in beat.\n")
                print("Consider re-evaluating your data.")
                break
            pm_electrode.append(current_pm)
            
            # Check whether the current pacemaker electrode is the same
            # electrode as in the previous beat
            if (num != 0):
                if (pm_electrode[num] != pm_electrode[num-1]):
                    old_elec = pm_electrode[num-1]
                    new_elec = pm_electrode[num]
                    # print(f"Electrode changed from {old_elec} to {new_elec}.")

                    pm_old_new = [old_elec, new_elec]

                    # Boolean check whether electrode distance exceeds
                    # threshold
                    check_thresh = distance_calc(
                        pm_only_all_beats[["Electrode", "X", "Y", beat]], 
                        pm_old_new, thresh=thresh,
                        calc_mode="new_min")

                    if check_thresh == True:
                        event_length_list.append(event_length)
                        event_length = 1
                    elif check_thresh == False:
                        event_length += 1
                else:
                    event_length += 1
            else:
                event_length += 1

        else:
            current_pm = pm_elecs[current_pm_idx[0][0]]
            # print(f"Inside else: {current_pm}")
            pm_electrode.append(current_pm)
            
            # Check whether the current pacemaker electrode is the same
            # electrode as in the previous beat
            if (num != 0):
                if (pm_electrode[num] != pm_electrode[num-1]):
                    old_elec = pm_electrode[num-1]
                    new_elec = pm_electrode[num]
                    # print(f"Electrode changed from {old_elec} to {new_elec}.")

                    pm_old_new = [old_elec, new_elec]

                    # Boolean check whether electrode distance exceeds
                    # threshold
                    check_thresh = distance_calc(
                        pm_only_all_beats[["Electrode", "X", "Y", beat]], 
                        pm_old_new, thresh=thresh,
                        calc_mode="new_min")

                    if check_thresh == True:
                        event_length_list.append(event_length)
                        event_length = 1
                    elif check_thresh == False:
                        event_length += 1
                else:
                    event_length += 1
            else:
                event_length += 1
    # Uncomment line below if you want to include the last count, which
    # is unlikely to accurately represent an event.
    # event_length_list.append(event_length)

    # Remove the first count from event list. Keeping it may introduce an 
    # artifact to the data, as we do not know when the translocation began
    # prior to recording.
    event_length_list.pop(0)

    # Store event list.
    pace_maker.transloc_events = event_length_list

    # Print event list to terminal.
    print("Event lengths:\n" + str(pace_maker.transloc_events))


def distance_calc(pacemaker_only_df: DataFrame, pacemaker_elec: List, 
thresh: int, calc_mode=""):

    if calc_mode == "multi_min":
        temp_dists = []
        calc_df = pacemaker_only_df[["X", "Y"]]
        for fixed_elec in pacemaker_elec:
            pm_fixed = calc_df.loc[fixed_elec]
        
            for elec in pacemaker_elec:
                x1 = calc_df.loc[elec, "X"]
                y1 = calc_df.loc[elec, "Y"]
                x2 = pm_fixed["X"]
                y2 = pm_fixed["Y"]
                dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                temp_dists.append(dist)
        
        if all(dist < thresh for dist in temp_dists):
            return pacemaker_elec[0]
        else:
            print("Problem: pacemakers for this beat are far apart.")
            return None

    if calc_mode == "new_min":
        temp_dists = []
        calc_df = pacemaker_only_df[["X", "Y"]]
        for fixed_elec in pacemaker_elec:
            pm_fixed = calc_df.loc[fixed_elec]
        
            for elec in pacemaker_elec:
                x1 = calc_df.loc[elec, "X"]
                y1 = calc_df.loc[elec, "Y"]
                x2 = pm_fixed["X"]
                y2 = pm_fixed["Y"]
                dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                temp_dists.append(dist)
        
        if any(dist > thresh for dist in temp_dists):
            return True
        else:
            return False