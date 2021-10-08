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
import pandas as pd
from pandas.core.frame import DataFrame
from colorama import Fore
from colorama import Style
from colorama import init
from colorama import deinit
from math import ceil

# Comment out init() if using Python on Windows.
init()

def pm_translocations(analysisGUI, pace_maker, electrode_config, beat_amp_int):
    try:
        min_pm = pace_maker.param_dist_normalized.drop(
            columns=["Electrode", "X", "Y"]).min(axis=1)
        min_elecs_idx = np.where(min_pm == 0)[0]
        pm_elecs = list(
            map(electrode_config.electrode_names.__getitem__, min_elecs_idx))
        pm_only_all_beats = pace_maker.param_dist_normalized.loc[pm_elecs]

        # max_pm = pace_maker.param_dist_normalized.drop(
        #     columns=["Electrode", "X", "Y"]).idxmax(axis=0)
        # max_elecs = max_pm.values

        max_pm = pace_maker.param_dist_normalized.drop(
            columns=["Electrode", "X", "Y"]).max()
        # print(pace_maker.param_dist_normalized.loc[
        #     pace_maker.param_dist_normalized["Beat 25"] == max_pm[24], 
        #     ["Electrode", "X", "Y", "Beat 25"]])

        # print(pace_maker.param_dist_normalized.loc[
        #     ["D11", "C11", "B10", "C10"], "Beat 25"])

        # "Size" of event, or length in beats the pacemaker is at some electrode
        event_length = 0
        # Store event lengths in list for however many events there are.
        event_length_list = []
        # Threshold distance in micrometers (microns)
        thresh = 500 # More than 2 electrodes away for 200x30 scheme
        print(f"Using threshold: {thresh} microns.")
        # List to store pacemaker electrodes for checking against previous beat
        pm_electrode = []
        
        # Algorithm begins.
        """
        Pseudocode below:

        Identify PM electrode
            Check whether multiple PMs in given beat
                If yes:
                    Calculate distance between electrodes
                    If distance < threshold, pick one electrode and proceed
                    If distance > threshold, pick PM electrode furthest from max
                        pacemaker time lag value
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
                    pace_maker.param_dist_normalized.loc[
                        pace_maker.param_dist_normalized[beat] == max_pm[num], 
                        ["Electrode", "X", "Y", beat]],
                    pm_only_all_beats[["Electrode", "X", "Y", beat]], 
                    temp_pm, thresh=thresh,
                    calc_mode="multi_min")
                # print(f"Inside multi-if: {current_pm}")
                if current_pm == None:
                    print(f"Widely dispersed pacemakers in {beat}.")
                    print("Consider re-evaluating your data.")
                    # Stops event detection
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
                            pace_maker.param_dist_normalized.loc[
                                pace_maker.param_dist_normalized[beat] == max_pm[num], 
                                ["Electrode", "X", "Y", beat]],
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
                            pace_maker.param_dist_normalized.loc[
                                pace_maker.param_dist_normalized[beat] == max_pm[num], 
                                ["Electrode", "X", "Y", beat]],
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

        file_length = analysisGUI.file_length
        num_beats = len(pace_maker.final_dist_beat_count)
        beat_rate = num_beats / file_length

        # Store event list.
        pace_maker.transloc_events = event_length_list

        # Store number of beats.
        pace_maker.number_beats = num_beats

        # Print event list to terminal.
        print("Event lengths:\n" + str(event_length_list))
        # print("Normalized event lengths:\n" + str(norm_event_length))
        deinit()
    except IndexError:
        print("No events.")
        pace_maker.transloc_events = [None]
        pace_maker.number_beats = None


def distance_calc(max_df: DataFrame, pacemaker_only_df: DataFrame, 
pacemaker_elec: List, thresh: int, calc_mode=""):

    if calc_mode == "multi_min":
        temp_dists = []
        calc_df = pacemaker_only_df[["X", "Y"]]
        current_beat = pacemaker_only_df.columns[-1]
        # print(pacemaker_elec)
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
            # Compare each minimum to each maximum time lag for given beat
            # Calculate distance
            max_elecs = max_df.index
            temp_dists2 = {}
            calc_max_df = max_df[["X", "Y"]]

            for fixed_elec in pacemaker_elec:
                pm_fixed = calc_df.loc[fixed_elec]
        
                for elec in max_elecs:
                    x1 = calc_max_df.loc[elec, "X"]
                    y1 = calc_max_df.loc[elec, "Y"]
                    x2 = pm_fixed["X"]
                    y2 = pm_fixed["Y"]
                    dist = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                    temp_dists2[fixed_elec + "_" + elec] = dist
            pm_to_elec_max_dist = max(
                temp_dists2.items(), key = lambda k : k[1])
            sep = "_"
            min_pm = pm_to_elec_max_dist[0].split(sep, 1)[0]

            # print(f"Furthest PM-to-max time lag: {temp_dists2}")
            # print(pacemaker_only_df.loc[pacemaker_elec])
            print(f"{Fore.LIGHTRED_EX}ALERT: pacemakers in {current_beat}" +
                f" are far apart. Choosing {min_pm}.{Style.RESET_ALL}")
            return min_pm

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
