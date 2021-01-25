# Author: Christopher Stuart Dunham (CSD)
# Emails: csdunham@chem.ucla.edu; azarhn@hotmail.com
# Github: https://github.com/Sapphyric/Python_Learning
# Organization: University of California, Los Angeles, Department of Chemistry & 
# Biochemistry
# Laboratory PI: James K. Gimzewski
# This is an original work, unless otherwise noted in comments, by CSD.
# Technical start date: 7/22/2020
# Effective start date: 9/10/2020
# Designed to run on Python 3.6 or newer.  Programmed under Python 3.8.
# Biggest known issues for Python versions earlier than 3.6:
# 1) Use of dictionary to contain electrode coordinates (ordered vs unordered)
# Consider using an OrderedDict instead if running under earlier versions of 
# Python.
# 2) If Python 2, tkinter vs Tkinter for GUI.
# Program is configured to work with both MEA120 and MEA60 systems from 
# Multichannel Systems for dimensions 200x30um (spacing and electrode width) 
# only.

import numpy as np
import importlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
# import dask.dataframe as dd
import pandasgui as pgui
import seaborn as sns
import os
import tkinter as tk
from tkinter import ttk
import importlib
from scipy import stats
# from dis import dis
import datetime
import determine_beats
from calculate_pacemaker import calculate_pacemaker
import calculate_upstroke_vel
import calculate_lat
from calculate_conduction_velocity import calculate_conduction_velocity
import param_vs_distance_stats
import psd_plotting


################################################################################
# Classes that serve similar to Matlab structures (C "struct") to house data and 
# allow it to be passed from one function to another.  Classes are generated for 
# ImportedData (where the raw data will go), PaceMakerData (where PM data will 
# go), UpstrokeVelData (where dV/dt data will go), LocalATData (where LAT data 
# will go), and CondVelData, where CV data will go.
class ImportedData:
    pass


class InputParameters:
    pass


class BeatAmplitudes:
    pass


class PacemakerData:
    pass


class UpstrokeVelData:
    pass


class LocalATData:
    pass


class CondVelData:
    pass


class MEAHeatMaps:
    pass


class StatisticsData:
    pass


class PSDData:
    pass


# Class containing electrode names and corresponding coordinates in x,y form, 
# units of micrometers (microns, um)
class ElectrodeConfig:
    # Electrode names and coordinates, using the system defined by CSD where 
    # origin (0,0) is at upper left corner of MEA.  Configured for 200x30um 
    # inter-electrode spacing and electrode diameter, respectively.

    def __init__(self, raw_data):
        self.mea_120_coordinates = {
            'F7': [1150, 1380], 'F8': [1150, 1610], 'F12': [1150, 2530], 
            'F11': [1150, 2300], 'F10': [1150, 2070], 'F9': [1150, 1840], 
            'E12': [920, 2530], 'E11': [920, 2300], 'E10': [920, 2070], 
            'E9': [920, 1840], 'D12': [690, 2530], 'D11': [690, 2300], 
            'D10': [690, 2070], 'D9': [690, 1840], 'C11': [460, 2300],
            'C10': [460, 2070], 'B10': [230, 2070], 'E8': [920, 1610], 
            'C9': [460, 1840], 'B9': [230, 1840], 'A9': [0, 1840], 
            'D8': [690, 1610], 'C8': [460, 1610], 'B8': [230, 1610], 
            'A8': [0, 1610], 'D7': [690, 1380], 'C7': [460, 1380], 
            'B7': [230, 1380], 'A7': [0, 1380], 'E7': [920, 1380], 
            'F6': [1150, 1150], 'E6': [920, 1150], 'A6': [0, 1150], 
            'B6': [230, 1150], 'C6': [460, 1150], 'D6': [690, 1150], 
            'A5': [0, 920], 'B5': [230, 920], 'C5': [460, 920], 
            'D5': [690, 920], 'A4': [0, 690], 'B4': [230, 690], 
            'C4': [460, 690], 'D4': [690, 690], 'B3': [230, 460],
            'C3': [460, 460], 'C2': [460, 230], 'E5': [920, 920], 
            'D3': [690, 460], 'D2': [690, 230], 'D1': [690, 0], 
            'E4': [920, 690], 'E3': [920, 460], 'E2': [920, 230], 
            'E1': [920, 0], 'F4': [1150, 690], 'F3': [1150, 460], 
            'F2': [1150, 230], 'F1': [1150, 0], 'F5': [1150, 920], 
            'G6': [1380, 1150], 'G5': [1380, 920], 'G1': [1380, 0], 
            'G2': [1380, 230], 'G3': [1380, 460], 'G4': [1380, 690], 
            'H1': [1610, 0], 'H2': [1610, 230], 'H3': [1610, 460], 
            'H4': [1610, 690], 'J1': [1840, 0], 'J2': [1840, 230], 
            'J3': [1840, 460], 'J4': [1840, 690], 'K2': [2070, 230], 
            'K3': [2070, 460], 'L3': [2300, 460], 'H5': [1610, 920], 
            'K4': [2070, 690], 'L4': [2300, 690], 'M4': [2530, 690], 
            'J5': [1840, 920], 'K5': [2070, 920], 'L5': [2300, 920], 
            'M5': [2530, 920], 'J6': [1840, 1150], 'K6': [2070, 1150], 
            'L6': [2300, 1150], 'M6': [2530, 1150], 'H6': [1610, 1150],
            'G7': [1380, 1380], 'H7': [1610, 1380], 'M7': [2530, 1380], 
            'L7': [2300, 1380], 'K7': [2070, 1380], 'J7': [1840, 1380], 
            'M8': [2530, 1610], 'L8': [2300, 1610], 'K8': [2070, 1610], 
            'J8': [1840, 1610], 'M9': [2530, 1840], 'L9': [2300, 1840], 
            'K9': [2070, 1840], 'J9': [1840, 1840], 'L10': [2300, 2070],
            'K10': [2070, 2070], 'K11': [2070, 2300], 'H8': [1610, 1610], 
            'J10': [1840, 2070], 'J11': [1840, 2300], 'J12': [1840, 2530], 
            'H9': [1610, 1840], 'H10': [1610, 2070], 'H11': [1610, 2300], 
            'H12': [1610, 2530], 'G9': [1380, 1840], 'G10': [1380, 2070], 
            'G11': [1380, 2300], 'G12': [1380, 2530], 'G8': [1380, 1610]}
        
        self.mea_60_coordinates = {
            '47A': [690, 1380], '48A': [690, 1610], '46A': [690, 1150], 
            '45A': [690, 920], '38A': [460, 1610], '37A': [460, 1380], 
            '28A': [230, 1610], '36A': [460, 1150], '27A': [230, 1380], 
            '17A': [0, 1380], '26A': [230, 1150], '16A': [0, 1150], 
            '35A': [460, 920], '25A': [230, 920], '15A': [0, 920], 
            '14A': [0, 690], '24A': [230, 690], '34A': [460, 690], 
            '13A': [0, 460], '23A': [230, 460], '12A': [0, 230], 
            '22A': [230, 230], '33A': [460, 460], '21A': [230, 0], 
            '32A': [460, 230], '31A': [460, 0], '44A': [690, 690], 
            '43A': [690, 460], '41A': [690, 0], '42A': [690, 230], 
            '52A': [920, 230], '51A': [920, 0], '53A': [920, 460], 
            '54A': [920, 690], '61A': [1150, 0], '62A': [1150, 230], 
            '71A': [1380, 0], '63A': [1150, 460], '72A': [1380, 230], 
            '82A': [1610, 230], '73A': [1380, 460], '83A': [1610, 460], 
            '64A': [1150, 690], '74A': [1380, 690], '84A': [1610, 690], 
            '85A': [1610, 920], '75A': [1380, 920], '65A': [1150, 920], 
            '86A': [1610, 1150], '76A': [1380, 1150], '87A': [1610, 1380], 
            '77A': [1380, 1380], '66A': [1150, 1150], '78A': [1380, 1610], 
            '67A': [1150, 1380], '68A': [1150, 1610], '55A': [920, 920], 
            '56A': [920, 1150], '58A': [920, 1610], '57A': [920, 1380]}

    def electrode_toggle(self, raw_data):
        if raw_data.new_data_size[1] > 100:
            # Key values (electrode names) from mea_120_coordinates only.
            self.electrode_names = list(self.mea_120_coordinates.keys())
            self.electrode_coords_x = np.array(
                [i[0] for i in self.mea_120_coordinates.values()])
            self.electrode_coords_y = np.array(
                [i[1] for i in self.mea_120_coordinates.values()])
        elif raw_data.new_data_size[1] < 100:
            self.electrode_names = list(self.mea_60_coordinates.keys())
            self.electrode_coords_x = np.array(
                [i[0] for i in self.mea_60_coordinates.values()])
            self.electrode_coords_y = np.array(
                [i[1] for i in self.mea_60_coordinates.values()])


# Import data files.  Files must be in .txt or .csv format.  May add toggles or 
# checks to support more data types.
def data_import(analysisGUI, raw_data, electrode_config):
    try:
        data_filename_and_path = tk.filedialog.askopenfilename(
            initialdir=analysisGUI.file_path.get(), title="Select file",
            filetypes=(("txt files", "*.txt"), ("all files", "*.*")))

        import_path, import_filename = os.path.split(data_filename_and_path)

        # Checks whether data was previously imported into program.  If True, 
        # the previous data is deleted.
        if hasattr(raw_data, 'imported') is True:
            print("Raw data is not empty; clearing before reading file.")
            delattr(raw_data, 'imported')
            delattr(raw_data, 'names')

        # print("Importing data...")
        print("Import data began at: ", 
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Import electrodes for column headers from file.
        raw_data.names = pd.read_csv(data_filename_and_path, sep="\s+\t", 
            lineterminator='\n', skiprows=[0, 1, 3], header=None, nrows=1, 
            encoding='iso-8859-15', skipinitialspace=True, engine='python')

        # # Import data from file.
        raw_data.imported = pd.read_csv(data_filename_and_path, sep='\s+', 
            lineterminator='\n', skiprows=3, header=0, encoding='iso-8859-15', 
            skipinitialspace=True, low_memory=False)

        print("Import data completed at: ", 
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Update file name display in GUI following import
        analysisGUI.file_name_label.configure(text=import_filename)
        analysisGUI.file_path.set(import_path)
        raw_data.new_data_size = np.shape(raw_data.imported)
        print(raw_data.new_data_size[1])
        electrode_config.electrode_toggle(raw_data)

    except FileNotFoundError:
        print()
    except TypeError:
        print()


# Usually just for debugging, prints out values upon button press.
def data_print(analysisGUI, raw_data, pace_maker, input_param, electrode_config):
    # adding .iloc to a data frame allows to reference [row, column], 
    # where rows and columns can be ranges separated by colons
    # input_param.beat_choice = int(analysisGUI.mea_beat_select.get()) - 1
    # print(pace_maker.param_dist_normalized.name)
    # print(electrode_config.electrode_names[0])
    # print(electrode_config.electrode_names[5])
    print(analysisGUI.sample_frequency_val.get())
    print(analysisGUI.sample_frequency_menu['menu'].keys())
    # print(input_param.beat_choice)


# Reloads given module.  This is used for testing/developing a module to save 
# time vs re-running the program over and over.
def reload_module():
    # importlib.reload(param_vs_distance_stats)
    # importlib.reload(determine_beats)
    # importlib.reload(calculate_lat)
    # importlib.reload(calculate_upstroke_vel)
    importlib.reload(psd_plotting)
    # from param_vs_distance_stats import param_vs_distance_graphing, param_vs_distance_analysis
    print("Reloaded module.")


# ##############################################################################
# ############################# Graphing Starts ################################
# ##############################################################################

# This function is called following the use of "Calculate All Parameters" from 
# the drop-down menu and from the GUI slider on the main window.  It generates 
# the heat maps observed in the main window of the program. The code is largely 
# identical with the individual functions, which (will soon) open their own 
# windows for each calculation.
def graph_all(analysisGUI, heat_map, cm_beats, pace_maker, upstroke_vel, local_act_time, 
conduction_vel, input_param):
    # ----------------------------- Pacemaker ----------------------------------
    if hasattr(heat_map, 'cbar_1') is True:
        heat_map.cbar_1.remove()
        delattr(heat_map, 'cbar_1')

    heat_map.axis1.cla()
    input_param.beat_choice = int(analysisGUI.mea_beat_select.get()) - 1

    electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table = pace_maker.param_dist_normalized.pivot(index='Y', 
        columns='X', values=pace_maker.final_dist_beat_count[
        input_param.beat_choice])

    heat_map.temp = sns.heatmap(heatmap_pivot_table, cmap="jet", 
        annot=electrode_names, fmt="", ax=heat_map.axis1,
        vmin=0, vmax=pace_maker.param_dist_normalized_max, cbar=False)
    mappable = heat_map.temp.get_children()[0]
    heat_map.cbar_1 = heat_map.axis1.figure.colorbar(mappable, ax=heat_map.axis1)
    heat_map.cbar_1.ax.set_title("Time Lag (ms)", fontsize=10)

    heat_map.axis1.set(title="Pacemaker", 
        xlabel="X coordinate (μm)",
        ylabel="Y coordinate (μm)")

    # --------------------------- Upstroke velocity ----------------------------
    if hasattr(heat_map, 'cbar_2') is True:
        heat_map.cbar_2.remove()
        delattr(heat_map, 'cbar_2')
    heat_map.axis2.cla()
    input_param.beat_choice_2 = int(analysisGUI.mea_beat_select.get()) - 1

    electrode_names_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', 
        columns='X', values=upstroke_vel.final_dist_beat_count[
        input_param.beat_choice_2])

    heat_map.temp_2 = sns.heatmap(heatmap_pivot_table_2, cmap="jet", 
        annot=electrode_names_2, fmt="", ax=heat_map.axis2,
        vmax=upstroke_vel.param_dist_normalized_max, cbar=False)
    mappable_2 = heat_map.temp_2.get_children()[0]
    heat_map.cbar_2 = heat_map.axis2.figure.colorbar(mappable_2, ax=heat_map.axis2)
    heat_map.cbar_2.ax.set_title("μV/ms", fontsize=10)

    heat_map.axis2.set(title="Upstroke Velocity", 
        xlabel="X coordinate (μm)", 
        ylabel="Y coordinate (μm)")

    # ------------------------- Local activation time --------------------------
    if hasattr(heat_map, 'cbar_3') is True:
        heat_map.cbar_3.remove()
        delattr(heat_map, 'cbar_3')
    heat_map.axis3.cla()
    input_param.beat_choice_3 = int(analysisGUI.mea_beat_select.get()) - 1

    electrode_names_3 = local_act_time.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table_3 = local_act_time.param_dist_normalized.pivot(index='Y', 
        columns='X', values=local_act_time.final_dist_beat_count[
        input_param.beat_choice_3])

    heat_map.temp_3 = sns.heatmap(heatmap_pivot_table_3, cmap="jet", 
        annot=electrode_names_3, fmt="", ax=heat_map.axis3,
        vmax=local_act_time.param_dist_normalized_max, cbar=False)
    mappable_3 = heat_map.temp_3.get_children()[0]
    heat_map.cbar_3 = heat_map.axis3.figure.colorbar(mappable_3, ax=heat_map.axis3)
    heat_map.cbar_3.ax.set_title("Time Lag (ms)", fontsize=10)

    heat_map.axis3.set(title="Local Activation Time", 
        xlabel="X coordinate (μm)", 
        ylabel="Y coordinate (μm)")

    # -------------------------- Conduction velocity ---------------------------
    if hasattr(heat_map, 'cbar_4') is True:
        heat_map.cbar_4.remove()
        delattr(heat_map, 'cbar_4')
    heat_map.axis4.cla()
    input_param.beat_choice_4 = int(analysisGUI.mea_beat_select.get()) - 1

    electrode_names_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
        columns='X', values=local_act_time.final_dist_beat_count[
        input_param.beat_choice_4])

    heat_map.temp_4 = sns.heatmap(heatmap_pivot_table_4, cmap="jet", 
        annot=electrode_names_4, fmt="", ax=heat_map.axis4, cbar=False)
    mappable_4 = heat_map.temp_4.get_children()[0]
    heat_map.cbar_4 = heat_map.axis4.figure.colorbar(mappable_4, ax=heat_map.axis4)
    heat_map.cbar_4.ax.set_title("μm/(ms)", fontsize=10)

    heat_map.axis4.set(title="Conduction Velocity" , 
        xlabel="X coordinate (μm)", 
        ylabel="Y coordinate (μm)")

    heat_map.curr_plot.tight_layout()
    heat_map.curr_plot.subplots_adjust(top=0.9)
    heat_map.curr_plot.suptitle("Parameter Heatmaps. Beat " + 
        str(input_param.beat_choice + 1) + " of " + 
        str(int(cm_beats.beat_count_dist_mode[0])) + ".")
    heat_map.curr_plot.canvas.draw()


# Construct heatmap from previously calculated pacemaker data.  Function is 
# called each time the slider is moved to select a new beat.
def graph_pacemaker(analysisGUI, heat_map, pace_maker, input_param):
    try:
        if hasattr(heat_map, 'pm_solo_cbar') is True:
            heat_map.pm_solo_cbar.remove()
            delattr(heat_map, 'pm_solo_cbar')

        heat_map.pm_solo_axis.cla()
        input_param.pm_solo_beat_choice = int(analysisGUI.pm_solo_beat_select.get()) - 1

        electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table = pace_maker.param_dist_normalized.pivot(index='Y', 
            columns='X', values=pace_maker.final_dist_beat_count[input_param.pm_solo_beat_choice])

        heat_map.pm_solo_temp = sns.heatmap(heatmap_pivot_table, cmap="jet", 
            annot=electrode_names, fmt="", ax=heat_map.pm_solo_axis, vmin=0, 
            vmax=pace_maker.param_dist_normalized_max, cbar=False)
        mappable = heat_map.pm_solo_temp.get_children()[0]
        heat_map.pm_solo_cbar = heat_map.pm_solo_axis.figure.colorbar(mappable, 
            ax=heat_map.pm_solo_axis)
        heat_map.pm_solo_cbar.ax.set_title("Time Lag (ms)", fontsize=10)

        heat_map.pm_solo_axis.set(title="Pacemaker, Beat " + 
            str(input_param.pm_solo_beat_choice+1), xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")
        heat_map.pm_solo_plot.tight_layout()
        heat_map.pm_solo_plot.canvas.draw()

    except AttributeError:
        print("Please calculate PM first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def graph_upstroke(analysisGUI, heat_map, upstroke_vel, input_param):
    try:
        if hasattr(heat_map, 'dvdt_solo_cbar') is True:
            heat_map.dvdt_solo_cbar.remove()
            delattr(heat_map, 'dvdt_solo_cbar')

        heat_map.dvdt_solo_axis.cla()
        input_param.dvdt_solo_beat_choice = int(analysisGUI.dvdt_solo_beat_select.get()) - 1

        electrode_names_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', 
            columns='X', values=upstroke_vel.final_dist_beat_count[input_param.dvdt_solo_beat_choice])

        heat_map.dvdt_solo_temp = sns.heatmap(heatmap_pivot_table_2, cmap="jet", 
            annot=electrode_names_2, fmt="", ax=heat_map.dvdt_solo_axis, 
            vmax=upstroke_vel.param_dist_normalized_max, cbar=False)
        mappable_2 = heat_map.dvdt_solo_temp.get_children()[0]
        heat_map.dvdt_solo_cbar = heat_map.dvdt_solo_axis.figure.colorbar(mappable_2, 
            ax=heat_map.dvdt_solo_axis)
        heat_map.dvdt_solo_cbar.ax.set_title("μV/ms", fontsize=10)

        heat_map.dvdt_solo_axis.set(title="Upstroke Velocity, Beat " + 
            str(input_param.dvdt_solo_beat_choice+1), xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")
        heat_map.dvdt_solo_plot.tight_layout()
        heat_map.dvdt_solo_plot.canvas.draw()

    except AttributeError:
        print("Please calculate dV/dt first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def graph_local_act_time(analysisGUI, heat_map, local_act_time, input_param):
    try:
        if hasattr(heat_map, 'lat_solo_cbar') is True:
            heat_map.lat_solo_cbar.remove()
            delattr(heat_map, 'lat_solo_cbar')
        
        heat_map.lat_solo_axis.cla()
        input_param.lat_solo_beat_choice = int(analysisGUI.lat_solo_beat_select.get()) - 1

        electrode_names_3 = local_act_time.param_dist_normalized.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table_3 = local_act_time.param_dist_normalized.pivot(
            index='Y', columns='X', 
            values=local_act_time.final_dist_beat_count[input_param.lat_solo_beat_choice])

        heat_map.lat_solo_temp = sns.heatmap(heatmap_pivot_table_3, cmap="jet", 
            annot=electrode_names_3, fmt="", ax=heat_map.lat_solo_axis, 
            vmax=local_act_time.param_dist_normalized_max, cbar=False)
        mappable_3 = heat_map.lat_solo_temp.get_children()[0]
        heat_map.lat_solo_cbar = heat_map.lat_solo_axis.figure.colorbar(mappable_3, 
            ax=heat_map.lat_solo_axis)
        heat_map.lat_solo_cbar.ax.set_title("Time Lag (ms)", fontsize=10)

        heat_map.lat_solo_axis.set(title="Local Activation Time, Beat " + 
            str(input_param.lat_solo_beat_choice+1), xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")
        heat_map.lat_solo_plot.tight_layout()
        heat_map.lat_solo_plot.canvas.draw()

    except AttributeError:
        print("Please calculate LAT first.")
    except IndexError:
        print("You entered a beat that does not exist.")


def graph_conduction_vel(analysisGUI, heat_map, local_act_time, conduction_vel, 
input_param):
    try:
        if hasattr(heat_map, 'cv_solo_cbar') is True:
            heat_map.cv_solo_cbar.remove()
            delattr(heat_map, 'cv_solo_cbar')
        
        heat_map.cv_solo_axis.cla()
        input_param.cv_solo_beat_choice = int(analysisGUI.cv_solo_beat_select.get()) - 1

        electrode_names_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
            columns='X', values='Electrode')
        heatmap_pivot_table_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
            columns='X', values=local_act_time.final_dist_beat_count[input_param.cv_solo_beat_choice])

        heat_map.cv_solo_temp = sns.heatmap(heatmap_pivot_table_4, cmap="jet", 
            annot=electrode_names_4, fmt="", ax=heat_map.cv_solo_axis, cbar=False)
        mappable_4 = heat_map.cv_solo_temp.get_children()[0]
        heat_map.cv_solo_cbar = heat_map.cv_solo_axis.figure.colorbar(mappable_4, 
            ax=heat_map.cv_solo_axis)
        heat_map.cv_solo_cbar.ax.set_title("μm/(ms)", fontsize=10)

        heat_map.cv_solo_axis.set(title="Conduction Velocity, Beat " + 
            str(input_param.cv_solo_beat_choice+1), xlabel="X coordinate (μm)", 
            ylabel="Y coordinate (μm)")
        heat_map.cv_solo_plot.tight_layout()
        heat_map.cv_solo_plot.canvas.draw()

    except AttributeError:
        print("Please make sure you've calculated Local Activation Time first.")
    except IndexError:
        print("You entered a beat that does not exist.")


# Calls the PandasGUI function from the pandasgui library (external dev!)
def show_dataframes(raw_data, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel):
    try:
        cm_beats_dist_data = cm_beats.dist_beats
        pm_normalized = pace_maker.param_dist_normalized
        dVdt_normalized = upstroke_vel.param_dist_normalized
        lat_normalized = local_act_time.param_dist_normalized
        lat_distances = local_act_time.distance_from_min
        cv_raw = conduction_vel.param_dist_raw
        pgui.show(cm_beats_dist_data, pm_normalized, dVdt_normalized, 
            lat_normalized, lat_distances, cv_raw, settings={'block': True})
    except(AttributeError):
        print("Please run all of your calculations first.")


# Toggles display of truncation start and end entry fields.
def trunc_toggle(analysisGUI):
    if analysisGUI.trunc_toggle_on_off.get() == True:
        analysisGUI.trunc_start_value.grid()
        analysisGUI.trunc_end_value.grid()
    if analysisGUI.trunc_toggle_on_off.get() == False:
        analysisGUI.trunc_start_value.grid_remove()
        analysisGUI.trunc_end_value.grid_remove()
        analysisGUI.trunc_start_text.set("Start (Min)")
        analysisGUI.trunc_end_text.set("End (Min)")


class MainGUI(tk.Frame):
    def __init__(self, master, raw_data, cm_beats, pace_maker, upstroke_vel, 
    local_act_time, conduction_vel, input_param, heat_map, cm_stats, 
    electrode_config, psd_data):
        tk.Frame.__init__(self, master)
        # The fun story about grid: columns and rows cannot be generated past 
        # the number of widgets you have (or at least I've yet to learn the way 
        # to do so, and will update if I find out how). It's all relative geometry,
        # where the relation is with other widgets.  If you have 3 widgets you 
        # can have 3 rows or 3 columns and organize accordingly.  If you have 2 
        # widgets you can only have 2 rows or 2 columns.
        # use .bind("<Enter>", "color") or .bind("<Leave>", "color") to change 
        # mouse-over color effects.
        self.grid()
        self.winfo_toplevel().title("MEA Analysis - v4")

        # Directory information for file import is stored here and called upon 
        # by import function.  Default/initial "/"
        self.file_path = tk.StringVar()
        self.file_path.set("/")

        # ############################ Menu ####################################
        self.master = master
        menu = tk.Menu(self.master, tearoff=False)
        self.master.config(menu=menu)
        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Data (.csv or .txt)", 
            command=lambda: data_import(self, raw_data, electrode_config))
        file_menu.add_command(label="Save Processed Data", command=None)
        file_menu.add_command(label="Save Heatmaps", command=None)
        file_menu.add_command(label="Print (Debug)", 
            command=lambda: data_print(self, raw_data, pace_maker, input_param,
            electrode_config))
        file_menu.add_command(label="Truncate Data", command=None)

        view_menu = tk.Menu(menu)
        menu.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="View Pandas Dataframes", 
            command=lambda: show_dataframes(raw_data, cm_beats, pace_maker, 
                upstroke_vel, local_act_time, conduction_vel))

        calc_menu = tk.Menu(menu)
        menu.add_cascade(label="Calculations", menu=calc_menu)
        calc_menu.add_command(label="Beat Detect (Run First!)", 
            command=lambda: [determine_beats.determine_beats(self, raw_data, cm_beats, input_param),
                self.beat_detect_window(cm_beats, input_param),
                determine_beats.graph_beats(self, cm_beats, input_param)])
        calc_menu.add_command(label="Calculate All Parameters",
            command=lambda: [calculate_pacemaker(self, cm_beats, pace_maker, 
                heat_map, input_param, electrode_config),
                calculate_upstroke_vel.calculate_upstroke_vel(self, cm_beats, 
                upstroke_vel, heat_map, input_param, electrode_config),
                calculate_lat.calculate_lat(self, cm_beats, local_act_time, heat_map, 
                input_param, electrode_config),
                calculate_conduction_velocity(self, conduction_vel, 
                local_act_time, heat_map, input_param, electrode_config),
                graph_all(self, heat_map, cm_beats, pace_maker, upstroke_vel, 
                local_act_time, conduction_vel, input_param)])
        # Add extra command for each solitary calculation that calls the 
        # appropriate graphing function. The graphing function will call the 
        # appropriate method to open the window.  This will allow for individual
        # parameter analysis.
        calc_menu.add_command(label="Pacemaker", 
            command=lambda: [calculate_pacemaker(self, cm_beats, pace_maker, 
                heat_map, input_param, electrode_config),
                self.pacemaker_heatmap_window(cm_beats, pace_maker, heat_map, 
                input_param),
                graph_pacemaker(self, heat_map, pace_maker, input_param)])
        calc_menu.add_command(label="Upstroke Velocity", 
            command=lambda: [calculate_upstroke_vel.calculate_upstroke_vel(self, 
                cm_beats, upstroke_vel, heat_map, input_param, electrode_config),
                self.dvdt_heatmap_window(cm_beats, upstroke_vel, heat_map, 
                input_param),
                graph_upstroke(self, heat_map, upstroke_vel, input_param)])
        calc_menu.add_command(label="Local Activation Time", 
            command=lambda: [calculate_lat.calculate_lat(self, cm_beats, 
                local_act_time, heat_map, input_param, electrode_config),
                self.lat_heatmap_window(cm_beats, local_act_time, heat_map, 
                input_param),
                graph_local_act_time(self, heat_map, local_act_time, input_param)])
        calc_menu.add_command(label="Conduction Velocity", 
            command=lambda: [calculate_conduction_velocity(self, conduction_vel, 
                local_act_time, heat_map, input_param, electrode_config),
                self.cv_heatmap_window(cm_beats, local_act_time, conduction_vel, 
                heat_map, input_param),
                graph_conduction_vel(self, heat_map, local_act_time, 
                conduction_vel, input_param)])

        statistics_menu = tk.Menu(menu)
        menu.add_cascade(label="Statistics", menu=statistics_menu)
        statistics_menu.add_command(label="Parameter vs Distance Plot w/ R-Square", 
            command=lambda: [self.param_vs_dist_stats_window(cm_beats, 
            pace_maker, upstroke_vel, local_act_time, conduction_vel, 
            input_param, cm_stats)])
        statistics_menu.add_command(label="Power Spectrum", 
            command=lambda: [self.psd_plot_window(cm_beats, pace_maker, 
            upstroke_vel, local_act_time, conduction_vel, input_param, 
            cm_stats, psd_data)])
        statistics_menu.add_command(label="Radial Binning Plot w/ R-Square", command=None)
        statistics_menu.add_command(label="Q-Q Plot",  command=None)

        advanced_tools_menu = tk.Menu(menu)
        menu.add_cascade(label="Advanced Tools", menu=advanced_tools_menu)
        advanced_tools_menu.add_command(label="K-Means Clustering", command=None)
        advanced_tools_menu.add_command(label="t-SNE", command=None)
        advanced_tools_menu.add_command(label="DBSCAN", command=None)
        advanced_tools_menu.add_command(label="PCA", command=None)

        testing_menu = tk.Menu(menu)
        menu.add_cascade(label="Testing", menu=testing_menu)
        testing_menu.add_command(label="Reload Module", command=lambda: reload_module())


        # ######################### Entry Fields ###############################
        # Frame for MEA parameters (e.g. plotted electrode, min peak distance, 
        # min peak amplitude, prominence, etc)
        self.mea_parameters_frame = tk.Frame(self, width=1620, height=80, bg="white")
        self.mea_parameters_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        self.mea_parameters_frame.grid_propagate(False)

        self.file_name_label = tk.Label(self.mea_parameters_frame, 
            text="No file", bg="white", wraplength=300)
        self.file_name_label.grid(row=0, column=11, columnspan=4, padx=5, pady=5)
        self.file_length_label = tk.Label(self.mea_parameters_frame, 
            text="TBD", bg="white", wraplength=200)
        self.file_length_label.grid(row=1, column=11, columnspan=4, padx=5, pady=5)

        # Min peak height label, entry field, trace and positioning.
        self.min_peak_height_label = tk.Label(self.mea_parameters_frame, 
            text="Min Peak Height", bg="white", wraplength=80)
        self.min_peak_height_label.grid(row=0, column=1, padx=5, pady=2)
        self.min_peak_height_val = tk.StringVar()
        self.min_peak_height_val.trace_add("write", self.min_peak_height_callback)
        self.min_peak_height_val.set("100")
        self.min_peak_height_entry = tk.Entry(self.mea_parameters_frame, 
            text=self.min_peak_height_val, width=8)
        self.min_peak_height_entry.grid(row=1, column=1, padx=5, pady=2)

        # Min peak distance label, entry field, trace and positioning.
        self.min_peak_dist_label = tk.Label(self.mea_parameters_frame, 
            text="Min Peak Distance", bg="white", wraplength=80)
        self.min_peak_dist_label.grid(row=0, column=2, padx=5, pady=2)
        self.min_peak_dist_val = tk.StringVar()
        self.min_peak_dist_val.trace_add("write", self.min_peak_dist_callback)
        self.min_peak_dist_val.set("1000")
        self.min_peak_dist_entry = tk.Entry(self.mea_parameters_frame, 
            text=self.min_peak_dist_val, width=8)
        self.min_peak_dist_entry.grid(row=1, column=2, padx=5, pady=2)

        # Peak prominence label, entry field, trace and positioning.
        self.parameter_prominence_label = tk.Label(self.mea_parameters_frame, 
            text="Peak Prominence", bg="white", wraplength=100)
        self.parameter_prominence_label.grid(row=0, column=3, padx=5, pady=2)
        self.parameter_prominence_val = tk.StringVar()
        self.parameter_prominence_val.trace_add("write", self.parameter_prominence_callback)
        self.parameter_prominence_val.set("100")
        self.parameter_prominence_entry = tk.Entry(self.mea_parameters_frame, 
            text=self.parameter_prominence_val, width=8)
        self.parameter_prominence_entry.grid(row=1, column=3, padx=5, pady=2)

        # Peak width label, entry field, trace and positioning.
        self.parameter_width_label = tk.Label(self.mea_parameters_frame, 
            text="Peak Width", bg="white", wraplength=100)
        self.parameter_width_label.grid(row=0, column=4, padx=5, pady=2)
        self.parameter_width_val = tk.StringVar()
        self.parameter_width_val.trace_add("write", self.parameter_width_callback)
        self.parameter_width_val.set("3")
        self.parameter_width_entry = tk.Entry(self.mea_parameters_frame, 
            text=self.parameter_width_val, width=8)
        self.parameter_width_entry.grid(row=1, column=4, padx=5, pady=2)

        # Peak threshold label, entry field, trace and positioning.
        self.parameter_thresh_label = tk.Label(self.mea_parameters_frame, 
            text="Peak Threshold", bg="white", wraplength=100)
        self.parameter_thresh_label.grid(row=0, column=5, padx=5, pady=2)
        self.parameter_thresh_val = tk.StringVar()
        self.parameter_thresh_val.trace_add("write", self.parameter_thresh_callback)
        self.parameter_thresh_val.set("50")
        self.parameter_thresh_entry = tk.Entry(self.mea_parameters_frame, 
            text=self.parameter_thresh_val, width=8)
        self.parameter_thresh_entry.grid(row=1, column=5, padx=5, pady=2)

        # Sample Frequency label, entry field and positioning.
        self.sample_frequency_label = tk.Label(self.mea_parameters_frame,
            text="Sample Freq. (Hz)", bg="white", wraplength=100)
        self.sample_frequency_label.grid(row=0, column=6, padx=5, pady=2)
        frequency_values = ('1000', '10000')
        self.sample_frequency_val = tk.StringVar()
        self.sample_frequency_val.set(frequency_values[0])
        self.sample_frequency_menu = tk.OptionMenu(self.mea_parameters_frame,
            self.sample_frequency_val, *frequency_values)
        self.sample_frequency_menu.grid(row=1, column=6, padx=5, pady=2)
        self.sample_frequency_menu.configure(bg="white", borderwidth=0, 
            width=8)

        # Truncation on/off toggle checkbox and positioning.
        self.trunc_toggle_on_off = tk.BooleanVar()
        self.trunc_toggle_box = tk.Checkbutton(self.mea_parameters_frame,
            text="Truncate Data", variable=self.trunc_toggle_on_off,
            onvalue=True, offvalue=False, background="white", width=18,
            justify="left", command=lambda: trunc_toggle(self))
        self.trunc_toggle_box.grid(row=0, column=7, columnspan=2, padx=5, pady=2)

        # Truncation start value entry text, field and positioning.
        self.trunc_start_text = tk.StringVar()
        self.trunc_start_text.set("Start (Min)")
        self.trunc_start_value = tk.Entry(self.mea_parameters_frame, width=9,
            bg="white", textvariable=self.trunc_start_text)
        self.trunc_start_value.grid(row=1, column=7, padx=5, pady=2)
        self.trunc_start_value.grid_remove()
        
        # Truncation end value entry text, field and positioning.
        self.trunc_end_text = tk.StringVar()
        self.trunc_end_text.set("End (Min)")
        self.trunc_end_value = tk.Entry(self.mea_parameters_frame, width=9,
            bg="white", textvariable=self.trunc_end_text)
        self.trunc_end_value.grid(row=1, column=8, padx=5, pady=2)
        self.trunc_end_value.grid_remove()

        # ############################# Heatmap ################################
        # Frame and elements for pacemaker heat map plot.
        self.mea_heatmap_frame = tk.Frame(self, width=1620, height=800, bg="white")
        self.mea_heatmap_frame.grid(row=1, column=0, padx=5, pady=5)
        self.mea_heatmap_frame.grid_propagate(False)
        self.gen_all_heatmap = FigureCanvasTkAgg(heat_map.curr_plot, self.mea_heatmap_frame)
        self.gen_all_heatmap.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        
        # Beat select slider, belongs to different frame.
        self.mea_beat_select = tk.Scale(self.mea_parameters_frame, length=125, 
            width=15, from_=1, to=20, orient="horizontal", bg="white", 
            label="Current Beat:")
        self.mea_beat_select.grid(row=0, column=9, rowspan=2, padx=5, pady=5)
        self.mea_beat_select.bind("<ButtonRelease-1>",
            lambda event: graph_all(self, heat_map, cm_beats, pace_maker, 
                upstroke_vel, local_act_time, conduction_vel, input_param))
        self.toolbar_all_heatmap_frame = tk.Frame(self.mea_parameters_frame)
        self.toolbar_all_heatmap_frame.grid(row=0, column=10, rowspan=2, padx=5, pady=5)
        self.toolbar_all_heatmap = NavigationToolbar2Tk(self.gen_all_heatmap, self.toolbar_all_heatmap_frame)

        # The following lines are for the GUI controls found in child 
        # windows when doing beat detect or the individual,
        # or solo, calculations for the respective parameter.
        self.elec_to_plot_val = tk.StringVar()
        self.elec_to_plot_val.set("1")
        self.elec_to_plot_val.trace_add("write", self.col_sel_callback)
        self.pm_solo_beat_select = None
        self.dvdt_solo_beat_select = None
        self.lat_solo_beat_select = None
        self.cv_solo_beat_select = None
        self.param_vs_dist_beat_select = None
        self.param_vs_dist_sigma_value = tk.StringVar()
        self.param_vs_dist_sigma_value.set("3")
        self.stat_readout_text = tk.StringVar()
        self.stat_readout_text.set("TBD")
        self.stat_file_name = tk.StringVar()
        self.stat_file_name.set("No file")
        self.psd_file_name = tk.StringVar()
        self.psd_file_name.set("No file")
        self.psd_start_beat = tk.StringVar()
        self.psd_start_beat.set("Start (Beat)")
        self.psd_end_beat = tk.StringVar()
        self.psd_end_beat.set("End (Beat)")
        self.psd_beats = ["Beat " + str(i) for i in range(1, 11)]
        self.psd_start_beat_value = None
        self.psd_end_beat_value = None
        # print(dir(self))

    def beat_detect_window(self, cm_beats, input_param):
        beat_detect = tk.Toplevel(self)
        beat_detect.title('Beat Detect Window')
        beat_detect.geometry('1250x850')
        beat_detect_frame = tk.Frame(beat_detect, width=1200, height=850, bg="white")
        beat_detect_frame.grid(row=0, column=0, padx=5, pady=5)
        beat_detect_frame.grid_propagate(False)
        gen_beats_fig = FigureCanvasTkAgg(cm_beats.comp_plot, beat_detect_frame)
        gen_beats_fig.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        # NavigationToolbar2Tk calls pack internally, conflicts with grid.  
        # Workaround: establish in own frame, use grid to place that frame 
        # inside of the chosen parent frame. This works because the parent 
        # frame is still a descent of "root", which is the overarching parent 
        # of all of these GUI elements.
        gen_beats_toolbar_frame = tk.Frame(beat_detect)
        gen_beats_toolbar_frame.grid(row=4, column=0, columnspan=2, in_=beat_detect_frame)
        gen_beats_toolbar = NavigationToolbar2Tk(gen_beats_fig, gen_beats_toolbar_frame)

        # Electrode entry field to change display for plot shown.
        elec_to_plot_label = tk.Label(beat_detect_frame, text="Electrode Plotted", 
            bg="white", wraplength=80)
        elec_to_plot_label.grid(row=0, column=0, padx=5, pady=2)
        elec_to_plot_entry = tk.Entry(beat_detect_frame, text=self.elec_to_plot_val, width=8)
        elec_to_plot_entry.grid(row=1, column=0, padx=5, pady=2)

        # Invoke graph_peaks function for plotting only.  Meant to be used after 
        # find peaks, after switching columns.
        graph_beats_button = tk.Button(beat_detect_frame, text="Graph Beats", 
            width=15, height=3, bg="red2",
            command=lambda: determine_beats.graph_beats(self, cm_beats, input_param))
        graph_beats_button.grid(row=0, rowspan=2, column=1, padx=2, pady=2)

    def pacemaker_heatmap_window(self, cm_beats, pace_maker, heat_map, input_param):
        pm_heatmap = tk.Toplevel(self)
        pm_heatmap.title("Pacemaker Heatmap")
        pm_heatmap_frame = tk.Frame(pm_heatmap, width=1400, height=900, bg="white")
        pm_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        pm_heatmap_frame.grid_propagate(False)
        pm_heatmap_fig = FigureCanvasTkAgg(heat_map.pm_solo_plot, pm_heatmap_frame)
        pm_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.pm_solo_beat_select = tk.Scale(pm_heatmap_frame, length=125, 
            width=15, from_=1, to=int(cm_beats.beat_count_dist_mode[0]),
            orient="horizontal", bg="white", label="Current Beat")
        self.pm_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.pm_solo_beat_select.bind("<ButtonRelease-1>",
            lambda event: graph_pacemaker(self, heat_map, pace_maker, input_param))

    def dvdt_heatmap_window(self, cm_beats, upstroke_vel, heat_map, input_param):
        dvdt_heatmap = tk.Toplevel(self)
        dvdt_heatmap.title("Upstroke Velocity Heatmap")
        dvdt_heatmap_frame = tk.Frame(dvdt_heatmap, width=1400, height=900, bg="white")
        dvdt_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        dvdt_heatmap_frame.grid_propagate(False)
        dvdt_heatmap_fig = FigureCanvasTkAgg(heat_map.dvdt_solo_plot, dvdt_heatmap_frame)
        dvdt_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.dvdt_solo_beat_select = tk.Scale(dvdt_heatmap_frame, length=125, 
            width=15, from_=1, to=int(cm_beats.beat_count_dist_mode[0]),
            orient="horizontal", bg="white", label="Current Beat")
        self.dvdt_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.dvdt_solo_beat_select.bind("<ButtonRelease-1>",
            lambda event: graph_upstroke(self, heat_map, upstroke_vel, input_param))

    def lat_heatmap_window(self, cm_beats, local_act_time, heat_map, input_param):
        lat_heatmap = tk.Toplevel(self)
        lat_heatmap.title("Local Activation Time Heatmap")
        lat_heatmap_frame = tk.Frame(lat_heatmap, width=1400, height=900, bg="white")
        lat_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        lat_heatmap_frame.grid_propagate(False)
        lat_heatmap_fig = FigureCanvasTkAgg(heat_map.lat_solo_plot, lat_heatmap_frame)
        lat_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.lat_solo_beat_select = tk.Scale(lat_heatmap_frame, length=125, 
            width=15, from_=1, to=int(cm_beats.beat_count_dist_mode[0]),
            orient="horizontal", bg="white", label="Current Beat")
        self.lat_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.lat_solo_beat_select.bind("<ButtonRelease-1>",
            lambda event: graph_local_act_time(self, heat_map, local_act_time, 
            input_param))

    def cv_heatmap_window(self, cm_beats, local_act_time, conduction_vel, 
    heat_map, input_param):
        cv_heatmap = tk.Toplevel(self)
        cv_heatmap.title("Conduction Velocity Heatmap")
        cv_heatmap_frame = tk.Frame(cv_heatmap, width=1400, height=900, bg="white")
        cv_heatmap_frame.grid(row=0, column=0, padx=5, pady=5)
        cv_heatmap_frame.grid_propagate(False)
        cv_heatmap_fig = FigureCanvasTkAgg(heat_map.cv_solo_plot, cv_heatmap_frame)
        cv_heatmap_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        self.cv_solo_beat_select = tk.Scale(cv_heatmap_frame, length=125, 
            width=15, from_=1,
            to=int(cm_beats.beat_count_dist_mode[0]),
            orient="horizontal", bg="white", label="Current Beat")
        self.cv_solo_beat_select.grid(row=1, column=0, padx=5, pady=5)
        self.cv_solo_beat_select.bind("<ButtonRelease-1>",
            lambda event: graph_conduction_vel(self, heat_map, local_act_time, 
            conduction_vel, input_param))

    def param_vs_dist_stats_window(self, cm_beats, pace_maker, upstroke_vel, 
    local_act_time, conduction_vel, input_param, cm_stats):
        param_vs_dist= tk.Toplevel(self)
        param_vs_dist.title("Parameter vs Distance Plot w/ R-Square")
        param_vs_dist_options_frame = tk.Frame(param_vs_dist, width=1300, 
            height=80, bg="white")
        param_vs_dist_options_frame.grid(row=0, column=0, columnspan=1, 
            padx=5, pady=5)
        param_vs_dist_options_frame.grid_propagate(False)
        param_vs_dist_sigma_label = tk.Label(param_vs_dist_options_frame, 
            text="Sigma Value", bg="white")
        param_vs_dist_sigma_label.grid(row=0, column=0, padx=2, pady=2)
        param_vs_dist_sigma_entry = tk.Entry(param_vs_dist_options_frame, 
            text=self.param_vs_dist_sigma_value, width=8)
        param_vs_dist_sigma_entry.grid(row=1, column=0, padx=5, pady=5)
        param_vs_dist_remove_outliers = tk.Button(param_vs_dist_options_frame, 
            text="Remove Outliers", bg="silver", height=2,
            command=lambda: param_vs_distance_stats.param_vs_distance_analysis(self, 
                cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, 
                input_param, cm_stats))
        param_vs_dist_remove_outliers.grid(row=0, rowspan=2, column=1, 
            padx=5, pady=5)
        
        # Display file name
        self.stat_file_name_label = tk.Label(param_vs_dist_options_frame, 
            textvariable=self.stat_file_name, bg="white", wraplength=300)
        self.stat_file_name_label.grid(row=0, column=5, 
            columnspan=4, padx=5, pady=5)
        
        # Figure frame for statistical best-fit plots.
        param_vs_dist_fig_frame = tk.Frame(param_vs_dist, width=1300, 
            height=800, bg="white")
        param_vs_dist_fig_frame.grid(row=1, column=0, padx=5, pady=5)
        param_vs_dist_fig_frame.grid_propagate(False)
        param_vs_dist_fig = FigureCanvasTkAgg(cm_stats.param_vs_dist_plot, 
            param_vs_dist_fig_frame)
        param_vs_dist_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        
        self.param_vs_dist_beat_select = tk.Scale(param_vs_dist_options_frame, 
            length=125, width=15, from_=1, to=5, 
            orient="horizontal", bg="white", label="Current Beat")
        self.param_vs_dist_beat_select.grid(row=0, rowspan=2, column=2, padx=5, pady=5)
        self.param_vs_dist_beat_select.bind("<ButtonRelease-1>", 
            lambda event: param_vs_distance_stats.param_vs_distance_graphing(self, 
                cm_beats, pace_maker, upstroke_vel, local_act_time, conduction_vel, 
                    input_param, cm_stats))
        
        param_vs_dist_toolbar_frame = tk.Frame(param_vs_dist)
        param_vs_dist_toolbar_frame.grid(row=0, rowspan=2, column=3, columnspan=2, 
            in_=param_vs_dist_options_frame)
        param_vs_dist_toolbar = NavigationToolbar2Tk(param_vs_dist_fig, 
            param_vs_dist_toolbar_frame)

        # Frame and canvas for text printout, with scrollbar.
        param_vs_dist_readout_frame = tk.Frame(param_vs_dist, width=200,
            height=890, bg="white")
        param_vs_dist_readout_frame.grid(row=0, rowspan=2, column=1, padx=5, 
            pady=5)
        param_vs_dist_readout_canvas = tk.Canvas(param_vs_dist_readout_frame, 
            width=200, height=890, bg="white")
        param_vs_dist_readout_canvas.grid(row=0, column=0, sticky="nesw")
        param_vs_dist_readout_scrollbar = tk.Scrollbar(
            param_vs_dist_readout_frame, orient="vertical", 
            command=param_vs_dist_readout_canvas.yview)
        param_vs_dist_readout_scrollbar.grid(row=0, rowspan=2, column=1, sticky="ns")
        param_vs_dist_readout_scrollframe = tk.Frame(param_vs_dist_readout_canvas)
        param_vs_dist_readout_scrollframe.bind("<Configure>", 
            lambda event: param_vs_dist_readout_canvas.configure(
                scrollregion=param_vs_dist_readout_canvas.bbox("all")))
        param_vs_dist_readout_canvas.create_window((0, 0), 
            window=param_vs_dist_readout_scrollframe, anchor="nw")
        param_vs_dist_readout_canvas.configure(
            yscrollcommand=param_vs_dist_readout_scrollbar.set)
        param_vs_dist_readout_header = tk.Label(param_vs_dist_readout_scrollframe,
            bg="white", anchor="nw", justify="left", width=24,
            text="Data Set-Wide Outputs:", 
            font=('Helvetica', '11', 'bold')).grid(row=0, column=0, sticky="nw")
        param_vs_dist_readout_text = tk.Label(param_vs_dist_readout_scrollframe,
            bg="white", anchor="w", justify="left", width=24,
            textvariable=self.stat_readout_text).grid(row=1, column=0, sticky="w")

    def psd_plot_window(self, cm_beats, pace_maker, upstroke_vel, 
    local_act_time, conduction_vel, input_param, cm_stats, psd_data):
        psd_window= tk.Toplevel(self)
        psd_window.title("Log-Log and Power Spectrum")
        psd_window_options_frame = tk.Frame(psd_window, width=1300, 
            height=80, bg="white")
        psd_window_options_frame.grid(row=0, column=0, columnspan=1, 
            padx=5, pady=5)
        psd_window_options_frame.grid_propagate(False)

        # Button to generate plots.
        psd_window_plotting = tk.Button(psd_window_options_frame, 
            text="Plot PSD", bg="silver", height=2,
            command=lambda: psd_plotting.psd_plotting(self, cm_beats, 
                pace_maker, upstroke_vel, local_act_time, conduction_vel, 
                input_param, psd_data))
        psd_window_plotting.grid(row=0, rowspan=2, column=0, 
            padx=5, pady=5)
        
        # Entry fields for defining range of interest, in terms of beats.
        psd_beat_interval_label = tk.Label(psd_window_options_frame, width=22,
            bg="white smoke", text="Start/End Beats", borderwidth=1)
        psd_beat_interval_label.grid(row=0, column=1, columnspan=2, padx=5,
            pady=2)

        self.psd_start_beat_value = ttk.Combobox(psd_window_options_frame, width=10, 
            textvariable=self.psd_start_beat, values=self.psd_beats)
        self.psd_start_beat_value.grid(row=1, column=1, padx=5, pady=2)
        self.psd_start_beat_value.state(['readonly'])
        
        self.psd_end_beat_value = ttk.Combobox(psd_window_options_frame, width=10,
            textvariable=self.psd_end_beat, values=self.psd_beats)
        self.psd_end_beat_value.grid(row=1, column=2, padx=5, pady=2)
        self.psd_end_beat_value.state(['readonly'])

        # Display file name
        self.psd_file_name_label = tk.Label(psd_window_options_frame, 
            textvariable=self.psd_file_name, bg="white", wraplength=300)
        self.psd_file_name_label.grid(row=0, column=6, 
            columnspan=4, padx=5, pady=5)
        
        # Figure frame for statistical best-fit plots.
        psd_fig_frame = tk.Frame(psd_window, width=1300, 
            height=800, bg="white")
        psd_fig_frame.grid(row=1, column=0, padx=5, pady=5)
        psd_fig_frame.grid_propagate(False)
        psd_fig = FigureCanvasTkAgg(psd_data.psd_plots, 
            psd_fig_frame)
        psd_fig.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        
        self.psd_electrode_select = tk.Scale(psd_window_options_frame, 
            length=125, width=15, from_=1, to=5, 
            orient="horizontal", bg="white", label="Beat")
        self.psd_electrode_select.grid(row=0, rowspan=2, column=3, padx=5, pady=5)
        self.psd_electrode_select.bind("<ButtonRelease-1>", 
            lambda event: psd_plotting.plot_log_vs_log(self, cm_beats, 
                pace_maker, upstroke_vel, local_act_time, conduction_vel, 
                input_param, psd_data))
        
        psd_toolbar_frame = tk.Frame(psd_window)
        psd_toolbar_frame.grid(row=0, rowspan=2, column=4, columnspan=2, 
            in_=psd_window_options_frame)
        psd_toolbar = NavigationToolbar2Tk(psd_fig, 
            psd_toolbar_frame)


    def col_sel_callback(self, *args):
        print("You entered: \"{}\"".format(self.elec_to_plot_val.get()))
        try:
            chosen_electrode_val = int(self.elec_to_plot_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def min_peak_dist_callback(self, *args):
        print("You entered: \"{}\"".format(self.min_peak_dist_val.get()))
        try:
            chosen_electrode_val = int(self.min_peak_dist_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def min_peak_height_callback(self, *args):
        print("You entered: \"{}\"".format(self.min_peak_height_val.get()))
        try:
            chosen_electrode_val = int(self.min_peak_height_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_prominence_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_prominence_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_prominence_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_width_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_width_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_width_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")

    def parameter_thresh_callback(self, *args):
        print("You entered: \"{}\"".format(self.parameter_thresh_val.get()))
        try:
            chosen_electrode_val = int(self.parameter_thresh_val.get())
        except ValueError:
            print("Only numbers are allowed.  Please try again.")


def main():
    raw_data = ImportedData()
    cm_beats = BeatAmplitudes()
    pace_maker = PacemakerData()
    upstroke_vel = UpstrokeVelData()
    local_act_time = LocalATData()
    conduction_vel = CondVelData()
    input_param = InputParameters()
    heat_map = MEAHeatMaps()
    cm_stats = StatisticsData()
    psd_data = PSDData()
    electrode_config = ElectrodeConfig(raw_data)

    # Heatmap axes for Calculate All (main window)
    heat_map.curr_plot = plt.Figure(figsize=(13, 6.5), dpi=120)
    heat_map.axis1 = heat_map.curr_plot.add_subplot(221)
    heat_map.axis2 = heat_map.curr_plot.add_subplot(222)
    heat_map.axis3 = heat_map.curr_plot.add_subplot(223)
    heat_map.axis4 = heat_map.curr_plot.add_subplot(224)

    # Heatmap axis for only Pacemaker window
    heat_map.pm_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.pm_solo_axis = heat_map.pm_solo_plot.add_subplot(111)

    # Heatmap axis for only Upstroke Velocity window
    heat_map.dvdt_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.dvdt_solo_axis = heat_map.dvdt_solo_plot.add_subplot(111)

    # Heatmap axis for only Local Activation Time window
    heat_map.lat_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.lat_solo_axis = heat_map.lat_solo_plot.add_subplot(111)

    # Heatmap axis for only Conduction Velocity window
    heat_map.cv_solo_plot = plt.Figure(figsize=(12, 6), dpi=120)
    heat_map.cv_solo_axis = heat_map.cv_solo_plot.add_subplot(111)

    # Subplot axes for Peak Finder (Beats) window
    cm_beats.comp_plot = plt.Figure(figsize=(10.5, 6), dpi=120)
    cm_beats.comp_plot.suptitle("Comparisons of find_peaks methodologies")
    cm_beats.axis1 = cm_beats.comp_plot.add_subplot(221)
    cm_beats.axis2 = cm_beats.comp_plot.add_subplot(222)
    cm_beats.axis3 = cm_beats.comp_plot.add_subplot(223)
    cm_beats.axis4 = cm_beats.comp_plot.add_subplot(224)

    # Subplot axes for Param vs Distance Stats window
    cm_stats.param_vs_dist_plot = plt.Figure(figsize=(10.5, 6.5), dpi=120)
    cm_stats.param_vs_dist_axis_pm = cm_stats.param_vs_dist_plot.add_subplot(221)
    cm_stats.param_vs_dist_axis_lat = cm_stats.param_vs_dist_plot.add_subplot(223)
    cm_stats.param_vs_dist_axis_dvdt = cm_stats.param_vs_dist_plot.add_subplot(222)
    cm_stats.param_vs_dist_axis_cv = cm_stats.param_vs_dist_plot.add_subplot(224)

    # Subplot axes for PSD Plot window
    psd_data.psd_plots = plt.Figure(figsize=(10.5, 6.5), dpi=120)
    psd_data.loglog_before_ax = psd_data.psd_plots.add_subplot(321)
    psd_data.loglog_during_ax = psd_data.psd_plots.add_subplot(323)
    psd_data.loglog_after_ax = psd_data.psd_plots.add_subplot(325)
    psd_data.psd_before_ax = psd_data.psd_plots.add_subplot(322)
    psd_data.psd_during_ax = psd_data.psd_plots.add_subplot(324)
    psd_data.psd_after_ax = psd_data.psd_plots.add_subplot(326)
    
    root = tk.Tk()
    # Dimensions width x height, distance position from right of screen + from 
    # top of screen
    # root.geometry("2700x1000+900+900")

    # Calls class to create the GUI window. *********
    analysisGUI = MainGUI(root, raw_data, cm_beats, pace_maker, upstroke_vel, 
        local_act_time, conduction_vel, input_param, heat_map, cm_stats, 
        electrode_config, psd_data)
    # print(vars(analysisGUI))
    # print(dir(analysisGUI))
    # print(hasattr(analysisGUI "elec_to_plot_entry"))
    root.mainloop()


main()
