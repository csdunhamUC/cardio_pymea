# Author: Christopher S. Dunham
# 11/1/2020
# Gimzewski Lab @ UCLA, Department of Chemistry & Biochem
# Original work
# This function is called following the use of "Calculate All" from 
# the drop-down menu and from the GUI slider on the main window.  It generates 
# the heat maps observed in the main window of the program.
# Graphing functions to produce heatmaps for individual parameters are located 
# within their respective calculation modules.
import seaborn as sns
from matplotlib import pyplot as plt

def graph_all(analysisGUI, heat_map, cm_beats, pace_maker, upstroke_vel, 
local_act_time, conduction_vel, input_param):
    # ----------------------------- Pacemaker ----------------------------------
    if hasattr(heat_map, 'cbar_1') is True:
        heat_map.cbar_1.remove()
        delattr(heat_map, 'cbar_1')

    analysisGUI.mainHeatmap.axis1.cla()
    input_param.beat_choice = analysisGUI.mainSlider.value()

    electrode_names = pace_maker.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table = pace_maker.param_dist_normalized.pivot(index='Y', 
        columns='X', values=pace_maker.final_dist_beat_count[
        input_param.beat_choice])

    temp = sns.heatmap(heatmap_pivot_table, cmap="jet", 
        annot=electrode_names, fmt="", ax=analysisGUI.mainHeatmap.axis1,
        vmin=0, vmax=pace_maker.param_dist_normalized_max, cbar=False)
    mappable = temp.get_children()[0]
    heat_map.cbar_1 = analysisGUI.mainHeatmap.axis1.figure.colorbar(mappable, 
        ax=analysisGUI.mainHeatmap.axis1)
    heat_map.cbar_1.ax.set_title("Time Lag (ms)", fontsize=10)

    analysisGUI.mainHeatmap.axis1.set(title="Pacemaker", 
        xlabel="X coordinate (μm)",
        ylabel="Y coordinate (μm)")

    # --------------------------- Upstroke velocity ----------------------------
    if hasattr(heat_map, 'cbar_2') is True:
        heat_map.cbar_2.remove()
        delattr(heat_map, 'cbar_2')
    analysisGUI.mainHeatmap.axis2.cla()

    electrode_names_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table_2 = upstroke_vel.param_dist_normalized.pivot(index='Y', 
        columns='X', values=upstroke_vel.final_dist_beat_count[
        input_param.beat_choice])

    temp_2 = sns.heatmap(heatmap_pivot_table_2, cmap="jet", 
        annot=electrode_names_2, fmt="", ax=analysisGUI.mainHeatmap.axis2,
        vmax=upstroke_vel.param_dist_normalized_max, cbar=False)
    mappable_2 = temp_2.get_children()[0]
    heat_map.cbar_2 = analysisGUI.mainHeatmap.axis3.figure.colorbar(mappable_2, 
        ax=analysisGUI.mainHeatmap.axis2)
    heat_map.cbar_2.ax.set_title("μV/(ms)", fontsize=10)

    analysisGUI.mainHeatmap.axis2.set(title="Upstroke Velocity", 
        xlabel="X coordinate (μm)", 
        ylabel="Y coordinate (μm)")

    # ------------------------- Local activation time --------------------------
    if hasattr(heat_map, 'cbar_3') is True:
        heat_map.cbar_3.remove()
        delattr(heat_map, 'cbar_3')
    analysisGUI.mainHeatmap.axis3.cla()

    electrode_names_3 = local_act_time.param_dist_normalized.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table_3 = local_act_time.param_dist_normalized.pivot(index='Y', 
        columns='X', values=local_act_time.final_dist_beat_count[
        input_param.beat_choice])

    temp_3 = sns.heatmap(heatmap_pivot_table_3, cmap="jet", 
        annot=electrode_names_3, fmt="", ax=analysisGUI.mainHeatmap.axis3,
        vmax=local_act_time.param_dist_normalized_max, cbar=False)
    mappable_3 = temp_3.get_children()[0]
    heat_map.cbar_3 = analysisGUI.mainHeatmap.axis3.figure.colorbar(mappable_3, 
        ax=analysisGUI.mainHeatmap.axis3)
    heat_map.cbar_3.ax.set_title("Time Lag (ms)", fontsize=10)

    analysisGUI.mainHeatmap.axis3.set(title="Local Activation Time", 
        xlabel="X coordinate (μm)", 
        ylabel="Y coordinate (μm)")

    # -------------------------- Conduction velocity ---------------------------
    if hasattr(heat_map, 'cbar_4') is True:
        heat_map.cbar_4.remove()
        delattr(heat_map, 'cbar_4')
    analysisGUI.mainHeatmap.axis4.cla()

    electrode_names_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
        columns='X', values='Electrode')
    heatmap_pivot_table_4 = conduction_vel.param_dist_raw.pivot(index='Y', 
        columns='X', values=local_act_time.final_dist_beat_count[
        input_param.beat_choice])

    temp_4 = sns.heatmap(heatmap_pivot_table_4, cmap="jet", 
        annot=electrode_names_4, fmt="", ax=analysisGUI.mainHeatmap.axis4, 
        cbar=False)
    mappable_4 = temp_4.get_children()[0]
    heat_map.cbar_4 = analysisGUI.mainHeatmap.axis4.figure.colorbar(mappable_4, 
        ax=analysisGUI.mainHeatmap.axis4)
    heat_map.cbar_4.ax.set_title("μm/(ms)", fontsize=10)

    analysisGUI.mainHeatmap.axis4.set(title="Conduction Velocity" , 
        xlabel="X coordinate (μm)", 
        ylabel="Y coordinate (μm)")

    analysisGUI.mainHeatmap.fig.tight_layout()
    analysisGUI.mainHeatmap.fig.subplots_adjust(top=0.9)
    analysisGUI.mainHeatmap.fig.suptitle("Parameter Heatmaps. Beat " + 
        str(input_param.beat_choice + 1) + " of " + 
        str(int(cm_beats.beat_count_dist_mode[0])) + ".")
    analysisGUI.mainHeatmap.draw()