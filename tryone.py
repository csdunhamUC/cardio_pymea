import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import seaborn as seab
import os
import time
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks


# data_two = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
# df = pd.DataFrame(data_two, index=['rank 1', 'rank 2', 'rank 3', 'rank 4'])
#
# print(df)

# Panel has been deprecated in pandas, so this operation must be done using dataframes.
# data_three = {'Item 1': pd.DataFrame(np.random.randn(4, 3)),
#               'Item 2': pd.DataFrame(np.random.randn(4, 2))}
# thing_one = pd.Panel()
#
# print(thing_one)

# data_three = pd.DataFrame(np.random.randn(5,3), index=['a', 'c', 'e', 'f', 'h'], columns=['one', 'two', 'three'])
#
# data_three = data_three.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
#
# print(data_three)
# prints true/false list for the given column if they satisfy the condition isnull = true (or isnull = 1)
# print(data_three['one'].isnull())

# Fills in NaN values with whatever you specify
# print("NaN replaced with 0.0123456")
# print(data_three.fillna(0.123456))
#
# print(data_three.fillna(method='pad'))

# Limited list of globals, purely out of necessity, for those functions (like the trace_add callback) for which it's
# impossible, at least at my current level of expertise, to get a damn value out of it.


# Classes that serve similar to Matlab structures (C "struct") to house data and allow it to be passed from
# one function to another.  Classes are generated for ImportedData (where the raw data will go), PaceMakerData
# (where PM data will go), UpstrokeVelData (where dV/dt data will go), LocalATData (where LAT data will go), and
# CondVelData, where CV data will go.
class ImportedData:
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


def data_import(raw_data):
    data_filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("txt files", "*.txt"), ("all files", "*.*")))

    start_time = time.process_time()

    # Checks whether data was previously imported into program.  If True, the previous data is deleted.
    if hasattr(raw_data, 'imported') is True:
        print("Raw data is not empty; clearing before reading file.")
        delattr(raw_data, 'imported')

    raw_data.imported = (pd.read_csv(data_filename,
                                     sep='\t', lineterminator='\n', skiprows=3, encoding='iso-8859-15',
                                     low_memory=False))

    new_data_size = np.shape(raw_data.imported)
    print(new_data_size)
    end_time = time.process_time()
    print(end_time - start_time)
    return raw_data.imported


def determine_beats(raw_data, cm_beats):
    print("Finding beats...\n")
    start_time = time.process_time()

    if hasattr(cm_beats, 'x_axis') is True:
        print("Beat data are not empty; clearing before finding peaks.")
        delattr(cm_beats, 'x_axis')
        delattr(cm_beats, 'dist_beats')
        delattr(cm_beats, 'prom_beats')
        delattr(cm_beats, 'width_beats')
        delattr(cm_beats, 'thresh_beats')

    cm_beats.x_axis = raw_data.imported.iloc[0:, 0]
    cm_beats.y_axis = raw_data.imported.iloc[0:, 1:]
    print("Y-axis data type is:: " + str(type(cm_beats.y_axis)) + "\n")
    print("Number of columns in cm_beats.y_axis: " + str(len(cm_beats.y_axis.columns)) + "\n")

    # cm_beats.dist_beats = {} ; if choice is made to return to dictionary, uncomment this and apply to for loop
    cm_beats.dist_beats = pd.DataFrame()
    cm_beats.prom_beats = pd.DataFrame()
    cm_beats.width_beats = pd.DataFrame()
    cm_beats.thresh_beats = pd.DataFrame()

    for column in range(len(cm_beats.y_axis.columns)):
        # cm_beats.dist_beats[column] = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, distance=1000)[0]
        dist_beats = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, distance=1000)[0]
        cm_beats.dist_beats.insert(column, column+1, dist_beats, allow_duplicates=True)

        # cm_beats.prom_beats = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, prominence=100)[0]
        prom_beats = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, distance=1000)[0]
        cm_beats.prom_beats.insert(column, column + 1, prom_beats, allow_duplicates=True)

        # cm_beats.width_beats = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, width=3)[0]
        width_beats = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, distance=1000)[0]
        cm_beats.width_beats.insert(column, column + 1, width_beats, allow_duplicates=True)

        # cm_beats.thresh_beats = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, threshold=50)[0]
        thresh_beats = find_peaks(cm_beats.y_axis.iloc[0:, column], height=100, distance=1000)[0]
        cm_beats.thresh_beats.insert(column, column + 1, thresh_beats, allow_duplicates=True)

    dist_beats_size = len(cm_beats.dist_beats)
    print("Shape of cm_beats.dist_beats: " + str(dist_beats_size))
    print(cm_beats.dist_beats)
    print("Data type of cm_beats.dist_beats: " + str(type(cm_beats.dist_beats)))
    #print(cm_beats.dist_beats[0])
    #print("Data type of cm_beats.dist_beats[0]: " + str(type(cm_beats.dist_beats)) + "\n")

    prom_beats_size = np.shape(cm_beats.prom_beats)
    print("Shape of cm_beats.prom_beats: " + str(prom_beats_size))
    print(cm_beats.prom_beats)
    print("Data type of cm_beats.prom_beats: " + str(type(cm_beats.prom_beats)) + "\n")

    width_beats_size = np.shape(cm_beats.width_beats)
    print("Shape of cm_beats.width_beats: " + str(width_beats_size) + ".\n")
    print(cm_beats.width_beats)
    print("Data type of cm_beats.width_beats: " + str(type(cm_beats.width_beats)) + "\n")

    thresh_beats_size = np.shape(cm_beats.thresh_beats)
    print("Shape of cm_beats.thresh_beats: " + str(thresh_beats_size) + ".\n")
    print(cm_beats.thresh_beats)
    print("Data type of cm_beats.thresh_beats: " + str(type(cm_beats.thresh_beats)) + "\n")

    # Backups, saving in case I screw this up royally.
    # cm_beats.dist_beats = find_peaks(cm_beats.y_axis.iloc[0:, 0], height=100, distance=1000)[0]
    # dist_beats_size = np.shape(cm_beats.dist_beats)
    # print("Shape of cm_beats.dist_beats: " + str(dist_beats_size) + ".\n")
    # print(cm_beats.dist_beats)
    # print(type(cm_beats.dist_beats))
    #
    # cm_beats.prom_beats = find_peaks(cm_beats.y_axis.iloc[0:, 0], height=100, prominence=100)[0]
    # prom_beats_size = np.shape(cm_beats.prom_beats)
    # print("Shape of cm_beats.prom_beats: " + str(prom_beats_size) + ".\n")
    # print(cm_beats.prom_beats)
    # print(type(cm_beats.prom_beats))
    #
    # cm_beats.width_beats = find_peaks(cm_beats.y_axis.iloc[0:, 0], height=100, width=3)[0]
    # width_beats_size = np.shape(cm_beats.width_beats)
    # print("Shape of cm_beats.width_beats: " + str(width_beats_size) + ".\n")
    # print(cm_beats.width_beats)
    # print(type(cm_beats.width_beats))
    #
    # cm_beats.thresh_beats = find_peaks(cm_beats.y_axis.iloc[0:, 0], height=100, threshold=50)[0]
    # thresh_beats_size = np.shape(cm_beats.thresh_beats)
    # print("Shape of cm_beats.thresh_beats: " + str(thresh_beats_size) + ".\n")
    # print(cm_beats.thresh_beats)
    # print(type(cm_beats.thresh_beats))

    print("Finished.")
    end_time = time.process_time()
    print(end_time - start_time)
    print("Plotting...")
    graph_peaks(cm_beats)


def graph_peaks(cm_beats):
    column_to_plot = 1

    cm_beats.axis1.plot(cm_beats.dist_beats.iloc[0:, column_to_plot].values,
                        cm_beats.y_axis.iloc[0:, column_to_plot].values[cm_beats.dist_beats.iloc[0:, column_to_plot].values], "xr")
    cm_beats.axis1.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, column_to_plot].values)
    cm_beats.axis1.legend(['distance = 1000'], loc='lower left')

    cm_beats.axis2.plot(cm_beats.prom_beats.iloc[0:, column_to_plot].values,
                        cm_beats.y_axis.iloc[0:, column_to_plot].values[cm_beats.prom_beats.iloc[0:, column_to_plot].values], "ob")
    cm_beats.axis2.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, column_to_plot].values)
    cm_beats.axis2.legend(['prominence = 100'], loc='lower left')

    cm_beats.axis3.plot(cm_beats.width_beats.iloc[0:, column_to_plot].values,
                        cm_beats.y_axis.iloc[0:, column_to_plot].values[cm_beats.width_beats.iloc[0:, column_to_plot].values], "vg")
    cm_beats.axis3.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, column_to_plot].values)
    cm_beats.axis3.legend(['width = 3'], loc='lower left')

    cm_beats.axis4.plot(cm_beats.thresh_beats.iloc[0:, column_to_plot],
                        cm_beats.y_axis.iloc[0:, column_to_plot].values[cm_beats.thresh_beats.iloc[0:, column_to_plot].values], "xk")
    cm_beats.axis4.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, column_to_plot].values)
    cm_beats.axis4.legend(['threshold = 50'], loc='lower left')

    # Backups, saving just in case.
    # cm_beats.axis1.plot(cm_beats.dist_beats, cm_beats.y_axis.iloc[0:, 0][cm_beats.dist_beats], "xr")
    # cm_beats.axis1.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, 0])
    # cm_beats.axis1.legend(['distance = 1000'], loc='lower left')
    #
    # cm_beats.axis2.plot(cm_beats.prom_beats, cm_beats.y_axis.iloc[0:, 0][cm_beats.prom_beats], "ob")
    # cm_beats.axis2.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, 0])
    # cm_beats.axis2.legend(['prominence = 100'], loc='lower left')
    #
    # cm_beats.axis3.plot(cm_beats.width_beats, cm_beats.y_axis.iloc[0:, 0][cm_beats.width_beats], "vg")
    # cm_beats.axis3.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, 0])
    # cm_beats.axis3.legend(['width = 3'], loc='lower left')
    #
    # cm_beats.axis4.plot(cm_beats.thresh_beats, cm_beats.y_axis.iloc[0:, 0][cm_beats.thresh_beats], "xk")
    # cm_beats.axis4.plot(cm_beats.x_axis, cm_beats.y_axis.iloc[0:, 0])
    # cm_beats.axis4.legend(['threshold = 50'], loc='lower left')

    cm_beats.comp_plot.canvas.draw()
    print("Plotting complete.")


def data_print(raw_data):
    # adding .iloc to a data frame allows to reference [row, column], where rows and columns can be ranges separated
    # by colons
    # print(id(raw_data.imported))
    print(chosen_electrode_val)
    # print(raw_data.imported.iloc[15:27, 0:15])


def graph_heatmap(vegetables, farmers, harvest, heat_map, axis1):
    # imshow() is the key heatmap function here.
    im = axis1.imshow(harvest, interpolation="nearest", aspect="auto", cmap="jet")

    cbar = axis1.figure.colorbar(im)
    cbar.ax.set_ylabel("Harvested Crops (t/year)", rotation=-90, va="bottom")

    axis1.set_xticks(np.arange(len(farmers)))
    axis1.set_yticks(np.arange(len(vegetables)))
    axis1.set_xticklabels(farmers)
    axis1.set_yticklabels(vegetables)
    plt.setp(axis1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = axis1.text(j, i, harvest[i, j], ha="center", va="center", color="w")
    axis1.set_title("Harvest Demo from Matplotlib Website")
    heat_map.tight_layout()
    heat_map.canvas.draw()
    return


class ElecGUI60(tk.Frame):
    def __init__(self, master, raw_data, cm_beats):
        tk.Frame.__init__(self, master)
        # The fun story about grid: columns and rows cannot be generated past the number of widgets you have (or at
        # least I've yet to learn the way to do so, and will update if I find out how).  It's all relative geometry,
        # where the relation is with other widgets.  If you have 3 widgets you can have 3 rows or 3 columns and
        # organize accordingly.  If you have 2 widgets you can only have 2 rows or 2 columns.
        self.grid()
        self.winfo_toplevel().title("Elec GUI 60 - Prototype")

        self.file_operations = tk.Frame(self, width=100, height=800, bg="white")
        self.file_operations.grid(row=1, column=0, padx=10, pady=10, sticky="nw")

        # use .bind("<Enter>", "color") or .bind("<Leave>", "color") to change mouse-over color effects.
        self.import_data_button = tk.Button(self.file_operations, text="Import Data", width=15, height=3, bg="skyblue",
                                            command=lambda: data_import(raw_data))
        self.import_data_button.grid(row=0, column=0, padx=2, pady=2)

        # to save data; not implemented.
        self.save_data_button = tk.Button(self.file_operations, text="Save Data", width=15, height=3, bg="lightgreen")
        self.save_data_button.grid(row=1, column=0, padx=2, pady=2)

        # prints data from import; eventually test to specify columns and rows.
        self.print_data_button = tk.Button(self.file_operations, text="Print Data", width=15, height=3, bg="yellow",
                                           command=lambda: data_print(raw_data))
        self.print_data_button.grid(row=2, column=0, padx=2, pady=2)

        self.calc_peaks_button = tk.Button(self.file_operations, text="Find Peaks", width=15, height=3, bg="orange",
                                           command=lambda: determine_beats(raw_data, cm_beats))
        self.calc_peaks_button.grid(row=3, column=0, padx=2, pady=2)

        # to generate heatmap; currently only generates for Matplotlib demo data.
        self.graph_heatmap_button = tk.Button(self.file_operations, text="Graph Heatmap", width=15, height=3, bg="red",
                                              command=lambda: graph_heatmap(vegetables, farmers, harvest, heat_map, axis1))
        self.graph_heatmap_button.grid(row=4, column=0, padx=2, pady=2)

        self.mea_parameters_frame = tk.Frame(self, width=2420, height=100, bg="white")
        self.mea_parameters_frame.grid(row=0, column=1, columnspan=3, padx=5, pady=5)
        self.mea_parameters_frame.grid_propagate(False)

        self.elec_to_plot_label = tk.Label(self.mea_parameters_frame, text="Electrode Plotted", bg="white")
        self.elec_to_plot_label.grid(row=0, column=0, padx=5, pady=5)
        self.elec_to_plot_val = tk.StringVar()
        self.elec_to_plot_val.trace_add("write", self.col_sel_callback)
        self.elec_to_plot_val.set("1")
        self.elec_to_plot_entry = tk.Entry(self.mea_parameters_frame, text=self.elec_to_plot_val)
        self.elec_to_plot_entry.grid(row=1, column=0, padx=5,pady=5)

        self.mea_array_frame = tk.Frame(self, width=1200, height=800, bg="white")
        self.mea_array_frame.grid(row=1, column=1, padx=10, pady=10)
        self.mea_array_frame.grid_propagate(False)

        self.beat_detect_frame = tk.Frame(self, width=1200, height=800, bg="white")
        self.beat_detect_frame.grid(row=1, column=2, padx=10, pady=10)
        self.beat_detect_frame.grid_propagate(False)

        self.gen_figure = FigureCanvasTkAgg(heat_map, self.mea_array_frame)
        self.gen_figure.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

        self.gen_beats = FigureCanvasTkAgg(cm_beats.comp_plot, self.beat_detect_frame)
        self.gen_beats.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

    def col_sel_callback(self, *args):
        print("You entered: \"{}\"".format(self.elec_to_plot_val.get()))
        global chosen_electrode_val
        try:
            chosen_electrode_val = int(self.elec_to_plot_val.get())
            print(type(chosen_electrode_val))
        except ValueError:
            print("Only numbers are allowed.  Please try again.")


def main():
    # Taken from matplotlib website for the sake of testing out a heatmap.  Still trying to figure out how to properly
    # integrate this into a GUI.

    global vegetables
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]

    global farmers
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    global harvest
    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    global heat_map
    heat_map = plt.Figure(figsize=(10, 6), dpi=120)
    global axis1
    axis1 = heat_map.add_subplot(111)

    raw_data = ImportedData()
    cm_beats = BeatAmplitudes()

    cm_beats.comp_plot = plt.Figure(figsize=(10, 7), dpi=120)
    cm_beats.comp_plot.suptitle("Comparisons of find_peaks methodologies")
    cm_beats.axis1 = cm_beats.comp_plot.add_subplot(221)
    cm_beats.axis2 = cm_beats.comp_plot.add_subplot(222)
    cm_beats.axis3 = cm_beats.comp_plot.add_subplot(223)
    cm_beats.axis4 = cm_beats.comp_plot.add_subplot(224)

    # import_raw_data = pd.DataFrame()
    # print("Import data memory reference: " + str(id(import_raw_data)))

    root = tk.Tk()
    # Dimensions width x height, distance position from right of screen + from top of screen
    root.geometry("2700x1200+900+900")

    # Calls class to create the GUI window. *********
    elecGUI60 = ElecGUI60(root, raw_data, cm_beats)
    # print(vars(elecGUI60))
    # print(dir(elecGUI60))
    root.mainloop()


main()
