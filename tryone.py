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


#
# data = np.array(['a', 'b', 'c', 'd'])
# s = pd.Series(data)
# pd.DataFrame(s)
#
# print(s)
#
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

# Classes that serve similar to Matlab structures (C "struct") to house data and allow it to be passed from
# one function to another.  Classes are generated for ImportedData (where the raw data will go), PaceMakerData
# (where PM data will go), UpstrokeVelData (where dV/dt data will go), LocalATData (where LAT data will go), and
# CondVelData, where CV data will go.
class ImportedData:
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

    print("Memory - raw_data, pre-fill: " + str(id(raw_data)))

    raw_data.imported = (pd.read_csv(data_filename,
                                     sep='\t', lineterminator='\n', skiprows=3, encoding='iso-8859-15',
                                     low_memory=False))

    new_data_size = np.shape(raw_data.imported)
    time_to_run = time.process_time()
    print("Memory - raw_data, post-fill: " + str(id(raw_data)))
    print("Memory - raw_data.imported, post-fill: " + str(id(raw_data.imported)))
    print(new_data_size)
    print(time_to_run)

    return raw_data.imported


# Original data_import function
# def data_import():
#     data_filename = filedialog.askopenfilename(initialdir="/", title="Select file",
#                                                filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
#     global raw_data
#     print(id(raw_data))
#     raw_data = (pd.read_csv(data_filename,
#                            sep='\t', lineterminator='\n', skiprows=3, encoding='iso-8859-15', low_memory=False))
#     new_data_size = np.shape(raw_data)
#     time_to_run = time.process_time()
#     print(new_data_size)
#     print(time_to_run)
#
#     print("In function, Raw data reference: " + str(id(raw_data)))
#
#     return raw_data


def data_print(raw_data):
    # adding .iloc to a data frame allows to reference [row, column], where rows and columns can be ranges separated
    # by colons
    print(id(raw_data.imported))
    print(raw_data.imported.iloc[15:27, 0:15])


# def data_print(raw_data):
#     # adding .iloc to a data frame allows to reference [row, column], where rows and columns can be ranges separated
#     # by colons
#     print(id(raw_data))
#     print(raw_data.iloc[15:27, 0:15])

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
    def __init__(self, master, raw_data):
        tk.Frame.__init__(self, master)
        # The fun story about grid: columns and rows cannot be generated past the number of widgets you have (or at
        # least I've yet to learn the way to do so, and will update if I find out how).  It's all relative geometry,
        # where the relation is with other widgets.  If you have 3 widgets you can have 3 rows or 3 columns and
        # organize accordingly.  If you have 2 widgets you can only have 2 rows or 2 columns.
        self.grid()
        self.file_management_widgets(raw_data)
        self.mea_array_widgets()

    def file_management_widgets(self, raw_data):
        self.winfo_toplevel().title("Elec GUI 60 - Prototype")
        file_operations = tk.Frame(self, width=100, height=800, bg="white")
        file_operations.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # use .bind("<Enter>", "color") or .bind("<Leave>", "color") to change mouse-over color effects.
        # A class can be set up for the sole purpose of handling these events.
        import_data_button = tk.Button(file_operations, text="Import Data", width=15, height=3, bg="skyblue",
                                       command=lambda: data_import(raw_data))
        import_data_button.grid(row=0, column=0, padx=2, pady=2)

        # to save data; not implemented.
        save_data_button = tk.Button(file_operations, text="Save Data", width=15, height=3, bg="lightgreen")
        save_data_button.grid(row=1, column=0, padx=2, pady=2)

        # prints data from import; eventually test to specify columns and rows.
        print_data_button = tk.Button(file_operations, text="Print Data", width=15, height=3, bg="yellow",
                                      command=lambda: data_print(raw_data))
        print_data_button.grid(row=2, column=0, padx=2, pady=2)

        # to generate heatmap; currently only generates for Matplotlib demo data.
        graph_heatmap_button = tk.Button(file_operations, text="Graph Heatmap", width=15, height=3, bg="red",
                                         command=lambda: graph_heatmap(vegetables, farmers, harvest, heat_map, axis1))
        graph_heatmap_button.grid(row=3, column=0, padx=2, pady=2)

        # Eventually this function will house the widgets necessary to display heat maps generated for MEAs.

    def mea_array_widgets(self):
        mea_array_frame = tk.Frame(self, width=1200, height=800, bg="white")
        mea_array_frame.grid(row=0, column=1, padx=10, pady=10)
        mea_array_frame.grid_propagate(False)

        gen_figure = FigureCanvasTkAgg(heat_map, mea_array_frame)
        gen_figure.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

        # tk.Label(mea_array_frame, text="Microelectrode Array Representation").grid(row=0, column=0, padx=240, pady=2,
        #                                                                            sticky="e")
        # tk.Label(mea_array_frame, text="MEA 1").grid(row=0, column=0, padx=5, pady=2, sticky="w")


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

    # import_raw_data = pd.DataFrame()
    # print("Import data memory reference: " + str(id(import_raw_data)))

    root = tk.Tk()
    # Dimensions width x height, distance position from right of screen + from top of screen
    root.geometry("1600x900+900+900")

    # Calls class to create the GUI window. *********
    elecGUI60 = ElecGUI60(root, raw_data)
    root.mainloop()


main()
