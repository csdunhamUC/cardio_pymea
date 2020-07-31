import numpy as np
from matplotlib import *
import pandas as pd
import os
import time
# from tkinter import *
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
# Dimensions width x height, distance position from right of screen + from top of screen
root.geometry("1600x900+900+900")

# a = np.array([[1,2],[3,4]])
# print(a)
#
# b = np.array([1, 2, 3, 4, 5], ndmin=2)
# print(b)
#
# c = np.array([1, 2, 3], dtype=complex)
# print(c)
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

# Python functions don't take in variables in the same way... they take in some # of arguments, whatever you want them
# to be referred to as.  This differs from Matlab where everything needs to be declared outside of the function.
def data_import():
    data_filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
    new_data = pd.read_csv(data_filename,
                           sep='\t', lineterminator='\n', skiprows=3, encoding='iso-8859-15', low_memory=False)
    new_data_size = np.shape(new_data)
    time_to_run = time.process_time()
    print(new_data_size)
    print(time_to_run)
    # adding .iloc to a data frame allows to reference [row, column], where rows and columns can be ranges separated
    # by colons
    print(new_data.iloc[2:17, 15:25])
    return new_data


# mea_data = data_import()
# print(mea_data.iloc[2:17, 15:25])

class ElecGUI60(tk.Frame):
    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        # The fun story about grid: columns and rows cannot be generated past the number of widgets you have (or at
        # least I've yet to learn the way to do so, and will update if I find out how).  It's all relative geometry,
        # where the relation is with other widgets.  If you have 3 widgets you can have 3 rows or 3 columns and
        # organize accordingly.  If you have 2 widgets you can only have 2 rows or 2 columns.
        self.grid()
        self.file_management_widgets()
        self.mea_array_widgets()

    def file_management_widgets(self):
        self.winfo_toplevel().title("Elec GUI 60 - Prototype")
        import_data_frame = tk.Frame(self, width=100, height=800, bg="white")
        import_data_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        # use .bind("<Enter>", "color") or .bind("<Leave>", "color") to change mouse-over color effects.
        # A class can be set up for the sole purpose of handling these events.
        import_data_button = tk.Button(import_data_frame, text="Import Data", width=15, height=3, bg="skyblue",
                                       command=data_import)
        import_data_button.grid(row=0, column=0, padx=2, pady=2)
        save_data_button = tk.Button(import_data_frame, text="Save Data", width=15, height=3, bg="lightgreen")
        save_data_button.grid(row=1, column=0, padx=2, pady=2)

    def mea_array_widgets(self):
        mea_array_frame = tk.Frame(self, width=800, height=800, bg="white")
        mea_array_frame.grid(row=0, column=1, padx=10, pady=10)
        mea_array_frame.grid_propagate(False)
        tk.Label(mea_array_frame, text="Microelectrode Array Representation").grid(row=0, column=0, padx=240, pady=2,
                                                                                   sticky="e")
        tk.Label(mea_array_frame, text="MEA 1").grid(row=0, column=0, padx=5, pady=2, sticky="w")



# Calls class to create the GUI window. *********
elecGUI60 = ElecGUI60(root)
root.mainloop()
