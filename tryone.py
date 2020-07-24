import numpy as np
from matplotlib import *
import pandas as pd
import os
import time
from tkinter import *

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
def dataImport():
    new_data = pd.read_csv("/media/csdunham/0E24340D2433F675/Gimzewski Lab/Stem Cells/Data Sets/chip10002.txt",
                       sep='\t', lineterminator='\n', names=list(range(121)), skiprows=3,
                       encoding='iso-8859-15', low_memory=False)
    new_data_size = np.shape(new_data)
    titorun = time.process_time()
    print(new_data_size)
    print(titorun)
    return new_data

# adding .iloc to a data frame allows to reference [row, column], where rows and columns can be ranges separated
# by colons
# mea_data = dataImport()

#print(mea_data.iloc[2:17, 15:25])

class ElecGUI60(Frame):
    def __init__(self, parent=None):
        Frame.__init__(self, parent)
        self.parent = parent
        self.pack()
        self.make_widgets()

    def make_widgets(self):
        self.winfo_toplevel().title("Elec GUI 60 - Prototype")
        # label = Entry(self, font=('Times New Roman', 20))
        # label.pack(side="top", fill="x")
        # gui_size = Canvas(self)
        # gui_size.pack()


root = Tk()
# Dimensions width x height, distance position from right of screen + from top of screen
root.geometry("1600x900+900+900")
# root.title('Prototyping: 7/23/2020')

elecGUI60 = ElecGUI60(root)

# mea60_window = Canvas(root, width=1600, height=900)
# mea60_window.pack()
#
# mea60_title = Label(root, text='60 electrode GUI prototyping')
# mea60_title.config(font=('Times New Roman', 20))

# fig.title("GUI Demo 1")
# fig.geometry("400x400")

root.mainloop()
