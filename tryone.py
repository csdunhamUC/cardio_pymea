import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# # Compute the x and y coordinates for points on a sine curve
# x = np.arange(0, 3 * np.pi, 0.1)
# y = np.sin(x)
# plt.title("sine wave form")
#
# # Plot the points using matplotlib
# plt.plot(x, y)
# plt.show()

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

new_data = pd.read_csv("/media/csdunham/0E24340D2433F675/Gimzewski Lab/Stem Cells/Data Sets/chip10002.txt",
                       sep='\t', lineterminator='\n', names=list(range(120)))

print(new_data[0:5])
