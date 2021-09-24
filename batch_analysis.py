# Author:
# Christopher S. Dunham
# Email:
# csdunham@chem.ucla.edu; csdunham@protonmail.com
# Principal Investigator:
# James K. Gimzewski
# Organization:
# University of California, Los Angeles
# Department of Chemistry and Biochemistry
# This is an original work, unless other noted in comments, by CSD
# Began 9/23/21

# Batch analysis software.
# Requires the use of a batch_params.xlsx file (see example from repo)
# Program will load the *.xlsx file, identify the save file path of the batch 
# files, the individual files, the parameters for each file, etc, and perform 
# batch calculations.

import sys
import pandas as pd
import determine_beats
import calculate_pacemaker
import calculate_upstroke_vel
import calculate_lat
import calculate_cv
import calculate_beat_amp_int
import detect_transloc


class BatchData:
    pass


def import_batch_file():
    # Load batch file.
    batch_dir = "~/Documents/Python_Learning"
    batch_name = "my_batch.xlsx"
    batch_file = batch_dir + "/" + batch_name
    batch_df = pd.read_excel(batch_file)
    print(batch_df)
    # print(batch_df.columns)
    # print(batch_df["toggle_silence"])

    for (file_dir, file_name) in zip(batch_df["file_dir"][0:1], 
    batch_df["file_name"][0:1]):
        file_path = "/".join([file_dir, file_name])
        print(file_path)
        temp_data = pd.read_csv(
            file_path, sep="\s+", lineterminator="\n", skiprows=3,header=0, 
            encoding='iso-8859-15', skipinitialspace=True, low_memory=False)
        print(temp_data)
        print(temp_data[temp_data.columns[:-1]])
        # temp_data = pd.read_csv(
            # file_path, sep="\s+\t", lineterminator="\n", skiprows=[0, 1, 3],
            # header=None, nrows=1, encoding='iso-8859-15', 
            # skipinitialspace=True)

import_batch_file()
