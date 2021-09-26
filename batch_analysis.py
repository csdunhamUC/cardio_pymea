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
import os
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QWidget
import pandas as pd
import numpy as np
import determine_beats
import calculate_pacemaker
import calculate_upstroke_vel
import calculate_lat
import calculate_cv
import calculate_beat_amp_int
import detect_transloc


class ImportedData:
    pass


class BatchData:
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


class StatisticsData:
    pass


class BeatAmpIntData:
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
        # If true, use 120 electrode config.  If false, use 60 electrode config.
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


def import_batch_file(analysisGUI, raw_data, batch_data, electrode_config, 
cm_beats, input_param, pace_maker, heat_map, local_act_time, upstroke_vel,
conduction_vel, beat_amp_int):
    # Set batch_config status to true
    batch_data.batch_config = True
    # Open dialog to select file (*.xlsx)
    batch_filename_and_path = QFileDialog.getOpenFileName(
        analysisGUI, "Select Batch File", "/home", "Excel files (*.xlsx)")
    # Get directory, file name
    batch_dir, batch_name = os.path.split(batch_filename_and_path[0])
    # Load data from file (*.xlsx) into dataframe
    batch_df = pd.read_excel(batch_filename_and_path[0])

    print(f"Processing batch: {batch_name}")

    batch_data.batch_translocs = []

    total_files = len(batch_df["file_name"].values)

    for num, (file_dir, file_name, pk_height, pk_dist, samp_freq, tog_trunc, 
    trunc_start, trunc_end, tog_silence, silenced_elecs) in enumerate(zip(
    batch_df["file_dir"][0:4], batch_df["file_name"][0:4], 
    batch_df["min_pk_height"][0:4], batch_df["min_pk_dist"][0:4], 
    batch_df["sample_frequency"][0:4], batch_df["toggle_trunc"][0:4], 
    batch_df["trunc_start"][0:4], batch_df["trunc_end"][0:4], 
    batch_df["toggle_silence"][0:4], batch_df["silenced_electrodes"][0:4])):
        print("")
        print(f"Analyzing file {num+1} of {total_files}: {file_name}.")
        print("")
        file_path = "/".join([file_dir, file_name])
        raw_data.imported = pd.read_csv(
            file_path, sep="\s+", lineterminator="\n", skiprows=3,header=0, 
            encoding='iso-8859-15', skipinitialspace=True, low_memory=False)
        # temp_data = pd.read_csv(
            # file_path, sep="\s+\t", lineterminator="\n", skiprows=[0, 1, 3],
            # header=None, nrows=1, encoding='iso-8859-15', 
            # skipinitialspace=True)
       
        print(f"Silence toggled?: {tog_silence}")

        raw_data.new_data_size = np.shape(raw_data.imported)
        electrode_config.electrode_toggle(raw_data)
        input_param.min_peak_dist = pk_dist
        input_param.min_peak_height = pk_height
        input_param.parameter_prominence = 100 # Defaults from GUI
        input_param.parameter_width = 3 # Defaults from GUI
        input_param.parameter_thresh = 50 # Defaults from GUI
        input_param.sample_frequency = samp_freq
        # # Perform batch calculations
        determine_beats.determine_beats(analysisGUI, raw_data, cm_beats, 
            input_param, electrode_config, batch_data)
        calculate_pacemaker.calculate_pacemaker(analysisGUI, cm_beats, 
            pace_maker, heat_map, input_param, electrode_config)
        calculate_lat.calculate_lat(analysisGUI, cm_beats, local_act_time,
            heat_map, input_param, electrode_config)
        calculate_upstroke_vel.calculate_upstroke_vel(analysisGUI, cm_beats, 
            upstroke_vel, heat_map, input_param, electrode_config)
        calculate_cv.calculate_conduction_velocity(analysisGUI, cm_beats, 
            conduction_vel, local_act_time, heat_map, input_param, 
            electrode_config)
        calculate_beat_amp_int.calculate_beat_amp(analysisGUI, cm_beats, 
            beat_amp_int, pace_maker, local_act_time, heat_map, input_param, 
            electrode_config)
        detect_transloc.pm_translocations(analysisGUI, pace_maker, electrode_config)
    
        temp_translocs = pace_maker.transloc_events
        batch_data.batch_translocs.append(temp_translocs)

    # Print number of translocations found in files contained in batch
    print("Translocations in batch: \n" + f"{batch_data.batch_translocs}")
    # Batch processing end.
    print("Batch processing complete.")
    # Reset batch_config flag to False to allow for single-file analysis
    batch_data.batch_config = False


class MainGUI(QMainWindow):
    def __init__(self, batch_data, electrode_config, raw_data,
    cm_beats, input_param, parent=None):
        super().__init__(parent)
        self.setupUI(batch_data, electrode_config, raw_data, cm_beats, 
            input_param)

    def setupUI(self, batch_data, electrode_config, raw_data, cm_beats, 
    input_param):
        self.setWindowTitle("Main GUI")
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        self.fileMenu = self.menuBar().addMenu("Test")
        self.fileMenu.addAction("Batch Import", 
            lambda: [import_batch_file(self, batch_data, electrode_config, 
                raw_data, cm_beats, input_param)])

def main():
    raw_data = ImportedData()
    batch_data = BatchData()
    cm_beats = BeatAmplitudes()
    pace_maker = PacemakerData()
    upstroke_vel = UpstrokeVelData()
    local_act_time = LocalATData()
    conduction_vel = CondVelData()
    input_param = InputParameters()
    cm_stats = StatisticsData()
    electrode_config = ElectrodeConfig(batch_data)
    beat_amp_int = BeatAmpIntData() 

    app = QApplication([])
    app.setStyle("Fusion")

    my_GUI = MainGUI(batch_data, electrode_config, raw_data, cm_beats,
        input_param)
    my_GUI.show()
    sys.exit(app.exec_())

# main()
