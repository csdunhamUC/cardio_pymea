# Author: Christopher S. Dunham
# Contact Email: csdunham@chem.ucla.edu, csdunham@protonmail.com
# Organization: University of California, Los Angeles
# Department of Chemistry & Biochemistry
# Laboratory PI: James K. Gimzewski
# This is an original work, unless otherwise noted in comments, by CSD.
# Began: 2/23/2021
# PyQt5 version of MEA analysis software.

import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QMainWindow, 
    QPushButton, QWidget, QDialog, QSlider, QComboBox, QProgressBar, QLineEdit, 
    QLabel, QFileDialog, QCheckBox)
from PyQt5.QtCore import QLine, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import os
import importlib
import datetime
import determine_beats
import calculate_pacemaker
import calculate_upstroke_vel
import calculate_lat
import calculate_cv
import param_vs_distance_stats
import psd_plotting
import cv_quiver
import calculate_beat_amp_int


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


# Import data files.  Files must be in .txt or .csv format.  May add toggles or 
# checks to support more data types.
def data_import(analysisGUI, raw_data, electrode_config):
    try:
        data_filename_and_path = QFileDialog.getOpenFileName(analysisGUI, "Select File",
            analysisGUI.file_path, "Text files (*.txt)")

        import_path, import_filename = os.path.split(data_filename_and_path[0])
        print(import_path)
        print(import_filename)

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
        raw_data.names = pd.read_csv(data_filename_and_path[0], sep="\s+\t", 
            lineterminator='\n', skiprows=[0, 1, 3], header=None, nrows=1, 
            encoding='iso-8859-15', skipinitialspace=True, engine='python')

        # # Import data from file.
        raw_data.imported = pd.read_csv(data_filename_and_path[0], sep='\s+', 
            lineterminator='\n', skiprows=3, header=0, encoding='iso-8859-15', 
            skipinitialspace=True, low_memory=False)

        print("Import data completed at: ", 
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Update file name display in GUI following import
        analysisGUI.fileName.setText(import_filename)
        analysisGUI.file_path = import_path

        raw_data.new_data_size = np.shape(raw_data.imported)
        print(raw_data.new_data_size[1])
        electrode_config.electrode_toggle(raw_data)
        print(analysisGUI.file_path)

    except FileNotFoundError:
        print()
    except TypeError:
        print()


def print_something():
    print("Something.")


def print_slider(analysisGUI):
    print(analysisGUI.mainSlider.value())
    print(analysisGUI.pkHeightEdit.text())


def trunc_toggle(analysisGUI):
    if analysisGUI.truncCheckbox.isChecked() == True:
        analysisGUI.truncStartEdit.show()
        analysisGUI.truncEndEdit.show()
    elif analysisGUI.truncCheckbox.isChecked() == False:
        analysisGUI.truncStartEdit.hide()
        analysisGUI.truncEndEdit.hide()

# Reloads given module.  This is used for testing/developing a module to save 
# time vs re-running the program over and over.
def reload_module():
    importlib.reload(param_vs_distance_stats)
    importlib.reload(calculate_cv)
    importlib.reload(determine_beats)
    # importlib.reload(calculate_lat)
    # importlib.reload(calculate_upstroke_vel)
    importlib.reload(psd_plotting)
    importlib.reload(cv_quiver)
    importlib.reload(calculate_beat_amp_int)
    print("Reloaded modules.")


# Classes for the plots of GUI.
class MainHeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axis1 = self.fig.add_subplot(221)
        self.axis2 = self.fig.add_subplot(222)
        self.axis3 = self.fig.add_subplot(223)
        self.axis4 = self.fig.add_subplot(224)
        super(MainHeatmapCanvas, self).__init__(self.fig)


class MinorHeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MinorHeatmapCanvas, self).__init__(self.fig)


class GenericPlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axis1 = self.fig.add_subplot(221)
        self.axis2 = self.fig.add_subplot(222)
        self.axis3 = self.fig.add_subplot(223)
        self.axis4 = self.fig.add_subplot(224)
        super(GenericPlotCanvas, self).__init__(self.fig)


# Classes for the actual GUI windows
class SoloHeatmapWindows(QWidget):
    def __init__(self):
        super(SoloHeatmapWindows, self).__init__()
        self.setupUI()

    def setupUI(self):
        layout = QGridLayout()
        self.paramPlot = MinorHeatmapCanvas(self, width=6, height=6, dpi=100)
        self.paramSlider = QSlider(Qt.Horizontal)
        paramToolbar = NavigationToolbar2QT(self.paramPlot, self)

        layout.addWidget(self.paramPlot, 0, 0)
        layout.addWidget(self.paramSlider, 1, 0)
        layout.addWidget(paramToolbar, 2, 0)

        self.setLayout(layout)


class GeneralPlotWindows(QWidget):
    def __init__(self):
        super(GeneralPlotWindows, self).__init__()
        self.setupUI()

    def setupUI(self):
        layout = QGridLayout()
        self.paramPlot = GenericPlotCanvas(self, width=7, height=7, dpi=100)
        self.paramSlider = QSlider(Qt.Horizontal)
        paramToolbar = NavigationToolbar2QT(self.paramPlot, self)

        layout.addWidget(self.paramPlot, 0, 0)
        layout.addWidget(self.paramSlider, 1, 0)
        layout.addWidget(paramToolbar, 2, 0)

        self.setLayout(layout)


class AnalysisGUI(QMainWindow):
    def __init__(self, x_var, y_var, raw_data, cm_beats, pace_maker, 
    upstroke_vel, local_act_time, conduction_vel, input_param, heat_map, 
    cm_stats, electrode_config, psd_data, beat_amp_int, parent=None):
        super().__init__(parent)
        # Function call to establish GUI widgets
        self.setup_UI(x_var, y_var, raw_data, cm_beats, pace_maker, 
            upstroke_vel, local_act_time, conduction_vel, input_param, heat_map, 
            cm_stats, electrode_config, psd_data, beat_amp_int,)
        # Initial file path
        self.file_path = "/"

    def setup_UI(self, x_var, y_var, raw_data, cm_beats, pace_maker, 
    upstroke_vel, local_act_time, conduction_vel, input_param, heat_map, 
    cm_stats, electrode_config, psd_data, beat_amp_int,):
        self.setWindowTitle("Analysis GUI - PyQt Version 0.1")
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        # Menu options.
        # File Menu
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction("&Import", lambda: data_import(self, 
            raw_data, electrode_config))
        self.fileMenu.addAction("&Save Processed Data")
        self.fileMenu.addAction("&Print (debug)", print_something)
        self.fileMenu.addAction("&Exit", self.close)
        
        # Calculation Menu
        self.calcMenu = self.menuBar().addMenu("&Calculations")
        self.calcMenu.addAction("&Find Beats (Use First!)", 
            lambda: [self.determineBeatsWindow(cm_beats, input_param, 
                electrode_config),
                determine_beats.determine_beats(self, raw_data, cm_beats, 
                input_param, electrode_config),
                determine_beats.graph_beats(self, cm_beats, input_param, 
                electrode_config)])
        self.calcMenu.addAction("&Calculate All (PM, LAT, dV/dt, CV, Amp, Int)")
        self.calcMenu.addAction("&Calculate Pacemaker", 
            lambda: [self.pacemakerWindow(cm_beats, pace_maker, heat_map, 
                input_param),
                calculate_pacemaker.calculate_pacemaker(self, cm_beats, 
                pace_maker, heat_map, input_param, electrode_config),
                calculate_pacemaker.graph_pacemaker(self, heat_map, pace_maker, 
                input_param)])
        self.calcMenu.addAction("&Calculate Local Act. Time")
        self.calcMenu.addAction("&Calculate Upstroke Velocity")
        self.calcMenu.addAction("&Calculate Conduction Velocity")

        # Plot Menu
        self.plotMenu = self.menuBar().addMenu("&Special Plots")
        self.plotMenu.addAction("&Cond. Vel. Vector Field")
        self.plotMenu.addAction("&Beat Amplitude & Interval")
        self.plotMenu.addAction("&Manual Electrode Filter")

        # Statistics Menu
        self.statMenu = self.menuBar().addMenu("&Statistics")
        self.statMenu.addAction("&Param vs Distance w/ R-value")
        self.statMenu.addAction("&Power Spectrum")

        # Tools Menu; To be filled later
        self.toolsMenu = self.menuBar().addMenu("&Tools")
        # Advanced Tools Menu (ML, etc); To be filled later
        self.advToolsMenu = self.menuBar().addMenu("&Advanced Tools")

        self.testingMenu = self.menuBar().addMenu("&Testing")
        self.testingMenu.addAction("&Reload modules (debug)", reload_module)

        # Arrange Widgets using row, col grid arrangement.
        mainLayout = QGridLayout()
        paramLayout = QGridLayout()
        plotLayout = QGridLayout()

        # Parameters, linked to paramLayout widget
        self.paramWidget = QWidget()
        self.plotWidget = QWidget()
        self.paramWidget.setLayout(paramLayout)
        self.plotWidget.setLayout(plotLayout)

        self.pkHeightLab = QLabel("Min Peak" + "\n" + "Height")
        self.pkHeightEdit = QLineEdit()
        self.pkHeightEdit.setFixedWidth(70)
        self.pkHeightEdit.setText("100")
        paramLayout.addWidget(self.pkHeightLab, 0, 0)
        paramLayout.addWidget(self.pkHeightEdit, 1, 0)
        self.pkDistLab = QLabel("Min Peak" + "\n" + "Distance")
        self.pkDistEdit = QLineEdit()
        self.pkDistEdit.setFixedWidth(70)
        self.pkDistEdit.setText("1000")
        paramLayout.addWidget(self.pkDistLab, 0, 1)
        paramLayout.addWidget(self.pkDistEdit, 1, 1)
        self.pkProm = QLabel("Peak" + "\n" + "Prominence")
        self.pkPromEdit = QLineEdit()
        self.pkPromEdit.setFixedWidth(70)
        self.pkPromEdit.setText("100")
        paramLayout.addWidget(self.pkProm, 0, 2)
        paramLayout.addWidget(self.pkPromEdit, 1, 2)
        self.pkWidth = QLabel("Peak" + "\n" + "Width")
        self.pkWidthEdit = QLineEdit()
        self.pkWidthEdit.setFixedWidth(70)
        self.pkWidthEdit.setText("3")
        paramLayout.addWidget(self.pkWidth, 0, 3)
        paramLayout.addWidget(self.pkWidthEdit, 1, 3)
        self.pkThresh = QLabel("Peak" + "\n" + "Threshold")
        self.pkThreshEdit = QLineEdit()
        self.pkThreshEdit.setFixedWidth(70)
        self.pkThreshEdit.setText("50")
        paramLayout.addWidget(self.pkThresh, 0, 4)
        paramLayout.addWidget(self.pkThreshEdit, 1, 4)
        self.sampleFreq = QLabel("Sample" + "\n" + "Frequency (Hz)")
        self.sampleFreqEdit = QComboBox()
        self.sampleFreqEdit.setFixedWidth(100)
        self.sampleFreqEdit.addItem("1000")
        self.sampleFreqEdit.addItem("10000")
        paramLayout.addWidget(self.sampleFreq, 0, 5)
        paramLayout.addWidget(self.sampleFreqEdit, 1, 5)
        # Truncation widgets.
        self.truncCheckbox = QCheckBox("Truncate Data")
        self.truncCheckbox.clicked.connect(lambda: trunc_toggle(self))
        self.truncStartEdit = QLineEdit()
        self.truncStartEdit.setFixedWidth(55)
        self.truncEndEdit = QLineEdit()
        self.truncEndEdit.setFixedWidth(55)
        paramLayout.addWidget(self.truncCheckbox, 0, 6, 1, 2)
        paramLayout.addWidget(self.truncStartEdit, 1, 6)
        paramLayout.addWidget(self.truncEndEdit, 1, 7)
        self.truncStartEdit.setVisible(False)
        self.truncEndEdit.setVisible(False)
        # File name label.
        self.fileName = QLabel("Waiting for file.")
        paramLayout.addWidget(self.fileName, 0, 8)
        
        # Plots, linked to plotLayout widget
        self.mainHeatmap = MainHeatmapCanvas(self, width=10, height=8, dpi=100)
        # (row, column, rowspan, colspan)
        plotLayout.addWidget(self.mainHeatmap, 2, 0, 1, 2)
        self.mainHeatmap.axis1.plot(x_var, y_var)

        self.mainSlider = QSlider(Qt.Horizontal)
        plotLayout.addWidget(self.mainSlider, 3, 0)
        self.mainSlider.valueChanged.connect(lambda: print_slider(self))

        mainToolbar = NavigationToolbar2QT(self.mainHeatmap, self)
        plotLayout.addWidget(mainToolbar, 4, 0)

        # Add parameter and plot widgets to the main GUI layout, then display.
        mainLayout.addWidget(self.paramWidget, 0, 0)
        mainLayout.addWidget(self.plotWidget, 1, 0)
        self.mainWidget.setLayout(mainLayout)
    
    def determineBeatsWindow(self, cm_beats, input_param, electrode_config):
        self.beatsWindow = GeneralPlotWindows()
        self.beatsWindow.setWindowTitle("Beat Finder Results")
        # self.beatsWindow.paramPlot.axis1.plot([1,2,3,4,5],[10,20,30,40,50])
        self.beatsWindow.show()
        self.beatsWindow.paramSlider.valueChanged.connect(lambda: [
            determine_beats.graph_beats(self, cm_beats, input_param, 
            electrode_config)])

    def pacemakerWindow(self, cm_beats, pace_maker, heat_map, input_param):
        self.pmWindow = SoloHeatmapWindows()
        self.pmWindow.setWindowTitle("Pacemaker Results")
        self.pmWindow.show()
        self.pmWindow.paramSlider.valueChanged.connect(lambda: [
            calculate_pacemaker.graph_pacemaker(self, heat_map, pace_maker, 
            input_param)])

    def upVelocityWindow(self, cm_beats, upstroke_vel, heat_map, input_param):
        self.dvdtWindow = SoloHeatmapWindows()
        self.dvdtWindow.setWindowTitle("Upstroke Velocity (dV/dt) Results")
        self.dvdtWindow.show()

    def localActTimeWindow(self, cm_beats, local_act_time, heat_map, 
    input_param):
        self.latWindow = SoloHeatmapWindows()
        self.latWindow.setWindowTitle("Local Activation Time (LAT) Results")
        self.latWindow.show()

    def condVelocityWindow(self, cm_beats, local_act_time, conduction_vel, 
    heat_map, input_param):
        self.cvWindow = SoloHeatmapWindows()
        self.cvWindow.setWindowTitle("Conduction Velocity (CV) Results")
        self.cvWindow.show()

    def condVelVectorWindow(self, cm_beats, local_act_time, conduction_vel, 
    input_param):
        self.cvVectWindow = GeneralPlotWindows()
        self.cvVectWindow.setWindowTitle("Conduction Velocity Vector Field")
        self.cvVectWindow.show()

    # This probably needs a new class for its window, as there's a lot of info
    # to display that the other windows don't need.
    def paramVsDistStatsWindow(self, cm_beats, pace_maker, upstroke_vel, 
    local_act_time, conduction_vel, input_param, cm_stats):
        self.pvdWindow = GeneralPlotWindows()
        self.pvdWindow.setWindowTitle("Parameter vs Distance w/ R-Square")
        self.pvdWindow.show()

    def psdPlotWindow(self, cm_beats, electrode_config, pace_maker, 
    upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats, 
    psd_data):
        self.psdWindow = GeneralPlotWindows()
        self.psdWindow.setWindowTitle("Power Spectra")
        self.psdWindow.show()

    def beatAmpIntWindow(self, cm_beats, pace_maker, local_act_time,
    beat_amp_int, input_param, electrode_config):
        self.ampIntWindow = GeneralPlotWindows()
        self.ampIntWindow.setWindowTitle("Beat Amplitude & Interval")
        self.ampIntWindow.show()


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
    beat_amp_int = BeatAmpIntData()
    
    app = QApplication([])
    app.setStyle("Fusion")

    x_var = [0,1,2,3,4,8]
    y_var = [10,20,30,40,50,60]

    analysisGUI = AnalysisGUI(x_var, y_var, raw_data, cm_beats, pace_maker, 
        upstroke_vel, local_act_time, conduction_vel, input_param, heat_map, 
        cm_stats, electrode_config, psd_data, beat_amp_int,)
    # analysisGUI.resize(1200, 800)
    analysisGUI.show()
    sys.exit(app.exec_())

main()

############################## Dump for later. #################################
    # self.plotTwo = MinorHeatmapCanvas(self, width=6, height=6, dpi=100)
    # plotLayout.addWidget(self.plotTwo, 4, 0, 1, 2)
    # self.plotTwo.axes.plot(x_var, y_var)