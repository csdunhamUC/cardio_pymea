# Cardio PyMEA: A Cardiomyocyte MEA Analaysis Application in Python.
# Author: Christopher S. Dunham
# Contact Email: csdunham@chem.ucla.edu, csdunham@protonmail.com
# Organization: University of California, Los Angeles
# Department of Chemistry & Biochemistry
# Laboratory PI: James K. Gimzewski
# PI email: gimzewski@cnsi.ucla.edu
# This is an original work, unless otherwise noted in comments, by CSD.
# Began: 2/23/2021
# Version 1.0 completed: 3/23/2022

import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QMainWindow, 
    QPushButton, QWidget, QDialog, QSlider, QComboBox, QProgressBar, QLineEdit, 
    QLabel, QFileDialog, QCheckBox, QPlainTextEdit, QStyledItemDelegate, qApp)
from PyQt5.QtCore import QLine, Qt, QEvent
from PyQt5.QtGui import QFont, QPalette, QFontMetrics, QStandardItem
from matplotlib.backends.backend_qt5 import SaveFigureQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import os
import importlib
import datetime
import determine_beats
import calculate_pacemaker
import calculate_upstroke_vel
import calculate_lat
import calculate_cv
import main_heatmap
import property_vs_distance_stats
import psd_plotting
import calculate_beat_amp_int
import pca_plotting
import detect_transloc
import batch_analysis
import powerlaw_analysis
import calculate_fpd


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


class FieldPotDurationData:
    pass


class MEAHeatMaps:
    pass


class StatisticsData:
    pass


class PSDData:
    pass


class BeatAmpIntData:
    pass


class BatchData:
    pass


# Custom widget class for checkable, multi-select combo box.
# From Stack Exchange: 
# https://gis.stackexchange.com/questions/350148/qcombobox-multiple-selection-pyqt5
class CheckableComboBox(QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, 
            self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res


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
# checks to support more data types, particularly if NOT using MEAs from
# Multichannel Systems or not using MC_Data converted MC_Rack *.mcd file
def data_import(analysisGUI, raw_data, electrode_config):
    try:
        data_filename_and_path = QFileDialog.getOpenFileName(analysisGUI, 
            "Select File", analysisGUI.file_path, "Text files (*.txt)")

        import_path, import_filename = os.path.split(data_filename_and_path[0])

        # Checks whether data was previously imported into program.  If True, 
        # the previous data is deleted.
        if hasattr(raw_data, 'imported') is True:
            print("Raw data is not empty; clearing before reading file.")
            delattr(raw_data, 'imported')

        print("Import data began at: ", 
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Import data from file.
        raw_data.imported = pd.read_csv(data_filename_and_path[0], sep='\s+', 
            lineterminator='\n', skiprows=3, header=0, encoding='iso-8859-15', 
            skipinitialspace=True, low_memory=False)

        # Update file name display in GUI following import
        analysisGUI.fileName.setText(f"File: {import_filename}")
        analysisGUI.file_path = import_path

        print("Import data completed at: ", 
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        raw_data.new_data_size = np.shape(raw_data.imported)
        print(raw_data.new_data_size[1])
        electrode_config.electrode_toggle(raw_data)
        analysisGUI.elecCombobox.clear()
        analysisGUI.elecCombobox.addItems(electrode_config.electrode_names)
        analysisGUI.mainHeatmap.axis1.cla()
        analysisGUI.mainHeatmap.axis2.cla()
        analysisGUI.mainHeatmap.axis3.cla()
        analysisGUI.mainHeatmap.axis4.cla()
        analysisGUI.mainHeatmap.draw()

    except FileNotFoundError:
        print("Operation cancelled.")
    except TypeError:
        print()


# Save DataFrame in Excel Spreadsheet 
# written by Madelynn E. Mackenize (MEM), undergrad, UCLA class of 2022
# create different sheet for each calculated parameter: 
# PM, LAT, dv/dt, CV, Amp, Int, Stats
# dataframe of PM data: pace_maker.param_dist_normalized
# Modifications to original code by CSD.
def export_excel(analysisGUI, pace_maker, local_act_time, upstroke_vel, 
conduction_vel, beat_amp_int, cm_beats, cm_stats, field_potential):
    try:
        print("Saving processed data...")

        # For save file dialog box to enter custom name and file save location.
        saveFile = QFileDialog.getSaveFileName(analysisGUI, "Save File", 
            analysisGUI.file_path, "Excel file (*.xlsx)")
        file_path, file_name = os.path.split(saveFile[0])

        with pd.ExcelWriter('{path}/{name}.xlsx'.format(
        path=file_path, name=file_name)) as writer:
            pace_maker.param_dist_normalized.to_excel(writer, 
                sheet_name='Pacemaker (Normalized)')
            pace_maker.param_dist_raw.to_excel(writer, 
                sheet_name='Pacemaker (Raw)')
            pace_maker.param_dist_normalized_per_beat_max.to_excel(writer, 
                sheet_name="Per Beat Max Time Lag")
            # pace_maker.transloc_events.to_excel(writer,
            #     sheet_name="Translocation Events", index=False)
            local_act_time.param_dist_normalized.to_excel(writer, 
                sheet_name='LAT Normalized')
            local_act_time.distance_from_min.to_excel(writer, 
                sheet_name='LAT Distance')
            upstroke_vel.param_dist_normalized.to_excel(writer, 
                sheet_name='Upstroke Velocity')
            conduction_vel.param_dist_raw.to_excel(writer, 
                sheet_name='Conduction Velocity')
            conduction_vel.vector_mag.to_excel(writer, 
                sheet_name='CV Vector Magnitude')
            conduction_vel.vector_x_comp.to_excel(writer, 
                sheet_name='CV Vector X Comp')
            conduction_vel.vector_y_comp.to_excel(writer, 
                sheet_name='CV Vector Y Comp')
            beat_amp_int.beat_amp.to_excel(writer, 
                sheet_name='Beat Amplitude') 
            beat_amp_int.delta_beat_amp.to_excel(writer, 
                sheet_name='Delta Beat Amplitude')
            beat_amp_int.beat_interval.to_excel(writer, 
                sheet_name='Beat Interval')
            field_potential.FPD.to_excel(writer,
                sheet_name='Field Potential Duration')
            cm_stats.pace_maker_filtered_data.to_excel(writer, 
                sheet_name='PM Stats')
            cm_stats.local_act_time_filtered_data.to_excel(writer, 
                sheet_name='LAT Stats')
            cm_stats.upstroke_vel_filtered_data.to_excel(writer, 
                sheet_name='Upstroke Velocity Stats')
            cm_stats.conduction_vel_filtered_data.to_excel(writer, 
                sheet_name='CV Stats')
        
        print("Data saved in path: {}".format(file_path))
    except AttributeError:
        print("Parameters missing. Please be sure to complete calculations " + 
            "(e.g. statistics)")
    except ValueError:
        print("Operation cancelled.")
    except FileNotFoundError:
        print("No such file or directory.")


# Function used to show or hide truncation range, based on whether
# truncation check box is checked.
def trunc_toggle(analysisGUI):
    if analysisGUI.truncCheckbox.isChecked() == True:
        analysisGUI.truncStartEdit.show()
        analysisGUI.truncEndEdit.show()
    elif analysisGUI.truncCheckbox.isChecked() == False:
        analysisGUI.truncStartEdit.hide()
        analysisGUI.truncEndEdit.hide()


# Function used to show or hide electrode list for silencing, based on whether
# silence electrodes check box is checked.
def silence_toggle(analysisGUI):
    if analysisGUI.toggleSilence.isChecked() == True:
        analysisGUI.elecCombobox.show()
    elif analysisGUI.toggleSilence.isChecked() == False:
        analysisGUI.elecCombobox.hide()


# Function used to show or hide entry field for number of bins in power law
# histogram, based on whether "Manual Entry" mode is chosen.
def nbins_toggle(analysisGUI):
    if analysisGUI.plWindow.binMethodSelect.currentText() == "Manual Entry":
        analysisGUI.plWindow.nbinsLabel.show()
        analysisGUI.plWindow.nbinsEdit.show()
    else:
        analysisGUI.plWindow.nbinsLabel.hide()
        analysisGUI.plWindow.nbinsEdit.hide()


# Reloads given module.  This is used for testing/developing a module to save 
# time vs re-running the program over and over.
def reload_module():
    importlib.reload(property_vs_distance_stats)
    importlib.reload(calculate_cv)
    importlib.reload(determine_beats)
    # importlib.reload(calculate_lat)
    # importlib.reload(calculate_upstroke_vel)
    importlib.reload(psd_plotting)
    importlib.reload(calculate_pacemaker)
    importlib.reload(calculate_beat_amp_int)
    importlib.reload(pca_plotting)
    importlib.reload(detect_transloc)
    importlib.reload(calculate_fpd)
    importlib.reload(powerlaw_analysis)
    print("Reloaded modules.")


###############################################################################
# FigureCanvas classes for the plots (figs and axes) used in GUI windows.
###############################################################################

# Canvas class for the heatmap shown upon starting Cardio PyMEA.
class MainHeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi, 
            tight_layout=True)
        self.axis1 = self.fig.add_subplot(221)
        self.axis2 = self.fig.add_subplot(222)
        self.axis3 = self.fig.add_subplot(223)
        self.axis4 = self.fig.add_subplot(224)
        super(MainHeatmapCanvas, self).__init__(self.fig)


# Canvas class for individual (single-plot) heatmaps.
# Used in pacemaker, local activation time, upstroke velocity, and conduction
# velocity functions.
class MinorHeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axis1 = self.fig.add_subplot(111)
        super(MinorHeatmapCanvas, self).__init__(self.fig)


# 2x2 plot used in "Find Beats" window.
class GenericPlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axis1 = self.fig.add_subplot(221)
        self.axis2 = self.fig.add_subplot(222)
        self.axis3 = self.fig.add_subplot(223)
        self.axis4 = self.fig.add_subplot(224)
        super(GenericPlotCanvas, self).__init__(self.fig)


# 1x2 plot used in power spectral density window.
class PSDPlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=7, height=7, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axis1 = self.fig.add_subplot(211)
        self.axis2 = self.fig.add_subplot(212)
        super(PSDPlotCanvas, self).__init__(self.fig)


# 1x3 plot used in power law analysis window.
class PowerlawCanvas (FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=5, dpi=100):
        self.fig=plt.Figure(figsize=(width, height), dpi=dpi)
        self.axis1 = self.fig.add_subplot(1,3,1)
        self.axis2 = self.fig.add_subplot(1,3,2)
        self.axis3 = self.fig.add_subplot(1,3,3)
        super(PowerlawCanvas, self).__init__(self.fig)


###############################################################################
# Classes for the GUI windows
###############################################################################

# Window for individual plots (e.g. heat map).
# This class is used for the individual property plots, including:
# pacemaker, local activation time, upstroke velocity, and conduction velocity.
class SoloHeatmapWindows(QWidget):
    def __init__(self, analysisGUI):
        super(SoloHeatmapWindows, self).__init__()
        self.setupUI(analysisGUI)

    def setupUI(self, analysisGUI):
        layout = QGridLayout()
        self.paramPlot = MinorHeatmapCanvas(self, width=9, height=7, dpi=100)
        self.paramSlider = QSlider(Qt.Horizontal)
        paramToolbar = NavigationToolbar2QT(self.paramPlot, self)

        layout.addWidget(self.paramPlot, 0, 0)
        layout.addWidget(self.paramSlider, 1, 0)
        layout.addWidget(paramToolbar, 2, 0)

        if (hasattr(analysisGUI, "pcaCheck") is True 
        and analysisGUI.pcaCheck is True):
            self.plotButton = QPushButton("Plot")
            self.plotButton.setFixedSize(70, 70)
            layout.addWidget(self.plotButton, 1, 0)
    
        self.setLayout(layout)


# Generates the window used for "Find Beats".
# Window contains two plots.
# Left plot shows voltage trace for the specified electrode, beat.
# Right plot shows the full MEA voltage trace for the specified beat.
class BeatSignalPlotWindow(QWidget):
    def __init__(self):
        super(BeatSignalPlotWindow, self).__init__()
        self.setupUI()

    def setupUI(self):
        # Establish grid layout.
        mainLayout = QGridLayout()
        paramLayout = QGridLayout()
        buttonLayout = QGridLayout()
        signalLayout = QGridLayout()
        plotLayout = QGridLayout()

        # Set up container widgets
        buttonWidget = QWidget()
        buttonWidget.setLayout(buttonLayout)
        buttonWidget.setFixedWidth(100)
        paramWidget = QWidget()
        paramWidget.setLayout(paramLayout)
        paramWidget.setFixedWidth(500)
        plotWidget = QWidget()
        plotWidget.setLayout(plotLayout)

        # Plot button
        self.plotButton = QPushButton("Find\nBeats")
        self.plotButton.setFixedSize(70, 70)
        buttonLayout.addWidget(self.plotButton, 0, 0)

        # Filter selection widgets.
        self.filterType = QLabel("Signal Filter\n(Butterworth)")
        self.filterTypeEdit = QComboBox()
        self.filterTypeEdit.setFixedWidth(100)
        self.filterTypeEdit.addItem("No filter")
        self.filterTypeEdit.addItem("Low-pass Only")
        self.filterTypeEdit.addItem("High-pass Only")
        self.filterTypeEdit.addItem("Bandpass")
        self.filterTypeEdit.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')

        # Filter parameters widgets.
        self.butterOrderLabel = QLabel("Butterworth\nOrder")
        self.butterOrderEdit = QLineEdit()
        self.butterOrderEdit.setText("4")
        self.butterOrderEdit.setFixedWidth(70)
        self.lowPassFreqLabel = QLabel("Low-pass\nFrequency (Hz)")
        self.lowPassFreqEdit = QLineEdit()
        self.lowPassFreqEdit.setText("30")
        self.lowPassFreqEdit.setFixedWidth(70)
        self.highPassFreqLabel = QLabel("High-pass\nFrequency (Hz)")
        self.highPassFreqEdit = QLineEdit()
        self.highPassFreqEdit.setText("0.5")
        self.highPassFreqEdit.setFixedWidth(70)

        # Set widget layouts
        signalWidget = QWidget()
        signalWidget.setLayout(signalLayout)
        signalWidget.setFixedWidth(400)
        signalLayout.addWidget(self.filterType, 0, 0, 
            alignment=Qt.AlignLeft)
        signalLayout.addWidget(self.filterTypeEdit, 1, 0, 
            alignment=Qt.AlignLeft)
        signalLayout.addWidget(self.butterOrderLabel, 0, 1, 
            alignment=Qt.AlignLeft)
        signalLayout.addWidget(self.butterOrderEdit, 1, 1,
            alignment=Qt.AlignLeft)
        signalLayout.addWidget(self.lowPassFreqLabel, 0, 2,
            alignment=Qt.AlignLeft)
        signalLayout.addWidget(self.lowPassFreqEdit, 1, 2,
            alignment=Qt.AlignLeft)
        signalLayout.addWidget(self.highPassFreqLabel, 0, 3,
            alignment=Qt.AlignLeft)
        signalLayout.addWidget(self.highPassFreqEdit, 1, 3,
            alignment=Qt.AlignLeft)
        self.butterOrderLabel.hide()
        self.butterOrderEdit.hide()
        self.lowPassFreqLabel.hide()
        self.lowPassFreqEdit.hide()
        self.highPassFreqLabel.hide()
        self.highPassFreqEdit.hide()
        self.filterTypeEdit.activated.connect(self.check_filter)
 
        paramLayout.addWidget(buttonWidget, 0, 0)
        paramLayout.addWidget(signalWidget, 0, 1)

        # From FPDWindow
        self.paramPlot1 = MinorHeatmapCanvas(self, width=9, height=7, dpi=100)
        self.paramPlot2 = MinorHeatmapCanvas(self, width=9, height=7, dpi=100)
        paramToolbar = NavigationToolbar2QT(self.paramPlot1, self)
        paramToolbar2 = NavigationToolbar2QT(self.paramPlot2, self)

        # Slider widget containing labels and (beat, electrode) sliders
        sliderWidget = QWidget()
        sliderLayout = QGridLayout()
        sliderWidget.setLayout(sliderLayout)
        paramSlider1aLabel = QLabel("Beat")
        paramSlider1bLabel = QLabel("Electrode")
        paramSlider1aLabel.setFixedWidth(60)
        paramSlider1bLabel.setFixedWidth(60)
        self.paramSlider1a = QSlider(Qt.Horizontal)
        self.paramSlider1b = QSlider(Qt.Horizontal)
        sliderLayout.addWidget(paramSlider1aLabel, 0, 0)
        sliderLayout.addWidget(paramSlider1bLabel, 1, 0)
        sliderLayout.addWidget(self.paramSlider1a, 0, 1)
        sliderLayout.addWidget(self.paramSlider1b, 1, 1)
        
        plotLayout.addWidget(self.paramPlot1, 0, 0)
        plotLayout.addWidget(self.paramPlot2, 0, 1)
        plotLayout.addWidget(sliderWidget, 2, 0)
        plotLayout.addWidget(paramToolbar, 1, 0)
        plotLayout.addWidget(paramToolbar2, 1, 1)

        mainLayout.addWidget(paramWidget, 0, 0)
        mainLayout.addWidget(plotWidget, 1, 0)
        self.setLayout(mainLayout)
    
    def check_filter(self):
        if self.filterTypeEdit.currentText() == "No filter":
            self.butterOrderLabel.hide()
            self.butterOrderEdit.hide()
            self.lowPassFreqLabel.hide()
            self.lowPassFreqEdit.hide()
            self.highPassFreqLabel.hide()
            self.highPassFreqEdit.hide()
        elif self.filterTypeEdit.currentText() == "Low-pass Only":
            self.butterOrderLabel.show()
            self.butterOrderEdit.show()
            self.lowPassFreqLabel.show()
            self.lowPassFreqEdit.show()
            self.highPassFreqLabel.hide()
            self.highPassFreqEdit.hide()
        elif self.filterTypeEdit.currentText() == "High-pass Only":
            self.butterOrderLabel.show()
            self.butterOrderEdit.show()
            self.lowPassFreqLabel.hide()
            self.lowPassFreqEdit.hide()
            self.highPassFreqLabel.show()
            self.highPassFreqEdit.show()
        elif self.filterTypeEdit.currentText() == "Bandpass":
            self.butterOrderLabel.show()
            self.butterOrderEdit.show()
            self.lowPassFreqLabel.show()
            self.lowPassFreqEdit.show()
            self.highPassFreqLabel.show()
            self.highPassFreqEdit.show()


# Used for parameter vs distance statistics window.
# Window provides a 2x2 plot and readout panel (right side of window)
# in addition to user parameter input field for sigma (standard deviation)
class PropertyStatWindows(QWidget):
    def __init__(self):
        super(PropertyStatWindows, self).__init__()
        self.setupUI()

    def setupUI(self):
        mainLayout = QGridLayout()
        paramLayout = QGridLayout()
        plotLayout = QGridLayout()
        statsSumLayout = QGridLayout()
        
        paramWidget = QWidget()
        paramWidget.setLayout(paramLayout)
        paramWidget.setFixedWidth(200)
        plotWidget = QWidget()
        plotWidget.setLayout(plotLayout)
        statsSumWidget = QWidget()
        statsSumWidget.setFixedSize(225, 700)
        statsSumWidget.setLayout(statsSumLayout)
        
        sigmaLabel = QLabel("Sigma Value")
        sigmaLabel.setFixedWidth(85)
        self.sigmaEdit = QLineEdit()
        self.sigmaEdit.setText("3")
        self.sigmaEdit.setFixedWidth(70)
        paramLayout.addWidget(sigmaLabel, 0, 0)
        paramLayout.addWidget(self.sigmaEdit, 1, 0)
        self.sigmaButton = QPushButton("Filter \n Outliers")
        self.sigmaButton.setFixedSize(70, 60)
        paramLayout.addWidget(self.sigmaButton, 0, 1, 2, 1)

        self.paramPlot = GenericPlotCanvas(self, width=8, height=7, dpi=100)
        self.paramSlider = QSlider(Qt.Horizontal)
        paramToolbar = NavigationToolbar2QT(self.paramPlot, self)
        plotLayout.addWidget(self.paramPlot, 0, 0)
        plotLayout.addWidget(self.paramSlider, 1, 0)
        plotLayout.addWidget(paramToolbar, 2, 0)

        statTextFont = QFont()
        statTextFont.setBold(True)
        self.statsLabel = QLabel("Statistics Readout")
        self.statsLabel.setFont(statTextFont)
        statsSumLayout.addWidget(self.statsLabel, 0, 0)
        self.statsPrintout = QPlainTextEdit("To be populated")
        self.statsPrintout.setFixedHeight(650)
        self.statsPrintout.setReadOnly(True)
        statsSumLayout.addWidget(self.statsPrintout, 1, 0)

        mainLayout.addWidget(paramWidget, 0, 0)
        mainLayout.addWidget(plotWidget, 1, 0)
        mainLayout.addWidget(statsSumWidget, 0, 1, 2, 1)
        self.setLayout(mainLayout)


# Window class shared by a few functions.
# Currently used for beat amplitude, power spectra GUI windows.
# Produces a window with drop-down boxes to select a range of beats for 
# analysis. 
# Only used by the beat amplitude and power spectra functions.
class PlotBeatSelectWindows(QWidget):
    def __init__(self, analysisGUI):
        super(PlotBeatSelectWindows, self).__init__()
        self.setupUI(analysisGUI)
    
    def setupUI(self, analysisGUI):
        mainLayout = QGridLayout()
        paramLayout = QGridLayout()
        plotLayout = QGridLayout()
        paramWidget = QWidget()
        plotWidget = QWidget()
        paramWidget.setLayout(paramLayout)
        paramWidget.setFixedWidth(530)
        plotWidget.setLayout(plotLayout)
        self.beatRangeLabel = QLabel("Start/End Beats")
        self.beatRangeLabel.setFixedWidth(170)
        self.startBeat = QComboBox()
        self.startBeat.setFixedWidth(80)
        self.startBeat.addItem("Beat 1")
        self.startBeat.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        self.endBeat = QComboBox()
        self.endBeat.setFixedWidth(80)
        self.endBeat.addItem("Beat 1")
        self.endBeat.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        self.elecLabel = QLabel("Electrode")
        self.elecSelect = QComboBox()
        self.elecSelect.setFixedWidth(70)
        self.elecSelect.addItem("F7")
        self.elecSelect.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        self.paramLabel = QLabel("Parameter")
        self.paramSelect = QComboBox()
        self.paramSelect.setFixedWidth(180)
        paramItems = ["Orig. Signal", "Cond. Vel.", "Up. Vel.", "Pacemaker", 
            "Local AT"]
        self.paramSelect.addItems(paramItems)
        self.paramSelect.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        self.plotButton = QPushButton("Plot")
        self.plotButton.setFixedSize(70, 70)
        paramLayout.addWidget(self.plotButton, 0, 0, 2, 1)
        paramLayout.addWidget(self.beatRangeLabel, 0, 1, 1, 2)
        paramLayout.addWidget(self.startBeat, 1, 1)
        paramLayout.addWidget(self.endBeat, 1, 2)
        paramLayout.addWidget(self.elecLabel, 0, 3)
        paramLayout.addWidget(self.elecSelect, 1, 3)
        paramLayout.addWidget(self.paramLabel, 0, 4)
        paramLayout.addWidget(self.paramSelect, 1, 4)

        # Toggle correct canvas function based on PSD vs Beat Amp window.
        if (hasattr(analysisGUI, "psdCheck") is True 
        and analysisGUI.psdCheck is True):
            self.paramPlot = PSDPlotCanvas(self, width=8, height=7, dpi=100)
        elif (hasattr(analysisGUI, "ampCheck") is True 
        and analysisGUI.ampCheck is True):
            self.paramPlot = GenericPlotCanvas(self, width=9, height=7, dpi=100)

        self.paramSlider = QSlider(Qt.Horizontal)
        paramToolbar = NavigationToolbar2QT(self.paramPlot, self)
        plotLayout.addWidget(self.paramPlot, 0, 0)
        plotLayout.addWidget(self.paramSlider, 1, 0)
        plotLayout.addWidget(paramToolbar, 2, 0)

        mainLayout.addWidget(paramWidget, 0, 0)
        mainLayout.addWidget(plotWidget, 1, 0)
        self.setLayout(mainLayout)


# Field potential duration window. Window contains 2 plots
# Plot 1 (left plot, paramPlot1) shows FPD calculation of the voltage trace
# Plot 2 (right plot, paramPlot2) shows FPD heatmap for the selected beat.
class FPDWindow(QWidget):
    def __init__(self, analysisGUI):
        super(FPDWindow, self).__init__()
        self.setupUI(analysisGUI)

    def setupUI(self, analysisGUI):
        layout = QGridLayout()
        self.paramPlot1 = MinorHeatmapCanvas(self, width=9, height=7, dpi=100)
        self.paramPlot2 = MinorHeatmapCanvas(self, width=9, height=7, dpi=100)
        self.paramSlider1a = QSlider(Qt.Horizontal)
        self.paramSlider1b = QSlider(Qt.Horizontal)
        paramToolbar = NavigationToolbar2QT(self.paramPlot1, self)

        layout.addWidget(self.paramPlot1, 0, 0)
        layout.addWidget(self.paramPlot2, 0, 1)
        layout.addWidget(self.paramSlider1a, 1, 0)
        layout.addWidget(self.paramSlider1b, 2, 0)
        layout.addWidget(paramToolbar, 3, 0)
    
        self.setLayout(layout)


# Power law window. Window contains a tri-panel plot and statistical readout
# of model parameters (log-normal, exponential, Weibull, power law, 
# doubly-truncated power law).
class PowerlawWindow(QWidget):
    def __init__(self, analysisGUI):
        super(PowerlawWindow, self).__init__()
        self.setupUI(analysisGUI)
    
    def setupUI(self, analysisGUI):
        mainLayout = QGridLayout()
        paramLayout = QGridLayout()
        plotLayout = QGridLayout()
        LLRvalueLayout = QGridLayout()

        #######################################################################
        # Components of paramLayout
        # Widgets
        paramWidget = QWidget()
        paramWidget.setFixedSize(710, 90)
        paramWidget.setLayout(paramLayout)
        # Drop-down box for enabling set_discrete setting
        discreteLabel = QLabel("Set Discrete\n(Data)")
        discreteLabel.setAlignment(Qt.AlignLeft)
        self.discreteSelect = QComboBox()
        self.discreteSelect.setFixedWidth(70)
        self.discreteSelect.addItem("True")
        self.discreteSelect.addItem("False")
        self.discreteSelect.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        paramLayout.addWidget(
            discreteLabel, 0, 0, 
            alignment=Qt.AlignLeft)
        paramLayout.addWidget(
            self.discreteSelect, 1, 0,
            alignment=Qt.AlignLeft)

        # Drop-down box for enabling multi_events setting   
        multiLabel = QLabel("Filter for\nMultiple Events")
        multiLabel.setAlignment(Qt.AlignLeft)
        self.multiSelect = QComboBox()
        self.multiSelect.setFixedWidth(70)
        self.multiSelect.addItem("True")
        self.multiSelect.addItem("False")
        self.multiSelect.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        paramLayout.addWidget(
            multiLabel, 0, 1, 
            alignment=Qt.AlignLeft)
        paramLayout.addWidget(
            self.multiSelect, 1, 1,
            alignment=Qt.AlignLeft)

        # Drop-down box for choosing bin_method
        binMethodLabel = QLabel("Histogram\nBin Method")
        binMethodLabel.setAlignment(Qt.AlignLeft)
        self.binMethodSelect = QComboBox()
        self.binMethodSelect.setFixedWidth(120)
        self.binMethodSelect.addItem("Sturges' Rule")
        self.binMethodSelect.addItem("Freedman-Diaconis")
        self.binMethodSelect.addItem("Manual Entry")
        self.binMethodSelect.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        paramLayout.addWidget(
            binMethodLabel, 0, 2, 
            alignment=Qt.AlignLeft)
        paramLayout.addWidget(
            self.binMethodSelect, 1, 2,
            alignment=Qt.AlignLeft)

        # Edit field for entering nbins
        self.nbinsLabel = QLabel("n_bins")
        self.nbinsLabel.setFixedWidth(50)
        self.nbinsEdit = QLineEdit()
        self.nbinsEdit.setText("15")
        self.nbinsEdit.setFixedWidth(50)
        paramLayout.addWidget(
            self.nbinsLabel, 0, 3, 
            alignment=Qt.AlignLeft)
        paramLayout.addWidget(
            self.nbinsEdit, 1, 3,
            alignment=Qt.AlignLeft)
        self.nbinsLabel.setVisible(False)
        self.nbinsEdit.setVisible(False)

        # Check whether to display nbins
        self.binMethodSelect.currentIndexChanged.connect(
            lambda: nbins_toggle(analysisGUI))

        # Drop-down box for choosing first_distrib
        distribLabel = QLabel("Distribution 1")
        distribLabel.setAlignment(Qt.AlignLeft)
        self.distribSelect = QComboBox()
        self.distribSelect.setFixedWidth(100)
        self.distribSelect.addItem("power_law")
        self.distribSelect.addItem("lognormal")
        self.distribSelect.addItem("exponential")
        self.distribSelect.addItem("truncated_power_law")
        self.distribSelect.addItem("stretched_exponential")
        self.distribSelect.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        paramLayout.addWidget(
            distribLabel, 0, 4, 
            alignment=Qt.AlignLeft)
        paramLayout.addWidget(
            self.distribSelect, 1, 4,
            alignment=Qt.AlignLeft)

        # Edit field for xmin
        xminLabel = QLabel("Power Law\nx_min")
        xminLabel.setFixedWidth(85)
        self.xminEdit = QLineEdit()
        self.xminEdit.setText("1")
        self.xminEdit.setFixedWidth(70)
        paramLayout.addWidget(
            xminLabel, 0, 5,
            alignment=Qt.AlignLeft)
        paramLayout.addWidget(
            self.xminEdit, 1, 5,
            alignment=Qt.AlignLeft)

        # Edit field for xmax
        xmaxLabel = QLabel("Trunc. PL\nx_max")
        xmaxLabel.setFixedWidth(85)
        self.xmaxEdit = QLineEdit()
        self.xmaxEdit.setText("150")
        self.xmaxEdit.setFixedWidth(70)
        paramLayout.addWidget(
            xmaxLabel, 0, 6,
            alignment=Qt.AlignLeft)
        paramLayout.addWidget(
            self.xmaxEdit, 1, 6,
            alignment=Qt.AlignLeft)

        self.plotPL = QPushButton("Plot")
        self.plotPL.setFixedSize(40, 70)
        paramLayout.addWidget(
            self.plotPL, 0, 7, 2, 1,
            alignment=Qt.AlignLeft)

        # End components of paramLayout
        #######################################################################

        plotWidget = QWidget()
        plotWidget.setLayout(plotLayout)
        LLRvalueWidget = QWidget()
        LLRvalueWidget.setFixedSize(760, 250)
        LLRvalueWidget.setLayout(LLRvalueLayout)

        self.powerlawPlot = PowerlawCanvas(self, width=9, height=6, dpi=150)
        plToolbar = NavigationToolbar2QT(self.powerlawPlot, self)
        plotLayout.addWidget(self.powerlawPlot, 0, 0)
        plotLayout.addWidget(plToolbar, 1, 0)
        
        self.statsPrintout = QPlainTextEdit(
            "powerlaw distribution_compare prinout (awaiting results)")
        self.statsPrintout.setFixedWidth(750)
        self.statsPrintout.setReadOnly(True)
        LLRvalueLayout.addWidget(self.statsPrintout, 1, 0)

        mainLayout.addWidget(paramWidget, 0, 0, 1, 2)
        mainLayout.addWidget(plotWidget, 1, 0)
        mainLayout.addWidget(LLRvalueWidget, 2, 0, 1, 1)
        self.setLayout(mainLayout)
 

# Primary GUI class.
class AnalysisGUI(QMainWindow):
    def __init__(self, raw_data, cm_beats, pace_maker, 
    upstroke_vel, local_act_time, conduction_vel, input_param, heat_map, 
    cm_stats, electrode_config, psd_data, beat_amp_int, batch_data, 
    field_potential, parent=None):
        super().__init__(parent)
        # Function call to establish GUI widgets
        self.setup_UI(raw_data, cm_beats, pace_maker, upstroke_vel, 
            local_act_time, conduction_vel, input_param, heat_map, cm_stats, 
            electrode_config, psd_data, beat_amp_int, batch_data,
            field_potential)

        # Initial file path
        self.file_path = "/"


    # Initiate the GUI by setting up widgets.
    def setup_UI(self, raw_data, cm_beats, pace_maker, upstroke_vel, 
    local_act_time, conduction_vel, input_param, heat_map, cm_stats, 
    electrode_config, psd_data, beat_amp_int, batch_data, field_potential):
        self.setWindowTitle("Cardio PyMEA v1.0")
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.setUnifiedTitleAndToolBarOnMac(True)

        #######################################################################
        # Menu options.
        #######################################################################
        # File Menu
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction("&Import", lambda: data_import(self, 
            raw_data, electrode_config))
        self.fileMenu.addAction("&Batch", lambda: [
            batch_analysis.import_batch_file(self, raw_data, batch_data, 
                electrode_config, cm_beats, input_param, pace_maker, 
                heat_map, local_act_time, upstroke_vel, conduction_vel,
                beat_amp_int)])
        self.fileMenu.addAction("&Save Processed Data",
            lambda: export_excel(self, pace_maker, local_act_time, upstroke_vel, 
                conduction_vel, beat_amp_int, cm_beats, cm_stats, 
                field_potential))
        self.fileMenu.addAction("&Exit", self.close)
       

        # Calculation Menu
        self.calcMenu = self.menuBar().addMenu("&Calculations")
        self.calcMenu.addAction("Find &Beats (Use First!)", 
            lambda: [self.determineBeatsWindow(raw_data, cm_beats, input_param, 
                electrode_config, batch_data)])
        self.calcMenu.addAction("&All (PM, LAT, dV/dt, CV, Amp, Int)",
            lambda: [calculate_pacemaker.calculate_pacemaker(self, cm_beats, 
                pace_maker, heat_map, input_param, electrode_config),
                calculate_lat.calculate_lat(self, cm_beats, local_act_time,
                heat_map, input_param, electrode_config),
                calculate_upstroke_vel.calculate_upstroke_vel(self, cm_beats, 
                upstroke_vel, heat_map, input_param, electrode_config),
                calculate_cv.calculate_conduction_velocity(self, cm_beats, 
                conduction_vel, local_act_time, heat_map, input_param, 
                electrode_config),
                calculate_beat_amp_int.calculate_beat_amp(self, cm_beats, 
                beat_amp_int, pace_maker, local_act_time, heat_map, input_param, 
                electrode_config),
                main_heatmap.graph_all(self, heat_map, cm_beats, 
                pace_maker, upstroke_vel, local_act_time, conduction_vel, 
                input_param)])
        self.calcMenu.addAction("&Pacemaker", 
            lambda: [self.pacemakerWindow(cm_beats, pace_maker, heat_map, 
                input_param),
                calculate_pacemaker.calculate_pacemaker(self, cm_beats, 
                pace_maker, heat_map, input_param, electrode_config),
                calculate_pacemaker.graph_pacemaker(self, heat_map, pace_maker, 
                input_param)])
        self.calcMenu.addAction("&Local Activation Time",
            lambda: [
                self.localActTimeWindow(cm_beats, local_act_time, heat_map, 
                    input_param),
                calculate_lat.calculate_lat(self, cm_beats, local_act_time,
                    heat_map, input_param, electrode_config),
                calculate_lat.graph_local_act_time(self, heat_map, 
                    local_act_time, input_param)])
        self.calcMenu.addAction("&Upstroke Velocity (dV/dt)",
            lambda: [
                self.upVelocityWindow(cm_beats, upstroke_vel, heat_map,
                    input_param),
                calculate_upstroke_vel.calculate_upstroke_vel(self, cm_beats, 
                    upstroke_vel, heat_map, input_param, electrode_config),
                calculate_upstroke_vel.graph_upstroke(self, heat_map, 
                    upstroke_vel, input_param)])
        self.calcMenu.addAction("Conduction &Velocity",
            lambda: [
                self.condVelocityWindow(cm_beats, local_act_time, 
                    conduction_vel, heat_map, input_param),
                calculate_cv.calculate_conduction_velocity(self, cm_beats, 
                    conduction_vel, local_act_time, heat_map, input_param, 
                    electrode_config),
                calculate_cv.graph_conduction_vel(self, heat_map, 
                    local_act_time, conduction_vel, input_param)])
        self.calcMenu.addAction("&Field Potential Duration",
            lambda: [
                self.fieldPotDurWindow(cm_beats, local_act_time,
                    field_potential, heat_map, input_param),
                calculate_fpd.calc_fpd(self, cm_beats, field_potential, 
                    local_act_time, heat_map, input_param, electrode_config)])
       

        # Plot Menu
        self.plotMenu = self.menuBar().addMenu("Other &Plots")
        self.plotMenu.addAction("Cond. Vel. Vector &Field",
            lambda: [self.condVelVectorWindow(cm_beats, local_act_time, 
                conduction_vel, input_param),
                calculate_cv.cv_quiver_plot(self, input_param, local_act_time, 
                conduction_vel)])
        self.plotMenu.addAction("&Beat Amplitude && Interval",
            lambda: [self.beatAmpIntWindow(cm_beats, pace_maker, local_act_time,
                beat_amp_int, input_param, electrode_config), 
                calculate_beat_amp_int.beat_amp_interval_graph(self, 
                electrode_config, beat_amp_int, pace_maker, local_act_time, 
                input_param)])
        self.plotMenu.addAction("Power &Spectrum", 
            lambda: [self.psdPlotWindow(cm_beats, electrode_config, pace_maker, 
                upstroke_vel, local_act_time, conduction_vel, input_param, 
                cm_stats, psd_data),
                psd_plotting.psd_plotting(self, cm_beats, 
                electrode_config, pace_maker, upstroke_vel, local_act_time, 
                conduction_vel, input_param, psd_data)])
        self.plotMenu.addAction("&Estimated Pacemaker Origin",
            lambda: [self.pmOriginWindow(cm_beats, pace_maker, heat_map, 
                input_param),
                calculate_pacemaker.estimate_pm_origin(self, pace_maker, 
                input_param)])


        # Statistics Menu
        self.statMenu = self.menuBar().addMenu("&Statistics")
        self.statMenu.addAction("&Property vs Distance w/ R\u00b2",
            lambda: [self.propertyVsDistStatsWindow(cm_beats, pace_maker, 
                upstroke_vel, local_act_time, conduction_vel, input_param, 
                cm_stats)])
        self.statMenu.addAction("Principal Component &Analysis (PCA)",
            lambda: [self.pcaPlotWindow(cm_beats, beat_amp_int, pace_maker, 
                local_act_time, heat_map, input_param, electrode_config),
                pca_plotting.pca_data_prep(self, cm_beats, beat_amp_int, 
                pace_maker, local_act_time, heat_map, input_param, 
                electrode_config)])
        self.statMenu.addAction("&Power Law Distribution Comparison", 
            lambda: [
                self.powerlaw_window(pace_maker, batch_data, 
                    electrode_config, beat_amp_int),  
                powerlaw_analysis.compare_distribs(self, 
                    pace_maker, batch_data, electrode_config, beat_amp_int)])


        # Tools Menu
        self.toolsMenu = self.menuBar().addMenu("&Tools")
        self.toolsMenu.addAction("&Detect translocations", 
            lambda: [detect_transloc.pm_translocations(
                self, pace_maker, electrode_config, beat_amp_int)])


        # Advanced Tools Menu (ML, etc); To be filled later
        self.advToolsMenu = self.menuBar().addMenu("Advanced T&ools")

        self.testingMenu = self.menuBar().addMenu("Testin&g")
        self.testingMenu.addAction("&Reload modules (debug)", reload_module)


        # Set menubar color scheme, #ccbcc8
        # Credit to this Stack Overflow post for the style sheet config
        # https://stackoverflow.com/questions/28353151/styling-the-menu-bar-in-pyqt4
        self.menuBar().setStyleSheet(
            """
            QMenuBar
            {
                background-color: #5D1049;
                color: #fff;
            }
            QMenuBar::item
            {
                background-color: #5D1049;
                color: #fff;
            }
            QMenuBar::item::selected
            {
                background-color: #ccbcc8;
                color: #000;
            }
            QMenu
            {
                background-color: #bf97b5;
                color: #000;
            }
            QMenu::item::selected
            {
                background-color: #483a78;
                color: #fff;
            }
             """
        )

        # Set window background color.
        self.setStyleSheet("background-color: white; color: black;")

        # Arrange Widgets using row, col grid arrangement.
        mainLayout = QGridLayout()
        paramLayout = QGridLayout()
        plotLayout = QGridLayout()

        # Parameters, linked to paramLayout widget.
        self.paramWidget = QWidget()
        self.paramWidget.setFixedWidth(830)
        self.plotWidget = QWidget()
        self.paramWidget.setLayout(paramLayout)
        self.plotWidget.setLayout(plotLayout)
        # Parameter entry widgets & labels.
        self.pkHeightLab = QLabel("Min Peak\nHeight (µV)")
        self.pkHeightEdit = QLineEdit()
        self.pkHeightEdit.setFixedWidth(85)
        self.pkHeightEdit.setText("100")
        paramLayout.addWidget(self.pkHeightLab, 0, 0)
        paramLayout.addWidget(self.pkHeightEdit, 1, 0)
        self.pkDistLab = QLabel("Min Peak\nDistance (ms)")
        self.pkDistEdit = QLineEdit()
        self.pkDistEdit.setFixedWidth(85)
        self.pkDistEdit.setText("1000")
        paramLayout.addWidget(self.pkDistLab, 0, 1)
        paramLayout.addWidget(self.pkDistEdit, 1, 1)
        # Frequency selection widget.
        self.sampleFreq = QLabel("Sample\nFrequency (Hz)")
        self.sampleFreqEdit = QComboBox()
        self.sampleFreqEdit.setFixedWidth(100)
        self.sampleFreqEdit.addItem("1000")
        self.sampleFreqEdit.addItem("10000")
        self.sampleFreqEdit.setStyleSheet(
            'selection-background-color: #483a78; selection-color: white;')
        paramLayout.addWidget(self.sampleFreq, 0, 2)
        paramLayout.addWidget(self.sampleFreqEdit, 1, 2)
        # Truncation widgets.
        self.truncCheckbox = QCheckBox("Truncate Data\n(minutes)")
        self.truncCheckbox.clicked.connect(lambda: trunc_toggle(self))
        self.truncCheckbox.setFixedWidth(120)
        self.truncStartEdit = QLineEdit()
        self.truncStartEdit.setFixedWidth(55)
        self.truncEndEdit = QLineEdit()
        self.truncEndEdit.setFixedWidth(55)
        paramLayout.addWidget(self.truncCheckbox, 0, 3, 1, 2)
        paramLayout.addWidget(self.truncStartEdit, 1, 3)
        paramLayout.addWidget(self.truncEndEdit, 1, 4)
        self.truncStartEdit.setVisible(False)
        self.truncEndEdit.setVisible(False)
        # Electrode silence widgets (toggle, special combobox w/ multi-select)
        self.toggleSilence = QCheckBox("Silence Electrodes")
        self.toggleSilence.setFixedWidth(150)
        self.toggleSilence.clicked.connect(lambda: silence_toggle(self))
        self.elecCombobox = CheckableComboBox()
        self.elecCombobox.setFixedWidth(140)
        paramLayout.addWidget(self.toggleSilence, 0, 5)
        paramLayout.addWidget(self.elecCombobox, 1, 5)
        self.elecCombobox.setVisible(False)
        # File name label.
        self.fileName = QLabel("Waiting for file.")
        self.fileName.setFixedWidth(300)
        self.fileName.setWordWrap(True)
        paramLayout.addWidget(self.fileName, 0, 6)
        # File duration label.
        self.fileLength = QLabel("Waiting for calculation.")
        self.fileLength.setFixedWidth(300)
        self.fileLength.setWordWrap(True)
        paramLayout.addWidget(self.fileLength, 1, 6)
        
        # Plots, linked to plotLayout widget
        self.mainHeatmap = MainHeatmapCanvas(self, width=10, height=8, dpi=100)
        # (row, column, rowspan, colspan)
        plotLayout.addWidget(self.mainHeatmap, 2, 0, 1, 2)

        self.mainSlider = QSlider(Qt.Horizontal)
        plotLayout.addWidget(self.mainSlider, 3, 0)
        self.mainSlider.valueChanged.connect(lambda: main_heatmap.graph_all(
            self, heat_map, cm_beats, pace_maker, upstroke_vel, 
            local_act_time, conduction_vel, input_param))

        mainToolbar = NavigationToolbar2QT(self.mainHeatmap, self)
        plotLayout.addWidget(mainToolbar, 4, 0)

        # Add parameter and plot widgets to the main GUI layout, then display.
        mainLayout.addWidget(self.paramWidget, 0, 0)
        mainLayout.addWidget(self.plotWidget, 1, 0)
        self.mainWidget.setLayout(mainLayout)
    
    # Generates the beat finder ("Find Beats") window.
    def determineBeatsWindow(self, raw_data, cm_beats, input_param, 
    electrode_config, batch_data):
        self.beatsWindow = BeatSignalPlotWindow()
        self.beatsWindow.setWindowTitle("Find Beats Results")
        self.beatsWindow.setStyleSheet(
            "background-color: white; color: black;")
        self.beatsWindow.show()
        # Change beat slider signal connection
        self.beatsWindow.paramSlider1a.valueChanged.connect(lambda: [
            determine_beats.graph_beats(self, cm_beats, electrode_config),
            determine_beats.full_mea_plot(self, cm_beats, electrode_config)])
        # Change electrode slider signal connection
        self.beatsWindow.paramSlider1b.valueChanged.connect(lambda: [
            determine_beats.graph_beats(self, cm_beats, electrode_config)])
        # Plot button signal connection
        self.beatsWindow.plotButton.clicked.connect(lambda: [
            determine_beats.determine_beats(self, raw_data, cm_beats, 
                input_param, electrode_config, batch_data)])

    # Generates the time lag (pacemaker) (single property) window.
    def pacemakerWindow(self, cm_beats, pace_maker, heat_map, input_param):
        try:
            self.pmWindow = SoloHeatmapWindows(self)
            self.pmWindow.setWindowTitle("Pacemaker Results")
            self.pmWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.pmWindow.show()
            # Set slider value to maximum number of beats
            self.pmWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.pmWindow.paramSlider.valueChanged.connect(lambda: [
                calculate_pacemaker.graph_pacemaker(self, heat_map, pace_maker, 
                input_param)])
        except AttributeError:
            print("")

    # Generates the pacemaker origin estimation window.
    def pmOriginWindow(self, cm_beats, pace_maker, heat_map, input_param):
        try:
            self.circFitWindow = SoloHeatmapWindows(self)
            self.circFitWindow.setWindowTitle("Predicted PM Origin")
            self.circFitWindow.show()
            self.circFitWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.circFitWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.circFitWindow.paramSlider.valueChanged.connect(lambda: [
                calculate_pacemaker.estimate_pm_origin(self, pace_maker, 
                input_param)])
        except AttributeError:
            print("")

    # Generates the upstroke velocity (dV/dt) (single property) window.
    def upVelocityWindow(self, cm_beats, upstroke_vel, heat_map, input_param):
        try:
            self.dvdtWindow = SoloHeatmapWindows(self)
            self.dvdtWindow.setWindowTitle("Upstroke Velocity (dV/dt) Results")
            self.dvdtWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.dvdtWindow.show()
            # Set slider value to maximum number of beats
            self.dvdtWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.dvdtWindow.paramSlider.valueChanged.connect(lambda: [
                calculate_upstroke_vel.graph_upstroke(self, heat_map, 
                upstroke_vel, input_param)])
        except (AttributeError):
            print("")

    # Generates the local activation time (single property) window.
    def localActTimeWindow(self, cm_beats, local_act_time, heat_map, 
    input_param):
        try:
            self.latWindow = SoloHeatmapWindows(self)
            self.latWindow.setWindowTitle("Local Activation Time (LAT) Results")
            self.latWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.latWindow.show()
            # Set slider value to maximum number of beats
            self.latWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.latWindow.paramSlider.valueChanged.connect(lambda: [
                calculate_lat.graph_local_act_time(self, heat_map, 
                local_act_time, input_param)])
        except (AttributeError):
            print("")

    # Generates the conduction velocity (single property) window.
    def condVelocityWindow(self, cm_beats, local_act_time, conduction_vel, 
    heat_map, input_param):
        try:
            self.cvWindow = SoloHeatmapWindows(self)
            self.cvWindow.setWindowTitle("Conduction Velocity (CV) Results")
            self.cvWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.cvWindow.show()
            # Set slider value to maximum number of beats
            self.cvWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.cvWindow.paramSlider.valueChanged.connect(lambda: [
                calculate_cv.graph_conduction_vel(self, heat_map, 
                local_act_time, conduction_vel, input_param)])
        except (AttributeError):
            print("")

    # Generates the conduction velocity vector-view window.
    def condVelVectorWindow(self, cm_beats, local_act_time, conduction_vel, 
    input_param):
        try:
            self.cvVectWindow = SoloHeatmapWindows(self)
            self.cvVectWindow.setWindowTitle("Conduction Velocity Vector Field")
            self.cvVectWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.cvVectWindow.show()
            self.cvVectWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.cvVectWindow.paramSlider.valueChanged.connect(lambda: [
                calculate_cv.cv_quiver_plot(self, input_param, local_act_time, 
                conduction_vel)])
        except AttributeError:
            print("No data found.")

    # Generates the field potential duration (FPD) window.
    def fieldPotDurWindow(self, cm_beats, local_act_time, field_potential, 
    heat_map, input_param):
        try:
            self.fpdWindow = FPDWindow(self)
            self.fpdWindow.setWindowTitle(
                "Field Potential Duration (FPD) Results")
            self.fpdWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.fpdWindow.show()
            # Set slider value to maximum number of beats
            self.fpdWindow.paramSlider1a.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.fpdWindow.paramSlider1a.valueChanged.connect(
                lambda: [
                    calculate_fpd.graph_T_wave(self, cm_beats, local_act_time,
                        field_potential, input_param),
                    calculate_fpd.heatmap_fpd(self, cm_beats, field_potential,
                        heat_map, input_param)
                ])
            self.fpdWindow.paramSlider1b.valueChanged.connect(
                lambda: [
                    calculate_fpd.graph_T_wave(self, cm_beats, local_act_time,
                        field_potential, input_param)
                ])
        except (AttributeError):
            print("")

    # Generates the property vs distance analysis window.
    def propertyVsDistStatsWindow(self, cm_beats, pace_maker, upstroke_vel, 
    local_act_time, conduction_vel, input_param, cm_stats):
        try:
            self.pvdWindow = PropertyStatWindows()
            self.pvdWindow.setWindowTitle("Property vs Distance w/ R\u00b2")
            self.pvdWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.pvdWindow.show()
            self.pvdWindow.sigmaButton.clicked.connect(lambda: [
                property_vs_distance_stats.property_vs_distance_analysis(self, 
                cm_beats, pace_maker, upstroke_vel, local_act_time, 
                conduction_vel, input_param, cm_stats)])
            self.pvdWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.pvdWindow.paramSlider.valueChanged.connect(lambda: [
                property_vs_distance_stats.property_vs_distance_graphing(self, 
                cm_beats, pace_maker, upstroke_vel, local_act_time, 
                conduction_vel, input_param, cm_stats)])
        except AttributeError:
            print("")

    # Generates the PSD plotting window.
    def psdPlotWindow(self, cm_beats, electrode_config, pace_maker, 
    upstroke_vel, local_act_time, conduction_vel, input_param, cm_stats, 
    psd_data):
        try:
            self.psdCheck = True
            self.psdWindow = PlotBeatSelectWindows(self)
            self.psdWindow.setWindowTitle("Power Spectra")
            self.psdWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.psdWindow.show()
            self.psdWindow.plotButton.clicked.connect(lambda: [
                psd_plotting.psd_plotting(self, cm_beats, electrode_config, 
                pace_maker, upstroke_vel, local_act_time, conduction_vel, 
                input_param, psd_data)])
            self.psdWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.psdWindow.paramSlider.valueChanged.connect(lambda: [
                psd_plotting.psd_plotting(self, cm_beats, electrode_config, 
                pace_maker, upstroke_vel, local_act_time, conduction_vel, 
                input_param, psd_data)])
            self.psdCheck = False
        except AttributeError:
            print("")

    # Generates the beat amplitude and interval window.
    def beatAmpIntWindow(self, cm_beats, pace_maker, local_act_time,
    beat_amp_int, input_param, electrode_config):
        try:
            self.ampCheck = True
            self.ampIntWindow = PlotBeatSelectWindows(self)
            self.ampIntWindow.setWindowTitle("Beat Amplitude & Interval")
            self.ampIntWindow.setStyleSheet(
                "background-color: white; color: black;")
            self.ampIntWindow.show()
            self.ampIntWindow.paramSlider.setMaximum(
                int(cm_beats.beat_count_dist_mode[0]) - 1)
            self.ampIntWindow.paramSlider.valueChanged.connect(lambda: [
                calculate_beat_amp_int.beat_amp_interval_graph(self, 
                electrode_config, beat_amp_int, pace_maker, local_act_time, 
                input_param)])
            self.ampIntWindow.plotButton.clicked.connect(lambda: [
                calculate_beat_amp_int.beat_amp_interval_graph(self, 
                electrode_config, beat_amp_int, pace_maker, local_act_time, 
                input_param)])
            self.ampIntWindow.elecLabel.hide()
            self.ampIntWindow.elecSelect.hide()
            self.ampIntWindow.paramLabel.hide()
            self.ampIntWindow.paramSelect.hide()
            self.ampCheck = False
        except AttributeError:
            print("")

    # Generates the principle component analysis window.
    def pcaPlotWindow(self, cm_beats, beat_amp_int, pace_maker, local_act_time, 
    heat_map, input_param, electrode_config):
        self.pcaCheck = True
        self.pcaWindow = SoloHeatmapWindows(self)
        self.pcaWindow.setWindowTitle("PCA Analysis")
        self.pcaWindow.setStyleSheet(
            "background-color: white; color: black;")
        self.pcaWindow.show()
        self.pcaWindow.plotButton.clicked.connect(lambda: [
            pca_plotting.pca_data_prep(self, cm_beats, beat_amp_int, pace_maker, 
            local_act_time, heat_map, input_param, electrode_config)])
        self.pcaWindow.paramSlider.hide()
        self.pcaCheck = False

    # Generates the power law analysis window.
    def powerlaw_window(self, pace_maker, batch_data, electrode_config, 
    beat_amp_int):
        self.plWindow = PowerlawWindow(self)
        self.plWindow.setWindowTitle(
            "Powerlaw Distribution Comparison")
        self.plWindow.setStyleSheet(
            "background-color: white; color: black;")
        self.plWindow.plotPL.clicked.connect(
            lambda: powerlaw_analysis.compare_distribs(self, 
                pace_maker, batch_data, electrode_config, beat_amp_int))
        self.plWindow.show()


# Main function. Instantiate data classes, GUI, execute application.
def main():
    raw_data = ImportedData()
    cm_beats = BeatAmplitudes()
    pace_maker = PacemakerData()
    upstroke_vel = UpstrokeVelData()
    local_act_time = LocalATData()
    conduction_vel = CondVelData()
    field_potential = FieldPotDurationData()
    input_param = InputParameters()
    heat_map = MEAHeatMaps()
    cm_stats = StatisticsData()
    psd_data = PSDData()
    electrode_config = ElectrodeConfig(raw_data)
    beat_amp_int = BeatAmpIntData()
    batch_data = BatchData()
    batch_data.batch_config = False

    app = QApplication([])
    app.setStyle("Fusion")

    analysisGUI = AnalysisGUI(raw_data, cm_beats, pace_maker, upstroke_vel, 
        local_act_time, conduction_vel, input_param, heat_map, cm_stats, 
        electrode_config, psd_data, beat_amp_int, batch_data, field_potential)
    analysisGUI.show()
    sys.exit(app.exec_())

main()
