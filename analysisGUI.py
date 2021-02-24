# Author: Christopher S. Dunham
# Gimzewski Laboratory, UCLA Dept. of Chemistry & Biochem
# Began: 2/23/2021
# PyQt5 version of MEA analysis software.

import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QMainWindow, 
    QPushButton, QWidget, QDialog, QSlider, QComboBox, QProgressBar, QLineEdit, 
    QLabel)
from PyQt5.QtCore import QLine, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import pyplot as plt


def print_something():
    print("Something.")


def print_slider(analysisGUI):
    print(analysisGUI.slider.value())


class MainHeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MainHeatmapCanvas, self).__init__(fig)


class MinorHeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MinorHeatmapCanvas, self).__init__(fig)


class AnalysisGUI(QMainWindow):
    def __init__(self, x_var, y_var, parent=None):
        super().__init__(parent)
        self.setup_UI(x_var, y_var)

    def setup_UI(self, x_var, y_var):
        self.setWindowTitle("Analysis GUI - PyQt Version 0.1")
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        # Menu options.
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction("&Import")
        self.fileMenu.addAction("&Save Processed Data")
        self.fileMenu.addAction("&Print (debug)", print_something)
        self.calcMenu = self.menuBar().addMenu("&Calculations")
        self.calcMenu.addAction("&Find Beats (Use First!)")
        self.calcMenu.addAction("&Calculate All (PM, LAT, dV/dt, CV, Amp, Int)")
        self.calcMenu.addAction("&Calculate Pacemaker")
        self.calcMenu.addAction("&Calculate Local Act. Time")
        self.calcMenu.addAction("&Calculate Upstroke Velocity")
        self.calcMenu.addAction("&Calculate Conduction Velocity")

        self.plotMenu = self.menuBar().addMenu("&Special Plots")
        self.plotMenu.addAction("&Cond. Vel. Vector Field")
        self.plotMenu.addAction("&Beat Amplitude & Interval")
        self.plotMenu.addAction("&Manual Electrode Filter")

        self.statMenu = self.menuBar().addMenu("&Statistics")
        self.statMenu.addAction("&Param vs Distance w/ R-value")
        self.statMenu.addAction("&Power Spectrum")

        # To be filled later
        self.toolsMenu = self.menuBar().addMenu("&Tools")
        # To be filled later
        self.advToolsMenu = self.menuBar().addMenu("&Advanced Tools")

        self.testingMenu = self.menuBar().addMenu("&Testing")
        self.testingMenu.addAction("&Reload modules (debug)")

        # Arrange Widgets using row, col grid arrangement.
        mainLayout = QGridLayout()
        paramLayout = QGridLayout()
        plotLayout = QGridLayout()

        # Entry layout using HBox

        # Parameters, linked to paramLayout widget
        self.paramWidget = QWidget()
        self.plotWidget = QWidget()
        self.paramWidget.setLayout(paramLayout)
        self.plotWidget.setLayout(plotLayout)

        self.pkHeightLab = QLabel("Min Peak" + "\n" + "Height")
        self.pkHeightEdit = QLineEdit()
        self.pkHeightEdit.setFixedWidth(70)
        paramLayout.addWidget(self.pkHeightLab, 0, 0)
        paramLayout.addWidget(self.pkHeightEdit, 1, 0)
        self.pkDistLab = QLabel("Min Peak" + "\n" + "Distance")
        self.pkDistEdit = QLineEdit()
        self.pkDistEdit.setFixedWidth(70)
        paramLayout.addWidget(self.pkDistLab, 0, 1)
        paramLayout.addWidget(self.pkDistEdit, 1, 1)
        self.pkProm = QLabel("Peak" + "\n" + "Prominence")
        self.pkPromEdit = QLineEdit()
        self.pkPromEdit.setFixedWidth(70)
        paramLayout.addWidget(self.pkProm, 0, 2)
        paramLayout.addWidget(self.pkPromEdit, 1, 2)
        self.pkWidth = QLabel("Peak" + "\n" + "Width")
        self.pkWidthEdit = QLineEdit()
        self.pkWidthEdit.setFixedWidth(70)
        paramLayout.addWidget(self.pkWidth, 0, 3)
        paramLayout.addWidget(self.pkWidthEdit, 1, 3)
        self.pkThresh = QLabel("Peak" + "\n" + "Threshold")
        self.pkThreshEdit = QLineEdit()
        self.pkThreshEdit.setFixedWidth(70)
        paramLayout.addWidget(self.pkThresh, 0, 4)
        paramLayout.addWidget(self.pkThreshEdit, 1, 4)
        self.sampleFreq = QLabel("Sample" + "\n" + "Frequency")
        self.sampleFreqEdit = QComboBox()
        self.sampleFreqEdit.setFixedWidth(85)
        self.sampleFreqEdit.addItem("1000 Hz")
        self.sampleFreqEdit.addItem("10,000 Hz")
        paramLayout.addWidget(self.sampleFreq, 0, 5)
        paramLayout.addWidget(self.sampleFreqEdit, 1, 5)
        
        # Plots, linked to plotLayout widget
        self.plotWindow = MainHeatmapCanvas(self, width=5, height=5, dpi=100)
        # (row, column, rowspan, colspan)
        plotLayout.addWidget(self.plotWindow, 2, 0, 1, 2)
        self.plotWindow.axes.plot(x_var, y_var)
        self.plotTwo = MinorHeatmapCanvas(self, width=5, height=5, dpi=100)
        plotLayout.addWidget(self.plotTwo, 4, 0, 1, 2)
        self.plotTwo.axes.plot(x_var, y_var)
        self.slider = QSlider(Qt.Horizontal)
        plotLayout.addWidget(self.slider, 3, 0)
        self.slider.sliderReleased.connect(lambda: print_slider(self))

        mainLayout.addWidget(self.paramWidget, 0, 0)
        mainLayout.addWidget(self.plotWidget, 1, 0)
        self.mainWidget.setLayout(mainLayout)


def main():
    app = QApplication([])
    app.setStyle("Fusion")

    x_var = [0,1,2,3,4,8]
    y_var = [10,20,30,40,50,60]

    analysisGUI = AnalysisGUI(x_var, y_var)
    # analysisGUI.resize(1200, 800)
    analysisGUI.show()
    sys.exit(app.exec_())

main()