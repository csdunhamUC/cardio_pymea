# Author: Christopher S. Dunham
# Gimzewski Laboratory, UCLA Dept. of Chemistry & Biochem
# Began: 2/23/2021
# PyQt5 version of MEA analysis software.

import sys
from PyQt5.QtWidgets import (QApplication, QGridLayout, QMainWindow, 
    QPushButton, QWidget, QDialog, QSlider, QComboBox, QProgressBar)
from PyQt5.QtCore import Qt
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
        self.setWindowTitle("This is a WINDOW")
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        self.menu = self.menuBar().addMenu("&File")
        self.menu.addAction("&Import")
        self.menu.addAction("&Save Processed Data")
        self.menu.addAction("&Print", print_something)

        self.pushButton = QPushButton("Button!")
        self.pushButton2 = QPushButton("Button 2!")
        self.pushButton3 = QPushButton("Button 3!3")
        self.slider = QSlider(Qt.Horizontal)
        self.plotWindow = MainHeatmapCanvas(self, width=5, height=5, dpi=100)
        self.plotWindow.axes.plot(x_var, y_var)

        self.plotTwo = MinorHeatmapCanvas(self, width=5, height=5, dpi=100)
        self.plotTwo.axes.plot(x_var, y_var)

        self.pushButton2.clicked.connect(print_something)
        self.slider.sliderReleased.connect(lambda: print_slider(self))

        # Arrange Widgets using row, col grid arrangement.
        layout = QGridLayout()
        layout.addWidget(self.pushButton, 0, 0)
        layout.addWidget(self.pushButton2, 0, 1)
        layout.addWidget(self.pushButton3, 1, 0)
        layout.addWidget(self.slider, 3, 0)
        # (row, column, rowspan, colspan)
        layout.addWidget(self.plotWindow, 2, 0, 1, 2)
        layout.addWidget(self.plotTwo, 4, 0, 1, 2)
        self.mainWidget.setLayout(layout)


def main():
    app = QApplication([])
    app.setStyle("Fusion")

    x_var = [0,1,2,3,4,8]
    y_var = [10,20,30,40,50,60]

    analysisGUI = AnalysisGUI(x_var, y_var)
    
    
    
    analysisGUI.show()
    sys.exit(app.exec_())

main()