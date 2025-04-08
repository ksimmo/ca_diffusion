from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MouseTrackingLabel(QLabel):
    moved = pyqtSignal(int,int) #pixel position y,x
    clicked = pyqtSignal(int, int, int)

    def __init__(self, parent: QWidget=None):
        super().__init__()
        QLabel.__init__(self, parent)
        self.setMouseTracking(True) #track mouse movement even if no button is pressed

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.isEnabled():
            pos = ev.position()
            x = int(pos.x())
            y = int(pos.y())
            self.moved.emit(x, y) #convert to shifted coordinates

    def mousePressEvent(self, ev: QMouseEvent):
        if self.isEnabled():
            pos = ev.position()
            x = int(pos.x())
            y = int(pos.y())

            action = ev.button() #left click=1, right click=2
            if action==Qt.MouseButton.LeftButton: #set annotation
                action = 1
            elif action==Qt.MouseButton.RightButton: #remove annotation
                action = 0
            else:
                action = 0

            self.clicked.emit(x, y, action)


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)