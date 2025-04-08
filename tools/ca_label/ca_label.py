import sys
#import signal
from PyQt6 import QtWidgets

from src.mainwindow import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    #signal.signal(signal.SIGINT, lambda *a: app.quit())
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())