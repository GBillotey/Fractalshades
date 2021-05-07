# -*- coding: utf-8 -*-
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QApplication)

import fractalshades.gui as fsgui


def getapp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_Image_widget():
    from viewer import Image_widget
    class Mywindow(QMainWindow):
        def __init__(self, im):
            super().__init__(parent=None)
            self._im = im
            self.setWindowTitle('Testing Image of Fractalshades')
            self.xy_ratio = 1.0
            #self.setWindowState(Qt.WindowMaximized)
    
            # And don't forget to call setCentralWidget to your main layout widget.
            mw =  Image_widget(self) # fsgui.
            self.setCentralWidget(mw)

    im = os.path.join("/home/geoffroy/Pictures/math/github_fractal_rep/Fractal-shades/tests/images_REF",
                      "test_M2_antialias_E0_2.png")
    app = getapp()
    win = Mywindow(im)
    win.show()
    app.exec()

def test_QDict_viewer_widget():
    from collections import OrderedDict
    from widgets import QDict_viewer
#    class Mywindow(QMainWindow):
#    # https://doc.qt.io/qt-5/qmainwindow.html
#        def __init__(self, odict):
#            super().__init__(parent=None)
#            self.setWindowTitle('Testing Image of Fractalshades')
#            mw =  QDict_viewer(self, odict)
#            self.setCentralWidget(mw)
    app = getapp()
    odict = OrderedDict([("a", 1.), ("b", 2.), ("c", 3.)])
    win = QDict_viewer(None, odict)
    win.show()
    win.values_update({"b": "voilou"})
    app.exec()
    
    
    

if __name__ == "__main__":
    test_Image_widget()
    # test_QDict_viewer_widget()
