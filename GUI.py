from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
)
from PyQt6.QtGui import (
    QColor,
)
from functools import partial

import sys

import MiDaS
import Tools
import inject
import os
from enum import Enum


class Style(Enum):
    #path = "QTextEdit {color: rgb(255, 170, 255)}"
    path = "QTextEdit {color: #FEF5AC}"
    regex = "QLineEdit {color: #90B77D}"
    photogrammetry = "style=\"color:#A8A4CE;\""
    dm_generation = "style=\"color:#A8A4CE;\""
    correction = "style=\"color:#A8A4CE;\""
    meta_injection = "style=\"color:Tomato;\""


class Gui(object):
    __instance = None

    # Configuration ToDo: changeable in gui (optional)
    injection_input_dir = "./tmp/render_out/"
    injection_output_dir = "./injection_out/"
    depth_maps_dir = "./tmp/depth_maps"
    photogrammetry_dir = "./tmp/photogrammetry"  # images ready for photogrammetry, replaces injection out

    # Data
    regex = None
    files = []

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Gui.__instance is None:
            Gui()
        return Gui.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Gui.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Gui.__instance = self

    def __path_txt(self, text: str) -> QTextEdit:
        txt_path = QTextEdit(text)
        txt_path.setStyleSheet(Style.path.value)
        txt_path.setReadOnly(True)
        txt_path.setMaximumHeight(27)
        return txt_path

    def __regex_txt(self, text: str) -> QLineEdit:
        txt_regex = QLineEdit(text)
        txt_regex.setStyleSheet(Style.regex.value)
        return txt_regex

    def __build_modul(self, title: str, action: tuple, path: str = "./") -> QGridLayout:
        layout = QGridLayout()

        # Title
        layout.addWidget(QLabel(title), 0, 0, 1, 2)

        # Path section
        layout.addWidget(QLabel(f"Path where it gets the files from: \t"), 1, 0)
        path_tbx = self.__path_txt(path)
        layout.addWidget(path_tbx, 1, 1)

        # Reg ex for file selection sections
        layout.addWidget(QLabel("Regex: \t"), 2, 0)
        tbx_regex = self.__regex_txt("")
        layout.addWidget(tbx_regex, 2, 1)
        print(f"regex: [{tbx_regex.text()}]")

        # Output text box
        txt_output = QTextEdit()
        txt_output.setMinimumHeight(90)
        txt_output.setDisabled(True)
        layout.addWidget(txt_output, 3, 0, 1, 2)

        # Buttons
        btn_select = QPushButton("Select")
        btn_select.clicked.connect(partial(self.select, tbx_regex, path_tbx, txt_output))
        # btn_select.clicked.connect(tests)
        btn_action = QPushButton(action[0])
        btn_action.clicked.connect(action[-1])
        layout.addWidget(btn_select, 4, 0)
        layout.addWidget(btn_action, 4, 1)

        return layout

    def select(self, text: QLineEdit, path: QTextEdit, out: QTextEdit):
        regex = text.text()
        self.files = Tools.load_files(regex, path.toPlainText())
        if not self.files:
            out.setText("No matching files found, check console for errors")
        else:
            width = 120  # ToDo format with given width, note width is number of characters not pixels!
            out.setText(Tools.col_format(self.files, width))

    def inject_meta(self):  # , files: list):
        if not self.files:
            print("No files selected")
            return -1
        source = "injectionStuff/raw.jpg"
        exif_data = inject.extract_exif(source)
        print("inject meta: \n", exif_data)
        print("into files: \n", self.files)
        return
        inject.inject_exif(exif_data, files)

    def generate_dm(self):
        print("generating dms")

    def photogrammetry(self):
        print("3d reconstruction frm images")

    def correct(self):
        print("Do my shit")

    def build(self):
        app = QApplication([])

        window = QWidget()
        window.setWindowTitle("Faust")
        window.setGeometry(100, 100, 960, 540)
        layout = QVBoxLayout()
        spacing = 20

        # Script info / data
        script_info = QGridLayout()
        script_info.addWidget(QLabel("Current path: "), 0, 0)
        script_info.addWidget(self.__path_txt(os.getcwd()), 0, 1)
        layout.addLayout(script_info)
        layout.addSpacing(spacing)

        # Photogrammetry
        title = f"<h2 {Style.photogrammetry.value}>3D reconstruction</h2>"
        path = self.photogrammetry_dir
        action = ("Run script", self.photogrammetry)
        layout.addLayout(self.__build_modul(title, action, path))
        layout.addSpacing(spacing)

        # Generate Depth Map
        title = f"<h2 {Style.dm_generation.value}>Generate depth maps</h2>"
        path = self.depth_maps_dir
        action = ("Generate", self.generate_dm)
        layout.addLayout(self.__build_modul(title, action, path))
        layout.addSpacing(spacing)

        # Correct model with depth maps
        title = f"<h2 {Style.correction.value}>Correct model with depth maps</h2>"
        path = self.depth_maps_dir
        action = ("Correct", self.correct)
        layout.addLayout(self.__build_modul(title, action, path))
        layout.addSpacing(spacing)

        # Inject Metadata - will be incorporated somewhere
        title = f"<h2 {Style.meta_injection.value}>Inject exif data into images</h2>"
        path = self.injection_input_dir
        action = ("Inject", self.inject_meta)
        layout.addLayout(self.__build_modul(title, action, path))

        # Finalise
        window.setLayout(layout)
        window.show()
        sys.exit(app.exec())

