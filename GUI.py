from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
)
from functools import partial

import sys

import Tools
import inject
import os
from enum import Enum


class Style(Enum):
    path = "QTextEdit {color: #FEF5AC}"
    regex = "QLineEdit {color: #90B77D}"
    output = "QTextEdit {color: #AAAAAA}"
    photogrammetry = "style=\"color:#A8A4CE;\""
    dm_generation = "style=\"color:#A8A4CE;\""
    correction = "style=\"color:#A8A4CE;\""
    meta_injection = "style=\"color:Tomato;\""


def _path_txt(text: str) -> QTextEdit:
    txt_path = QTextEdit(text)
    txt_path.setStyleSheet(Style.path.value)
    txt_path.setReadOnly(True)
    txt_path.setMaximumHeight(27)
    return txt_path


def _regex_txt(text: str) -> QLineEdit:
    txt_regex = QLineEdit(text)
    txt_regex.setStyleSheet(Style.regex.value)
    return txt_regex


class GuiModul(QGridLayout):
    __selection = None  # list
    __tbx_regex = None  # QLineEdit
    __txt_path = None  # QTextEdit
    __txt_output = None  # QTextEdit

    def __init__(self, title: str, action: tuple, path: str = "./"):
        super().__init__()
        layout = self

        # Title
        layout.addWidget(QLabel(title), 0, 0, 1, 2)

        # Path section
        layout.addWidget(QLabel(f"Path where it gets the files from: \t"), 1, 0)
        self.__txt_path = _path_txt(path)
        layout.addWidget(self.__txt_path, 1, 1)

        # Regex for file selection section
        layout.addWidget(QLabel("Regex: \t"), 2, 0)
        self.__tbx_regex = _regex_txt(".*")
        layout.addWidget(self.__tbx_regex, 2, 1)

        # Output text box
        self.__txt_output = QTextEdit()
        self.__txt_output.setObjectName("output_box")
        self.__txt_output.setMinimumHeight(90)
        self.__txt_output.setMinimumWidth(120*6)
        font = QFont("Consolas")
        font.setPixelSize(13)
        self.__txt_output.setFont(font)
        self.__txt_output.setStyleSheet(Style.output.value)
        self.__txt_output.setReadOnly(True)
        layout.addWidget(self.__txt_output, 3, 0, 1, 2)

        # Buttons
        btn_select = QPushButton("Select")
        btn_select.clicked.connect(self.__select)
        btn_action = QPushButton(action[0])
        btn_action.clicked.connect(partial(action[-1], self))
        layout.addWidget(btn_select, 4, 0)
        layout.addWidget(btn_action, 4, 1)

        print("Modul created: ", title)
        print(f"regex: [{self.__tbx_regex.text()}]")

    def __select(self):
        regex = self.__tbx_regex.text()
        self.__selection = Tools.load_files(regex, self.__txt_path.toPlainText())
        if not self.__selection:
            self.__txt_output.setText("No matching files found, check console for errors")
        else:
            width = 120  # ToDo format with given width, note width is number of characters not pixels!
            #print("actual width: ", self.__txt_output.width())  # is always 2px bigger than actual edges
            # inner border is 5px
            # one Character is 9px
            #print("width in mm: ", self.__txt_output.widthMM())
            #print("width in characters: ", int(self.__txt_output.width() / 9))

            # ToDo: fix rough estimation, because the size is the height not width, which I actually need
            font_width = (self.__txt_output.currentFont().pixelSize() + 2.4) / 2
            width = int(self.__txt_output.width() / font_width)
            print("txt width: ", self.__txt_output.width())
            print("px size:", self.__txt_output.currentFont().pixelSize())
            print("space", self.__txt_output.currentFont().letterSpacing())
            # self.__txt_output.setText("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmMMmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
            self.__txt_output.setText(Tools.columnify(self.__selection, width))

    def get_selection(self) -> list:
        return self.__selection

    def get_path(self) -> str:
        return self.__txt_path.toPlainText()


def _inject_meta(exif_donor: QLineEdit, modul: GuiModul):
    files = modul.get_selection()
    if not files:
        print("No files selected")
        return -1

    source = exif_donor.text()
    if not os.path.exists(source):
        print("Source file does not exist: ", source)
        return -1

    exif_data = inject.extract_exif(source)
    print("inject meta: \n", exif_data)
    print("into files: \n", files)
    print()
    # inject.inject_exif(exif_data, modul.get_selection())
    return


def _generate_dm(modul: GuiModul):
    files = modul.get_selection()
    if not files:
        print("No files selected")
        return -1

    print("midas files: ", modul.get_selection())
    print("generating dms")
    print()


def _photogrammetry(modul: GuiModul):
    files = modul.get_selection()
    if not files:
        print("No files selected")
        return -1

    print("photogrammetry files: ", modul.get_selection())
    print("3d reconstruction from images")
    print()


def _correct(modul: GuiModul):
    files = modul.get_selection()
    if not files:
        print("No files selected")
        return -1

    print("correction files: ", modul.get_selection())
    print("Do my shit")
    print()


class Gui(object):
    __instance = None

    # Configuration ToDo: changeable in gui (optional)
    injection_input_dir = "./tmp/render_out/"
    injection_output_dir = "./injection_out/"
    photogrammetry_dir = "./tmp/photogrammetry/"  # images ready for photogrammetry, replaces injection out
    depth_maps_dir = "./tmp/depth_maps/"

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

    def build(self):
        app = QApplication([])

        window = QWidget()
        window.setWindowTitle("Faust")
        window.setGeometry(100, 100, 960, 540)
        layout = QGridLayout()

        # Script info / data
        script_info = QGridLayout()
        script_info.addWidget(QLabel("Current path: "), 0, 0)
        script_info.addWidget(_path_txt(os.getcwd()), 0, 1)
        # Add to window
        layout.addLayout(script_info, 0, 0, 1, 2)

        # Photogrammetry
        title = f"<h2 {Style.photogrammetry.value}>3D reconstruction</h2>"
        path = self.photogrammetry_dir
        action = ("Run script", _photogrammetry)
        photogrammetry_modul = GuiModul(title, action, path)
        # Add to window
        layout.addLayout(photogrammetry_modul, 1, 0)

        # Generate Depth Map
        title = f"<h2 {Style.dm_generation.value}>Generate depth maps</h2>"
        path = self.depth_maps_dir
        action = ("Generate", _generate_dm)
        dm_modul = GuiModul(title, action, path)
        # Add to window
        layout.addLayout(dm_modul, 1, 1)

        # Correct model with depth maps
        title = f"<h2 {Style.correction.value}>Correct model with depth maps</h2>"
        path = self.depth_maps_dir
        action = ("Correct", _correct)
        correction_modul = GuiModul(title, action, path)
        # Add to window
        layout.addLayout(correction_modul, 2, 0)

        # Inject Metadata - will be incorporated somewhere
        title = f"<h2 {Style.meta_injection.value}>Inject exif data into images</h2>"
        path = self.injection_input_dir
        txt_source = _regex_txt("injectionStuff/raw.jpg")
        action = ("Inject", partial(_inject_meta, txt_source))
        inject_modul = GuiModul(title, action, path)
        inject_modul.addWidget(QLabel("Source of exif data"), 5, 0)
        inject_modul.addWidget(txt_source, 5, 1)
        # Add to window
        layout.addLayout(inject_modul, 2, 1)

        # Finalise
        window.setLayout(layout)
        window.show()
        sys.exit(app.exec())
