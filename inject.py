#! /usr/bin/python3

# requires
# pip install exif

import os
# import io
import re
import sys
import Tools

# Exif stuff
import piexif
from PIL import Image

input_dir = "./tmp/render_out/"
output_dir = "./injection_out/"

# ToDo test other libs how they can handle exif data (PIL, exif, piexif)
# ToDo clear output folder


def inject_exif(source: str, regex: str) -> int:
    source = "injectionStuff/raw.jpg"

    # extract real exif data
    exif_bytes = extract_exif(source)

    # inject exif data into render images
    from PIL import Image
    # files = load_files(regex)
    files = Tools.load_files(regex)

    Tools.cli_confirm_files(files)

    for file in files:
        # edit_piexif(file)
        out = Image.open(input_dir + file)
        # out.save('_%s' % file, "jpeg", exif=exif_bytes)
        out.save(output_dir + file, "jpeg", exif=exif_bytes)

    return 0


def extract_exif(donor: str) -> bytes:
    # Extract source exif data
    source = Image.open(donor)
    exif_dict = piexif.load(source.info['exif'])

    # Dump exif data as bytes
    exif_bytes = piexif.dump(exif_dict)

    return exif_bytes


# def edit():
    # source = Image.open(path)
    # exif_dict = piexif.load(source.info['exif'])

    # Edit data
    # for attribute in exifs:
    # exif_dict['Exif'][piexif.ExifIFD.FocalLength]
    # exif_dict['Exif'][attribute] = exifs[attribute]


# For CLI
def main():
    # laodfiles:
    # print(f"{filtered}\n")
    # Tools.colprint(filtered)
    #if input("Confirm ? [y/N] ") not in ("y", "Y"):
    #    print("Aborting")
    #    sys.exit()

    return 0