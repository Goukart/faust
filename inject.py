#! /usr/bin/python3

# requires
# pip install exif

import Tools

# Exif stuff
import piexif
from PIL import Image


# ToDo test other libs how they can handle exif data (PIL, exif, piexif)
# ToDo clear output folder


# ToDo only uses one source file not a list, yet
def inject_exif(exif_bytes: bytes, files: list) -> int:
    if not exif_bytes or not files:
        print("Parameter not valid:")
        print("exif_bytes :", exif_bytes)
        print("files:", files)
        return -1
    for file in files:
        # edit_piexif(file)
        out = Image.open(file)
        # out.save('_%s' % file, "jpeg", exif=exif_bytes)
        out.save(file, "png", exif=exif_bytes)
    return 0


def extract_exif(source: str) -> bytes:
    # Extract source exif data
    source = Image.open(source)
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
# call: python inject.py ".*.jpg"
def main():
    # laodfiles:
    # print(f"{filtered}\n")
    # Tools.colprint(filtered)
    #if input("Confirm ? [y/N] ") not in ("y", "Y"):
    #    print("Aborting")
    #    sys.exit()

    source = ""  # ToDo get from argument
    regex = ""  # ToDo get from argument
    # extract real exif data
    exif_bytes = extract_exif(source)

    # inject exif data into render images
    files = Tools.load_files(regex)

    Tools.cli_confirm_files(files)

    return 0