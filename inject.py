#! /usr/bin/python3

# requires
# pip install exif

import os
# import io
import re
import sys
import Tools

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
    for file in files:
        # edit_piexif(file)
        out = Image.open(input_dir + file)
        # out.save('_%s' % file, "jpeg", exif=exif_bytes)
        out.save(output_dir + file, "jpeg", exif=exif_bytes)

    return 0


# ToDo move to Tools module, fuse with MiDaS version
def load_files(_pattern: str) -> list:
    regex = None
    try:
        regex = re.compile(_pattern)
    except re.error:
        print("Expression not valid")
        sys.exit()
    print(f"Loading all files matching expression [{regex.pattern}]\n")

    filtered = []
    for entry in os.listdir(input_dir):
        if regex.match(entry):
            filtered.append(entry)

    print(f"Loading Files [{len(filtered)}]: ")
    # print(f"{filtered}\n")
    Tools.colprint(filtered)
    if input("Confirm ? [y/N] ") not in ("y", "Y"):
        print("Aborting")
        sys.exit()

    return filtered


def extract_exif(donor):
    import piexif
    from PIL import Image

    # Load source exif
    source = Image.open(donor)
    exif_dict = piexif.load(source.info['exif'])

    # Edit data
    #for attribute in exifs:
        #exif_dict['Exif'][attribute] = exifs[attribute]
    print(exif_dict['Exif'][piexif.ExifIFD.FocalLength])
    # sys.exit()

    # Inject into target file
    exif_bytes = piexif.dump(exif_dict)
    # out = Image.open("out.jpg")
    # # out.save('_%s' % file, "png", exif=exif_bytes)
    # out.save("out.jpg", "jpeg", exif=exif_bytes)

    return exif_bytes
