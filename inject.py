#! /usr/bin/python3

# requires
# pip install exif

import os
# import io
import re
import sys

input_dir = "./tmp/render_out/"
output_dir = "./injection_out/"


# ToDo clear output folder

def main(args):
    donor = "injectionStuff/raw.jpg"

    if len(args) > 3:
        print("Too many arguments given")
        sys.exit()
    elif len(args) < 2:
        print("No expression given")
        sys.exit()

    # extract real exif data
    exif_bytes = extract_exif(donor)

    # inject exif data into render images
    from PIL import Image
    # begin at 1, because first argument is the script itself
    files = load_files(args[1])
    for file in files:
        # edit_piexif(file)
        out = Image.open(input_dir + file)
        # out.save('_%s' % file, "jpeg", exif=exif_bytes)
        out.save(output_dir + file, "jpeg", exif=exif_bytes)


def edit_exif(file):
    from exif import Image
    # only works with jpg, so: mogrify -format jpg *.png
    # https://gitlab.com/TNThieding/exif/-/issues/36
    # https://www.lifewire.com/convert-linux-command-unix-command-4097060
    with open(f"./{file}", "rb") as loaded_file:
        image = Image(loaded_file)

    # Debug?
    if image.has_exif:
        status = f"contains EXIF (version {image.exif_version}) information."
    else:
        status = "does not contain any EXIF information."
    print(f"Image {status}")

    print(image)
    # image_subject = dir(image)

    # Set meta data
    image.set("make", "Blender Render")
    # image.set("image_height", "1080")
    # image.set("image_width", "1920")
    image.set("f_number", "2.8")
    # image.set("exposure_time", "Blender")  #: 0.058823529411764705 # -> 1/17 s
    image.set("focal_length", "50")
    # image.set("orientation", "Orientation.TOP_LEFT")
    image.set("aperture_value", "2.8")

    print(f"Image {file} contains {len(image.list_all())} members:")
    print(f"{image.list_all()}\n")

    # Write to file
    with open(file, 'wb') as image_file:  # with open('modified_render.jpg', 'wb') as new_image_file:
        image_file.write(image.get_file())


def edit_piexif(file):
    import piexif
    from PIL import Image
    zeroth_ifd = {piexif.ImageIFD.Make: u"Canon",
                  piexif.ImageIFD.XResolution: (96, 1),
                  piexif.ImageIFD.YResolution: (96, 1),
                  piexif.ImageIFD.Software: u"piexif"
                  }
    exif_ifd = {piexif.ExifIFD.DateTimeOriginal: u"2099:09:29 10:10:10",
                piexif.ExifIFD.LensMake: u"LensMake",
                piexif.ExifIFD.Sharpness: 65535,
                piexif.ExifIFD.LensSpecification: ((1, 1), (1, 1), (1, 1), (1, 1)),
                }
    gps_ifd = {piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
               piexif.GPSIFD.GPSAltitudeRef: 1,
               piexif.GPSIFD.GPSDateStamp: u"1999:99:99 99:99:99",
               }
    first_ifd = {piexif.ImageIFD.Make: u"Canon",
                 piexif.ImageIFD.XResolution: (40, 1),
                 piexif.ImageIFD.YResolution: (40, 1),
                 piexif.ImageIFD.Software: u"piexif"
                 }

    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": first_ifd}
    exif_bytes = piexif.dump(exif_dict)
    im = Image.open(file)
    im.save("out.jpg", exif=exif_bytes)


def load_files(_pattern):
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
    print(f"{filtered}\n")
    if input("Confirm ? [y/N] ") not in ("y", "Y"):
        print("Aborting")
        sys.exit()
    return filtered


def extract_exif(donor):
    import piexif
    from PIL import Image
    # Set meta data
    exifs = {
        piexif.ExifIFD.FNumber: (28000, 10000),
        # piexif.ExifIFD.FocalLength: (5000, 100),
        piexif.ExifIFD.ApertureValue: (280, 100),

    }

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


def remove_exif(file):
    import piexif
    piexif.remove(file)


def old(file):
    from exif import Image
    # from PIL import Image as pil
    with open(f"./{file}", "rb") as loaded_file:
        image = Image(loaded_file)

    # Debug?
    if image.has_exif:
        status = f"contains EXIF (version {image.exif_version}) information."
    else:
        status = "does not contain any EXIF information."
    print(f"Image {status}")

    print(image)
    # image_subject = dir(image)
    # print("dir()", image_subject)

    print(f"Image {file} contains {len(image.list_all())} members:")
    print(f"{image.get_all()}\n")

    exif = image.list_all()

    dict = image.get_all().items()

    with open(f"./out.png", "rb") as loaded_file:
        image = Image(loaded_file)

    for attribute, value in dict:
        image.set(attribute, value)
        # print(f"attr: {attribute}; val: {value}")

    # Write to file
    with open("out2.png", 'wb') as image_file:
        image_file.write(image.get_file())

    # im = pil.open(file)
    # im.save("out.jpg", exif=exif)

    return exif
# ToDo test other libs how they can handle exif data (PIL, exif, piexif)


if __name__ == '__main__':
    main(sys.argv)
