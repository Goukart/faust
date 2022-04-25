# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pprint
import sys          # Handle cli parameter
import getopt       # Handle named cli parameter
import os           # Handle platform independent paths
import time         # Measure execution time
import re           # Utilize RegEx

# MiDaS dependencies
import cv2
import torch
import matplotlib.pyplot as plt

# Image handling
import PIL
from PIL import Image


def columnify(iterable):
    # First convert everything to its str
    strings = [str(x) for x in iterable]  # repr seems to be for debugging
    # Now pad all the strings to match the widest
    widest = max(len(x) for x in strings)
    padded = [x.ljust(widest) for x in strings]
    return padded


def colprint(iterable, width=120):
    columns = columnify(iterable)
    colwidth = len(columns[0])+2
    perline = (width-4) // colwidth
    for i, column in enumerate(columns):
        print(column, end='\t')
        if i % perline == perline-1:
            print('\n', end='')
    print("\n")


def error_print(function_name, error_message):
    print("---------------------------------------")
    # ToDo format so: ---- // <name> \\ ---- [...] ---- \\ <name> // ----
    print(f"in {function_name}")
    print(error_message)
    print("---------------------------------------")


def array_is_equal(array_1, array_2):
    # if the length of arrays are different return false
    n_array_1 = len(array_1)
    n_array_2 = len(array_2)
    if n_array_1 != n_array_2:
        error_print(array_is_equal.__name__, f"Array not same lengths: {n_array_1} ≠ {n_array_2}")
        return False
    else:
        # sort both the arrays
        array_1.sort()
        array_2.sort()
        # traverse each index of arrays
        for i in range(n_array_1):
            # for same index if the value in the sorted arrays are different return false
            if array_1[i] != array_2[i]:
                error_print(array_is_equal.__name__, f"Item at i={i} not equal: [{array_1[i]}] ≠ [{array_2[i]}]")
                return False

    # if none of the above conditions satisfied return true
    return True


def load_files(expression):
    # ToDo test if works on Windows, using '\' as path seperator
    parts = os.path.split(expression)
    regex = ""
    try:
        regex = re.compile(parts[1])
    except re.error:
        print("Non valid regex pattern")
        exit(-1)

    path = parts[0]
    files = os.listdir(path)

    image_paths = []
    for file in files:
        if regex.match(file):
            image_paths.append(os.path.join(path, file))

    if len(image_paths) < 1:
        print(f"Could not match [{regex.pattern}] on files in [{path}]:")
        colprint(files)
        exit(-1)

    print("Loading files:\n")
    colprint(image_paths)
    print("Confirm selection ? [y/N] ")

    if input().lower() in ("y", "yes"):
        print("Confirmed")
    else:
        print("Aborting")
        exit()

    return image_paths


def select_model(_input):
    #######################################################################
    #      Load a model
    #      (see https://github.com/isl-org/MiDaS#Accuracy for an overview)
    #######################################################################
    if _input.lower() in ("large", "dpt_large"):
        # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        return "DPT_Large"
    elif _input.lower() in ("hybrid", "dpt_hybrid"):
        # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        return "DPT_Hybrid"
    elif _input.lower() in ("small", "midas_small"):
        # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        return "MiDaS_small"
    else:
        print("Unknown model, defaulting to small")
        return "MiDaS_small"


def generate_image(original_image, model_type):

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    #######################################################################
    #
    #      Move model to GPU if available
    #
    #######################################################################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    #######################################################################
    #
    #      Generate Depth Map
    #
    #######################################################################
    # time exectution:
    start = time.time()

    # Load transforms to resize and normalize the image for large or small model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Load image and apply transforms
    img = cv2.imread(original_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Predict and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    end = time.time()
    print("execution time: ", end - start)
    return output


def save_result(byte_array, name):
    #################################################
    #      Show result
    #################################################
    plt.imshow(byte_array, cmap='gray')
    # plt.savefig(subject + "." + "png", dpi=680)

    # Must save as 16 bit single channel image
    # plt.imsave(subject + "_z." + "png", output, cmap='gray', format="png") # original resolution
    # It ust be saved in gray scale with 1 channel to properly work later
    # print(f"Depth map saved under ./{subject}_z.png")

    import numpy as np
    # import png

    print("extremes")
    mins = []
    for list in byte_array:
        mins.append(min(list))
    print(min(mins))
    maxes = []
    for list in byte_array:
        maxes.append(max(list))
    print(max(maxes))

    print("Array: ")
    print(f"len(output): {len(byte_array)} \nlen(output[0]): {len(byte_array[0])}")

    # map [fmin; fmax] to [imin-imax]
    fmax = min(mins)
    fmin = max(maxes)

    imax = 3000
    imin = 0

    factor = (imax - imin) / ((fmax - fmin))
    # data = np.array([np.arange(fmin, fmax + 1, 1.0), np.arange(fmin, fmax + 1, 1.0), np.arange(fmin, fmax + 1, 1.0)], dtype=np.float32)
    # dataD = np.array(dataD, dtype=np.uint16)

    x = (byte_array - fmin) * factor

    output16 = np.array(byte_array, np.uint16)
    image_pil = PIL.Image.fromarray(output16)
    # image_pil = PIL.Image.fromarray(byte_array)
    # print(image_pil)
    plt.imshow(image_pil)
    new = name + '_z-tmp.png'
    print(f"saved as: [{new}]")
    image_pil.save(new)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Remove 1st argument from the list of command line arguments
    argumentList = sys.argv[1:]

    # dict = {
    #     "Help": 'h',
    #     "Model=": 'm:',
    #     "Output=": 'o:',
    #     "Images=": 'i:'
    # }
    options = [
        # [long option, option, input] options
        ["Help", 'h', False],
        ["Model", 'm', True],
        ["Output", 'o', True],
        ["Images", 'i', True]
    ]

    # Options
    # options = "hm:o:"
    # a_short_options = ''.join(dict.values())
    short_options = "".join([(row[1] + ":" if row[2] else row[1]) for row in options])

    # Long options
    # long_options = ["Help", "Output=", "Mega=", "Model="]
    # a_long_options = list(dict)
    long_options = [(row[0] + "=" if row[2] else row[0]) for row in options]

    # print(f"    is: {short_options}")
    # print("should: " + a_short_options)
    # print("\n\n")

    # if array_is_equal(a_long_options, long_options):
    #     print("equal")
    # else:
    #     print("ALARM, long_options")

    # if a_short_options == short_options:
    #     print("equal")
    # else:
    #     print("ALARM, short_options")


    # if not array_is_equal(list(dict), long_options):
    #     print("ALARM!")
    #     exit(-1)

    # Load and handle parameter
    model_input = ""
    images = []
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, short_options, long_options)

        # ToDo change to dynamic checking from dict, not manually expanding else ifs
        # Check if required options are set
        # if ("-i", "--Images") not in arguments:
        #     print(arguments)
        #     print("Image is required")
        #     exit(-1)

        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--Help"):
                print("Displaying Help")
                # ToDo actually helpful help, with minimal manual labor
                for option in options:
                    print(option)
                exit()

            elif currentArgument in ("-m", "--Model"):
                # print(f"Loading MiDaS Model: [{currentValue}]")
                model_input = currentValue

            elif currentArgument in ("-o", "--Output"):
                print(f"Saving to file: [{currentValue}]")

            elif currentArgument in ("-i", "--Images"):
                print(f"Try loading file")
                images = load_files(currentValue)

            else:
                print("Option not implemented!")
                exit(-1)

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    model = select_model(model_input)

    for image in images:
        image_name = os.path.split(image)[1].split('.')[0]
        save_result(generate_image(image, model), image_name)
