# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys        # Handle cli parameter
import getopt     # Handle named cli parameter
import os         # Handle platform independent paths
import time       # Measure execution time
import re         # Utilize RegEx
import Tools      # Custom helpful functions

import numpy as np
# import png

# MiDaS dependencies
import cv2
import torch
import matplotlib.pyplot as plt
# ToDo: build in fail safes in every funtions, check parameter for correctness, so functions can be used freely


class __Config:
    # Define script options
    Help = "Help"
    Model = "Model"
    Output = "Output"
    Images = "Images"

    options = [
        # [long option, option, input] options
        [Help, 'h', False],
        [Model, 'm', True],
        [Output, 'o', True],
        [Images, 'i', True]
    ]

    # Collecting and managing input parameters
    parameters = {
        Output: None,
        Images: None,
    }

    # Variables
    image_paths = None
    model = None
    output_file = None


# Load all files as paths into an array
def __load_files(expression):
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

    # ToDo: ius this where optimization can be con using filter and apply a funtion on every match?
    # if so using maps and filter together maybe what i looked for
    image_paths = []
    for file in files:
        if regex.match(file):
            image_paths.append(os.path.join(path, file))

    if len(image_paths) < 1:
        print(f"Could not match [{regex.pattern}] on files in [{path}]:")
        Tools.colprint(files)
        exit(-1)

    print("Loading files:\n")
    Tools.colprint(image_paths)
    print("Confirm selection ? [y/N] ")

    if input().lower() in ("y", "yes"):
        print("Confirmed.")
    else:
        print("Aborting.")
        exit()

    __Config.image_paths = image_paths


# Select model, default is small
def __select_model(_input):
    #######################################################################
    #      Load a model
    #      (see https://github.com/isl-org/MiDaS#Accuracy for an overview)
    #######################################################################
    if _input.lower() in ("large", "dpt_large"):
        # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        __Config.model = "DPT_Large"
    elif _input.lower() in ("hybrid", "dpt_hybrid"):
        # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        __Config.model = "DPT_Hybrid"
    elif _input.lower() in ("small", "midas_small"):
        # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        __Config.model = "MiDaS_small"
    else:
        print("Unknown model, defaulting to small")
        __Config.model = "MiDaS_small"


# generate a single depth map from original image
# ToDo wip
def __generate_image(original_image):
    model_type = __Config.model

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


# ToDo: don't change parameter (pass copy not reference)
# Funny image if I sort the array
def _max(array):
    copy = array.copy()
    copy[-1].sort()
    return copy[-1][-1]


def _min(array):
    copy = array.copy()
    copy[0].sort()
    return copy[0][0]


# ToDo wip
def __convert(byte_array):
    # Must save as 16 bit single channel image
    # plt.imsave(subject + "_z." + "png", output, cmap='gray', format="png") # original resolution
    # It ust be saved in gray scale with 1 channel to properly work later
    # print(f"Depth map saved under ./{subject}_z.png")

    output16 = np.array(byte_array, np.uint16)
    # output16 = np.array(byte_array)

    print("extremes")
    # print("what is faster")
    # Tools.cmp_runtime(_min, np.min, byte_array)
    minimum = byte_array.min()
    maximum = byte_array.max()
    print("min", minimum)
    print("max", maximum)

    print("Array: ")
    print(f"len(byte_array): {len(byte_array)} \nlen(byte_array[0]): {len(byte_array[0])}")

    # map [fmin; fmax] to [imin-imax]
    fmax = maximum
    fmin = minimum

    # print(np.iinfo(output16[0][0].dtype).max)
    print(type(byte_array[0][0]))
    imax = 65535
    imin = 0

    factor = (imax - imin) / ((fmax - fmin))
    # data = np.array([np.arange(fmin, fmax + 1, 1.0), np.arange(fmin, fmax + 1, 1.0), np.arange(fmin, fmax + 1, 1.0)], dtype=np.float32)
    # dataD = np.array(dataD, dtype=np.uint16)

    # offset so ot goes from 0 to max
    test = byte_array - minimum
    # recalculate max to be 65535
    test = test * (maximum / imax)

    # this might only work for 0-x putput range, does it work for [-5;5] and [-1;9] and [0-10] mapped mixed to same * 10
    # determine range of input values
    input_range = imax - imin
    output_range = fmax - fmin

    print("ranges: ")
    print("input: ", input_range)
    print("output: ", output_range)
    # mapping factor
    mf = (input_range / output_range)
    print("mf: ", mf)
    print("int scaled min", ((fmin - fmin) * mf))  # * or / and why
    print("int scaled max", ((fmax - fmin) * mf))
    calc = (byte_array - fmin) / mf
    return calc


def __save_result(byte_array, name, path=""):
    new = f"z_{name}{__Config.output_file}.png"
    plt.imsave(path + new, byte_array, cmap='gray', format="png")  # original resolution
    print(f"saved as: [{new}]")


def __setup(parameter):
    # Load files that, match given expression, into array
    # ToDo: here make sure those check input parameter
    __load_files(parameter.get(__Config.Images))
    __select_model(parameter.get(__Config.Model))

    # Set of the output file
    if parameter.get(__Config.Output) is None:
        __Config.output = ""
    else:
        __Config.output = f"_{parameter.get(__Config.Output)}"


def generate(options):
    __setup(options)
    for image in __Config.image_paths:
        depth_map = __generate_image(image)
        # ToDo convert image to 16 bit single channel gray scale
        __save_result(depth_map, os.path.split(image)[1].split('.')[0])


def dry():
    print("### Dry run ###")

    __Config.output_file = ""

    # Load working depth map
    import imageio
    img = imageio.imread('PointCloud/depth/00000.png')
    redwood = np.asarray(img)

    depth_map = np.load('z_1_binary.npy')

    # convert
    output16 = __convert(depth_map)
    width = 640  # 3480
    height = 480  # 4640
    scale = np.zeros((height, width), np.uint16)
    for i in range(height):
        scale[i] = np.ones((1, width), np.uint16) * i

    #for i in range(len(output16)):
    #    for j in range(len(output16[0])):
    #        output16[i][j] = 0
    print(scale)
    #print(output16)

    # Show result
    plt.imshow(scale, cmap='gray')
    plt.show()

    __save_result(scale, 'dry', "PointCloud/depth/")


def __cli_main():
    # Remove 1st argument from the list of command line arguments
    argument_list = sys.argv[1:]

    short_options = "".join([(row[1] + ":" if row[2] else row[1]) for row in __Config.options])
    long_options = [(row[0] + "=" if row[2] else row[0]) for row in __Config.options]

    # Load and handle parameter
    images = []
    parameter = {}
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argument_list, short_options, long_options)

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
                for option in __Config.options:
                    print(option)
                exit()

            elif currentArgument in ("-m", "--Model"):
                # print(f"Loading MiDaS Model: [{currentValue}]")
                parameter[__Config.Images] = currentValue

            elif currentArgument in ("-o", "--Output"):
                print(f"Saving to file: [{currentValue}]")
                parameter[__Config.Output] = currentValue

            elif currentArgument in ("-i", "--Images"):
                parameter[__Config.Images] = currentValue

            else:
                print("Option not implemented!")
                exit(-1)

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    generate(parameter)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    __cli_main()
