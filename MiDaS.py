# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys        # Handle cli parameter
import getopt     # Handle named cli parameter
import os         # Handle platform independent paths
import time       # Measure execution time
import re         # Utilize RegEx
from PIL import Image

import MiDaS
import Tools      # Custom helpful functions

import numpy as np
# import png
import imageio

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

    # ToDo: use this where optimization can be done using filter and apply a funtion on every match?
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
    if _input is None:
        __Config.model = "MiDaS_small"
        return
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


# ToDo wip
# Convert and scale to a (16 bit) unsigned integer and normalize it
def __convert(byte_array, _type=np.uint16, _from=None, _to=None):
    print("########## __convert ##########")
    # Must save as 16 bit single channel image. unclear if integer or float
    # It ust be saved in gray scale with 1 channel to properly work later

    input_min = byte_array.min()
    input_max = byte_array.max()

    input_range = input_max - input_min
    # print("Input data:")
    # print("Input type: ", type(byte_array[0][0]))
    # print("Input min value: ", input_min)
    # print("Input max value: ", input_max)
    # print("Range: ", input_range)
    # print("\n")

    # print("Array: ")
    # print(f"len(byte_array): {len(byte_array)} \nlen(byte_array[0]): {len(byte_array[0])}")

    output_min = _from
    output_max = _to
    if _from is None:
        print("Why is _from none?", _from)
        output_min = Tools.limits(_type).min  # np.iinfo(_type).min
        output_max = Tools.limits(_type).max  # np.iinfo(_type).max

    output_range = output_max - output_min
    # print("Output data:")
    # print("Conversion Type: ", _type)
    # print("Type min value: ", limits(_type).min)
    # print("Type max value: ", limits(_type).max)
    # print("Range: ", output_range)
    # print(f"From {output_min} to {output_max}")
    # print("\n")

    # this might only work for [0; x] output range
    # does it work for [-5;5] and [-1;9] and [0-10] mapped mixed to same ranges but * 10:
    # [-5;5] => [-50; 50]; [-5;5] => [-10; 90]

    # factor to map [fmin, fmax] to [imin, imax]
    # input_value * mf = mapped_output_value
    mf = (output_range / input_range)

    # Sanity check
    _round = 6
    mapped_min = np.round((input_min - input_min) * mf, _round)
    mapped_max = np.round((input_max - input_min) * mf, _round)
    # print("Sanity Check:")
    # print(f"Factor: [{mf}]")
    # print(f"Round to {_round} decimal")
    # print(f"Scale/Map input [{input_min}] to output [{output_min}] \n "
    #       f"\t calculated input[{mapped_min}] == output[{output_min}] : {mapped_min == output_min}")
    # print(f"Scale/Map input [{input_max}] to output [{output_max}] \n"
    #       f"\t calculated output[{mapped_max}] == output[{output_max}] : {mapped_max == output_max}")
    if not (mapped_min == output_min):
        Tools.error_print(__convert.__name__, "Could not Map input min value to output min value")
        exit()
    if not (mapped_max == output_max):
        Tools.error_print(__convert.__name__, "Could not Map input max value to output max value")
        exit()

    print(f"converted {type(byte_array[0][0])} to {_type}")
    # ToDo: round here too?
    output = np.array((byte_array - input_min) * mf, _type)
    if type(output[0][0]) != _type:
        print("new array is not proper type...")
        print("should: ", _type)
        print("is: ", type(output[0][0]))
        exit()
    # Invert values
    # output = output_max - output
    return output


def __save_result(byte_array, name, path=""):
    new_file = f"z_{name}{__Config.output_file}.png"
    # plt.imsave(path + new, byte_array, cmap='gray', format="png")  # original resolution
    img = Image.fromarray(byte_array, "I")
    img.save(path + new_file)
    # imageio.imwrite(path + new_file, byte_array, "png")
    print(f"saved as: [{new_file}] under {path}")
    # print(f"Depth map saved under ./{subject}_z.png")


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
    print("Received following parameter:")
    print(options)
    __setup(options)
    for image in __Config.image_paths:
        depth_map = __generate_image(image)
        # ToDo convert image to 16 bit single channel gray scale
        __save_result(depth_map, os.path.split(image)[1].split('.')[0])


# ToDo; Wip fix border or discard completely
def __generate_scale_image(width, height, data_type, _range=None):
    if _range is None:
            _range = range(0, height, 1)
    _from = Tools.limits(data_type).min
    _to = Tools.limits(data_type).max
    # generate gradient image from 0 to image height with given step
    scale = np.zeros((height, width), data_type)
    border_width = 1
    border_value = 1
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for i in range(_range.start, _range.stop):
        print(i)
        row = np.ones((1, width), data_type) * (i * _range.step)
        row[0][-border_width:width] = np.ones((1, border_width), data_type) * border_value
        row[0][0:border_width] = np.ones((1, border_width), data_type) * border_value
        scale[i] = row
    scale[0:border_width] = np.ones((border_width, width), data_type) * border_value
    scale[-border_width:width] = np.ones((border_width, width), data_type) * border_value
    print(scale)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # exit()
    return scale


def dry():
    print("########## Dry run ##########")
    __Config.output_file = ""
    data_type = np.float32

    # Load/Generate depth map
    case = "hand"
    depth_map = None
    if case == "hand":
        depth_map = np.load('z_1_binary.npy')
    elif case == "scale":
        # scale = __generate_scale_image(3480, 4640, data_type)
        depth_map = __generate_scale_image(640, 480, data_type, range(0, 480, 2))
    elif case == "stool":
        img = Image.open("PointCloud/depth/00000.png")
        depth_map = np.array(img)
    elif case == "generate":
        options = {
            __Config.Model: "large",
            __Config.Output: "dry_gen",
            __Config.Images: "PointCloud/color/00000.jpg"
        }
        MiDaS.generate(options)
        exit()
    elif case == "mstool":
        img = Image.open("PointCloud/depth/z_00000.png")
        depth_map = np.array(img)
    else:
        print("No case in dry run in MiDaS")
        exit()

    print("depth map:")
    print(depth_map)

    # Here a quick interception to test
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    Q = 
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    __save_result(depth_map, 'dry', "PointCloud/depth/")


    return
    # End of interception
    # convert
    # output16 = __convert(depth_map)
    scaled = __convert(depth_map, np.uint32, 0, 100)
    print("converted depth map:")
    print(scaled)

    # Show result
    # plt.imshow(scale, cmap='gray')
    # plt.show()

    # Save result ot file [z_dry{__Config.output_file}.png]
    __save_result(scaled, 'dry', "PointCloud/depth/")


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


if __name__ == '__main__':
    __cli_main()
