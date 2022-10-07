import sys  # Handle cli parameter
import getopt  # Handle named cli parameter
import os  # Handle platform independent paths
import time  # Measure execution time
import re  # Utilize RegEx
from PIL import Image
import numpy as np

import modules.Tools as Tools  # Custom helpful functions

# MiDaS dependencies
import cv2
import torch

# Free cache
from GPUtil import showUtilization as gpu_usage
# from numba import cuda

# ToDo build in fail-safes in every functions, check parameter for correctness, so functions can be used freely

# ToDo Final cleanup
#############################################################################
#
#           This Module sole purpose is to generate a depth map
#           Any additional functionalities should be removed
#           Load one or many files -> generate -> export results?
#
#############################################################################


def free_gpu_cache():
    # print("Initial GPU Usage")
    # gpu_usage()

    torch.cuda.empty_cache()

    # print("GPU Usage after emptying the cache")
    # gpu_usage()


# Select model, default is small
def __select_model(_input):
    #######################################################################
    #      Load a model
    #      (see https://github.com/isl-org/MiDaS#Accuracy for an overview)
    #######################################################################
    model = None
    if _input.lower() in ("large", "dpt_large"):
        # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        model = "DPT_Large"
    elif _input.lower() in ("hybrid", "dpt_hybrid"):
        # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        model = "DPT_Hybrid"
    elif _input.lower() in ("small", "midas_small"):
        # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        model = "MiDaS_small"
    else:
        print("Unknown model, defaulting to small")
        model = "MiDaS_small"
    return model


# generate a single depth map from original image
# ToDo wip
def __generate_depth_map(_input, _size, _midas) -> np.array:
    # time execution:
    start = time.time()

    # Predict and resize to original resolution
    with torch.no_grad():
        prediction = _midas(_input)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    end = time.time()
    print("execution time: ", end - start)

    del _input
    return output


def generate_dms_list(_images: list, _model: str, _out: str = None) -> dict[str, np.ndarray]:
    """
    :param _images: All images to generate dm from in a list
    :param _model: Which model MiDaS uses
    :param _out: Name of the output file
    :return: Generates depth maps and returns them as byte arrays in a dictionary with the name as the key
    """
    output_prefix = "z_"
    time.sleep(3)
    print("CUDA available? ", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("CUDA not available ", torch.cuda.is_available())
        return {}

    # Load model
    # ToDo fix "Using cache found in /home/ben/.cache/torch/hub/intel-isl_MiDaS_master" printed two times
    model = __select_model(_model)
    midas = torch.hub.load("intel-isl/MiDaS", model)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas = midas.to(device)
    midas.eval()

    # Load transforms to resize and normalize the image for large or small model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if _model == "DPT_Large" or _model == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Generate each image
    i = 0
    depth_maps = {}
    for image in _images:
        if os.path.isdir(image) or not Image.open(image).format:
            print("Not an image, skipping")
            continue
        # Load image and apply transforms
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        size = img.shape[:2]

        # ToDo make multiprocessing (Performance?), so infinite images can be processed one after another
        # without filling memory and not releasing old memory
        depth_map = __generate_depth_map(input_batch, size, midas)
        if _out is None:
            name = f"{output_prefix}{os.path.split(image)[1].split('.')[0]}"
        else:
            name = f"{_out}_{i}"
            i += 1

        depth_maps[name] = depth_map

        # Free up space on GPU
        torch.cuda.empty_cache()

    return depth_maps


def generate_dms_regex(_images_regex: str, _model: str, _out: str = None) -> dict[str, np.ndarray]:
    """
    :param _images_regex: Regular expression to "select" images
    :param _model: Which model MiDaS uses
    :param _out: Name of the output file
    :return: Generates depth maps and returns them as byte arrays in a dictionary with the name as the key
    """
    # Load files that match expression into array
    images_as_paths = Tools.load_files(_images_regex)
    Tools.cli_confirm_files(images_as_paths)
    print("generate_dms in MiDaS is: ", type(images_as_paths))

    return generate_dms_list(images_as_paths, _model, _out)


# ToDo wip, changes some things need to test a little
def __cli_main():
    # Define script options, only for console
    options = [
        # [long option, option, input] options
        ["Help", 'h', False],
        ["Model", 'm', True],
        ["Output", 'o', True],
        ["Images", 'i', True]
    ]

    model = None
    output_name = None
    images_regex = None
    # Remove 1st argument from the list of command line arguments
    argument_list = sys.argv[1:]

    short_options = "".join([(row[1] + ":" if row[2] else row[1]) for row in options])
    long_options = [(row[0] + "=" if row[2] else row[0]) for row in options]

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
                for option in options:
                    print(option)
                exit()

            elif currentArgument in ("-m", "--Model"):
                # print(f"Loading MiDaS Model: [{currentValue}]")
                model = __select_model(currentValue)

            elif currentArgument in ("-o", "--Output"):
                print(f"Saving to file: [{currentValue}]")
                output_name = currentValue

            elif currentArgument in ("-i", "--Images"):
                images_regex = currentValue

            else:
                print("Option not implemented!")
                exit(-1)

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    generate_dms_regex(images_regex, model, output_name)


if __name__ == '__main__':
    __cli_main()
