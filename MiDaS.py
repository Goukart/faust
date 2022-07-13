import sys  # Handle cli parameter
import getopt  # Handle named cli parameter
import os  # Handle platform independent paths
import time  # Measure execution time
import re  # Utilize RegEx
import Tools  # Custom helpful functions

# MiDaS dependencies
import cv2
import torch


# ToDo: build in fail-safes in every functions, check parameter for correctness, so functions can be used freely

# ToDo: Final cleanup
#############################################################################
#
#           This Module sole purpose is to generate a depth map
#           Any additional functionalities should be removed
#           Load one or many files -> generate -> export results?
#
#############################################################################


# Load all files as paths into an array
# ToDo wip
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

    # ToDo: use this where optimization can be done using filter and apply a funktion on every match?
    # if so using maps and filter together maybe what i looked for
    images = []
    for file in files:
        if regex.match(file):
            images.append(os.path.join(path, file))

    if len(images) < 1:
        print(f"Could not match [{regex.pattern}] on files in [{path}]:")
        Tools.colprint(files)
        exit(-1)

    print("Loading files:\n")
    Tools.colprint(images)
    print("Confirm selection ? [y/N] ")

    if input().lower() in ("y", "yes"):
        print("Confirmed.")
    else:
        print("Aborting.")
        exit()

    return images


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
def __generate_depth_map(original_image, _model):
    midas = torch.hub.load("intel-isl/MiDaS", _model)

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
    # time execution:
    start = time.time()

    # Load transforms to resize and normalize the image for large or small model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if _model == "DPT_Large" or _model == "DPT_Hybrid":
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


def generate_dms(_images_regex, _model, _out=None):
    """
    :param _images_regex: Regular expression to "select" images
    :param _model: Which model MiDaS uses
    :param _out: Name of the output file
    :return: Generates depth maps and returns them as byte arrays in a dictionary with the name as the key
    """
    output_prefix = "z_"

    # Load and process images
    model = __select_model(_model)
    # Load files that match expression into array
    images_as_paths = __load_files(_images_regex)

    output = f"{_out} -> {output_prefix}<original_file_name>" if _out is None else _out
    # output = lambda o: f"{o} -> {output_prefix}<original_file_name>" if o is None else o
    print(f"\n___________________________________________________________________\n"
          f"All given parameters:\n"
          f"RegEx:\t{_images_regex}\n"
          f"Images:\t{images_as_paths}\n"
          f"Output:\t{output}\n"
          f"Model:\t{_model}\n"
          f"___________________________________________________________________\n")
    # Generate each image
    i = 0
    depth_maps = {}
    for image in images_as_paths:
        depth_map = __generate_depth_map(image, model)
        # ToDo convert image to 16 bit single channel gray scale
        if _out is None:
            name = f"{output_prefix}{os.path.split(image)[1].split('.')[0]}"
        else:
            name = f"{_out}_{i}"
            i += 1

        print("image: ", image)
        print(f"generated image with name/key: [{name}]")
        # Tools.export_bytes_to_image(depth_map, name)
        depth_maps[name] = depth_map

    return depth_maps


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

    generate_dms(images_regex, model, output_name)


if __name__ == '__main__':
    __cli_main()
