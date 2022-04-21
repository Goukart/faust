# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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


def load_files(file_path):
    # ToDo Make this a script with parameter in console, also select images by regex
    # ToDo optional load from internet? maybe later in finished product
    # Download an image from the PyTorch homapage
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # urllib.request.urlretrieve(url, filename)
    # ToDo test if works on Windows

    image_formats = ("jpg", "png")
    path_parts = os.path.split(file_path)

    subject = path_parts[1]
    path = path_parts[0]

    image_paths = []
    filename = ""

    for root, dirs, files in os.walk(path):
        for name in files:
            name_parts = name.split(".")
            if name_parts[0] == subject and name.endswith(image_formats):
                file_type = name_parts[1]
                filename = os.path.join(path, subject + '.' + file_type)

    if filename == "":
        print(f"Could not find [{subject}] in [{path}]")
        exit(-1)

    print("Loading file: " + filename)

    image_paths.append(filename)
    return image_paths


def test_regex(expression):
    parts = os.path.split(expression)
    regex = ""
    try:
        regex = re.compile(parts[1])
    except re.error:
        print("Non valid regex pattern")
        exit(-1)

    path = parts[0]
    files = os.listdir(path)

    # lambda file: f"oop {file}",
    # image_paths = list(filter(regex.match, files))
    # ToDo eine funktion die jeden match erst bearbeitet und dann in eine neue liste packt
    image_paths = []
    for file in files:
        if regex.match(file):
            image_paths.append(os.path.join(path, file))

    # print(list(filter(lambda x: x[0].lower() in 'aeiou', creature_names)))

    # matches = list(filter(regex.match, files))
    # print(f"Parameter: {expression}")
    # print(f"RegEx: '{regex.pattern}'")
    # print(f"Files: {files}")
    # print(f"Matches: {matches}")

    # ToDo show selection and ask for confirmation

    if len(image_paths) < 1:
        print(f"Could not match [{regex}] in [{path}], {files}")
        exit(-1)

    print(f"Loading files: {image_paths}")
    return image_paths


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

    output16 = np.array(byte_array * 4, np.uint16)
    image_pil = PIL.Image.fromarray(output16)
    # print(image_pil)
    plt.imshow(image_pil)
    new = name + '_z-tmp.png'
    print(f"saved as: [{new}]")
    image_pil.save(new)


def array_is_equal(array_1, array_2):
    # if the length of arrays are different return false
    n_array_1 = len(array_1)
    n_array_2 = len(array_2)
    if n_array_1 != n_array_2:
        print(f"Array not same lengths: {n_array_1} ≠ {n_array_2}")
        return False
    else:
        # sort both the arrays
        array_1.sort()
        array_2.sort()
        # traverse each index of arrays
        for i in range(n_array_1):
            # for same index if the value in the sorted arrays are different return false
            if array_1[i] != array_2[i]:
                print(f"{array_1[i]} ≠ {array_2[i]}")
                return False

    # if none of the above conditions satisfied return true
    return True


def select_model(_input):
    #######################################################################
    #      Load a model
    #      (see https://github.com/isl-org/MiDaS#Accuracy for an overview)
    #######################################################################
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    if _input.lower() in ("large", "dpt_large"):
        # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        model_type = "DPT_Large"
    elif _input.lower() in ("hybrid", "dpt_hybrid"):
        # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        model_type = "DPT_Hybrid"
    else:
        return model_type

    return model_type


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print("The name of this script is %s" % (sys.argv[0]))
    # args = len(sys.argv) - 1
    # print("The script was called with %i arguments" % args)

    # Remove 1st argument from the list of command line arguments
    argumentList = sys.argv[1:]

    dict = {
        "Help": 'h',
        "Model=": 'm:',
        "Output=": 'o:',
        "Images=": 'i:',
        "RegEx=": 'r:'
    }

    # Options
    # options = "hm:o:"
    options = ''.join(dict.values())

    # Long options
    # long_options = ["Help", "Output=", "Mega=", "Model="]
    long_options = list(dict)

    # if not array_is_equal(list(dict), long_options):
    #     print("ALARM!")
    #     exit(-1)

    # python MiDaS.py -o 123 --Model large -i images/4

    # Load and handle parameter
    model_input = ""
    images = []
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)

        # checking each argument
        # ToDo change to dynamic checking from dict, not manually expanding else ifs
        # Todo make sure to check required parameter
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--Help"):
                print("Displaying Help")
                # ToDo actually helpful help, with minimal manual labor
                for option in dict:
                    print(option)
                exit()

            elif currentArgument in "--Mega":
                print(f"Loading Mega: [{currentValue}]")

            elif currentArgument in ("-m", "--Model"):
                # print(f"Loading MiDaS Model: [{currentValue}]")
                model_input = currentValue

            elif currentArgument in ("-o", "--Output"):
                print(f"Saving to file: [{currentValue}]")

            elif currentArgument in ("-i", "--Images"):
                # print(f"Try loading file")
                images = load_files(currentValue)

            elif currentArgument in ("-r", "--RegEx"):
                print(f"Test RegEx")
                test_regex(currentValue)

            else:
                print("Option not implemented!")
                exit(-1)

    except getopt.error as err:
        # output error, and return with an error code
        # print("Invalid argument")
        print(str(err))

    #model = select_model(model_input)

    #for image in images:
    #    image_name = os.path.split(image)[1].split('.')[0]
    #    save_result(generate_image(image, model), image_name)
