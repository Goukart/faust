# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pprint
import sys          # Handle cli parameter
import getopt       # Handle named cli parameter
import os           # Handle platform independent paths
import time         # Measure execution time
import re           # Utilize RegEx
import Tools        # Custom helpful functions

import numpy as np
# import png

# MiDaS dependencies
import cv2
import torch
import matplotlib.pyplot as plt


# Options
_Model = "Model"
_Output = "Output"
_Images = "Images"


class MiDaS:
    options = [
        # [long option, option, input] options
        ["Help", 'h', False],
        [_Model, 'm', True],
        [_Output, 'o', True],
        [_Images, 'i', True]
    ]

    # Load all files as paths into an array
    def __load_files(self, expression):
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
            Tools.colprint(files)
            exit(-1)

        print("Loading files:\n")
        Tools.colprint(image_paths)
        print("Confirm selection ? [y/N] ")

        if input().lower() in ("y", "yes"):
            print("Confirmed")
        else:
            print("Aborting")
            exit()

        self.image_paths = image_paths

    # Select model, default is small
    def __select_model(self, _input):
        #######################################################################
        #      Load a model
        #      (see https://github.com/isl-org/MiDaS#Accuracy for an overview)
        #######################################################################
        if _input.lower() in ("large", "dpt_large"):
            # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
            self.model = "DPT_Large"
        elif _input.lower() in ("hybrid", "dpt_hybrid"):
            # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            self.model = "DPT_Hybrid"
        elif _input.lower() in ("small", "midas_small"):
            # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
            self.model = "MiDaS_small"
        else:
            print("Unknown model, defaulting to small")
            self.model = "MiDaS_small"

    # generate a single depth map from original image
    def __generate_image(self, original_image):
        model_type = self.model

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

    @staticmethod
    def __convert(byte_array):
        # Must save as 16 bit single channel image
        # plt.imsave(subject + "_z." + "png", output, cmap='gray', format="png") # original resolution
        # It ust be saved in gray scale with 1 channel to properly work later
        # print(f"Depth map saved under ./{subject}_z.png")

        output16 = np.array(byte_array * 4, np.uint16)
        plt.imshow(output16, cmap='gray')

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
        # output16 = np.array(byte_array)
        return output16

    def __save_result(self, byte_array, name):
        #################################################
        #      Show result
        #################################################
        # Show result
        # plt.imshow(byte_array, cmap='gray')

        # convert
        output16 = byte_array  # self.__convert(byte_array)
        plt.imshow(output16, cmap='gray')
        # ---

        new = f"z_{name}{self.output}.png"
        print(f"saved as: [{new}]")
        plt.show()

        plt.imsave(new, byte_array, cmap='gray', format="png")  # original resolution
        # image_pil.save(new)
        # ---------------------

    def __init__(self, options):
        self.image_paths = None
        self.model = None

        # Load files that, match given expression, into array
        self.__load_files(options.get(_Images))
        self.__select_model(options.get(_Model))

        # Set of the output file
        if options.get(_Output) is None:
            self.output = ""
        else:
            self.output = f"_{options.get(_Output)}"

    def generate(self):
        for image in self.image_paths:
            depth_map = self.__generate_image(image)
            self.__save_result(depth_map, os.path.split(image)[1].split('.')[0])


def main():
    # Remove 1st argument from the list of command line arguments
    argumentList = sys.argv[1:]

    options = MiDaS.options

    short_options = "".join([(row[1] + ":" if row[2] else row[1]) for row in options])
    long_options = [(row[0] + "=" if row[2] else row[0]) for row in options]

    # Load and handle parameter
    images = []
    parameter = {}
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
                parameter[_Model] = currentValue

            elif currentArgument in ("-o", "--Output"):
                print(f"Saving to file: [{currentValue}]")
                parameter[_Output] = currentValue

            elif currentArgument in ("-i", "--Images"):
                parameter[_Images] = currentValue

            else:
                print("Option not implemented!")
                exit(-1)

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    midas = MiDaS(parameter)
    midas.generate()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
