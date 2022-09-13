import numpy as np
import time       # Measure execution time
import re
import sys
import os

import Tools


def columnify(iterable):
    # First convert everything to its str
    strings = [str(x) for x in iterable]  # repr seems to be for debugging
    # Now pad all the strings to match the widest
    widest = max(len(x) for x in strings)
    padded = [x.ljust(widest) for x in strings]
    return padded


def col_format(iterable, width=120):
    if len(iterable) < 1:
        return ""

    columns = columnify(iterable)
    colwidth = len(columns[0]) + 2
    per_line = (width - 4) // colwidth
    text = ""
    for i, column in enumerate(columns):
        text += column + '\t'
        if i % per_line == per_line - 1:
            text += '\n'
    text += '\n'
    return text


def error_print(function_name, error_message):
    print("---------------------------------------")
    # ToDo format so: ---- // <name> \\ ---- [...] ---- \\ <name> // ----
    print(f"in {function_name}")
    print(error_message)
    print("---------------------------------------")


def array_is_equal(array_1, array_2):
    n_array_1 = len(array_1)
    n_array_2 = len(array_2)

    # Guarding ifs
    if n_array_1 != n_array_2:
        error_print(array_is_equal.__name__, f"Array not same lengths: {n_array_1} ≠ {n_array_2}")
        return False

    # sort both the arrays
    array_1.sort()
    array_2.sort()

    # traverse each index of arrays
    for i in range(n_array_1):
        if (type(array_1[i]) in (list, np.ndarray)) and (type(array_2[i]) in (list, np.ndarray)):
            if not array_is_equal(array_1[i], array_2[i]):
                return False
        else:
            # for same index if the value in the sorted arrays are different return false
            if array_1[i] != array_2[i]:
                error_print(array_is_equal.__name__, f"Item at i={i} not equal: [{array_1[i]}] ≠ [{array_2[i]}]")
                return False

    # if none of the above conditions satisfied return true
    return True


# pass two functions and pit them against each other to see which one is faster
# ToDo, make variable parameter list?
def cmp_runtime(old, opt, parameter, n=10):
    times_old = 0
    times_opt = 0
    output = []
    for i in range(n):
        # old methode
        start = time.time()
        output.append(old(parameter))
        end = time.time()
        times_old += end - start

        # optimized methode
        start = time.time()
        output.append(opt(parameter))
        end = time.time()
        times_opt += end - start

    old = times_old / n
    print("average time: ", old)
    print("outputs:\n", output)
    opt = times_opt / n
    print("average time: ", opt)
    print("outputs:\n", output)

    # ToDo proper calculation of percenutal increase decrease
    if opt > old:
        quality = "slower"
        percent = (100 / old) * opt
    else:
        quality = "faster"
        percent = (100 / opt) * old
    print(f"optimization is {np.round(percent)}% {quality} than initial procedure")


def limits(data_type):
    value = data_type(1)
    if isinstance(value, (int, np.integer)):
        # print("int")
        return np.iinfo(data_type)
    elif isinstance(value, (float, np.floating)):
        # print("float")
        return np.finfo(data_type)
    else:
        message = f"Provided Type [{data_type}] does not match any integer or floating point."
        error_print(limits.__name__, message)
        exit()


# ToDo: which format works and does not lose resolution?
# ToDo save image as 16 bit single channel gray scale
def export_bytes_to_image(byte_array: np.ndarray, name: str, path: str = ""):
    # import png
    import imageio
    # from PIL import Image
    # import cv2
    # import matplotlib.pyplot as plt

    new_file = f"{name}.png"

    # Save image
    # With matplotlib
    # plt.imsave(path + new_file, byte_array, cmap='gray', format="png")  # original resolution

    # With PIL Image
    # img = Image.fromarray(byte_array, "I")
    # img.save(path + new_file)

    # With imageio
    imageio.imwrite(path + new_file, byte_array, "png")

    # With cv
    # cv2.imwrite(path + new_file, byte_array)

    print(f"saved as: [{new_file}] under {path}")
    # print(f"Depth map saved under ./{subject}_z.png")


def load_files(_pattern: str, _dir: str = None) -> list:
    """Load all files as Paths into array

    :param _pattern: [str]: All file that match this regular expression get selected.
    :param _dir: [str, optional]: Path in which all files will be searched. If this is not given, the '_pattern' argument wil be split into path and regex.
    :return: [list]: A list of all files with their path.
    """
    expression = None
    path = None
    # Handle path combined with expression and both given separately
    if _dir is None:
        # ToDo test if works on Windows, using '\' as path seperator
        parts = _pattern.split(os.path.sep)
        expression = parts[-1]
        path = os.path.sep.join(parts[:-1]) + os.path.sep
    else:
        expression = _pattern
        path = _dir

    if not os.path.exists(path):
        print(f"No such file or directory: '{path}'")
        return []

    regex = None
    # Check regular rexpression to be correct
    try:
        regex = re.compile(expression)
    except re.error:
        print("Expression not valid")
        # sys.exit(-1)
        return []
    if not regex.pattern:
        print("Expression empty")
        return []
    print(f"Loading all files matching expression [{regex.pattern}]\n")

    # ToDo: use this where optimization can be done using filter and apply a funktion on every match?
    # if so using maps and filter together maybe what i looked for
    matches = []
    for entry in os.listdir(path):
        if regex.match(entry):
            # ToDo: make it list all files or all paths to file, selectable
            # Image.open(path + file) must be done with the result
            matches.append(path + entry)
            # matches.append(entry)

    if len(matches) < 1:
        print(f"Could not match [{regex.pattern}] on files in [{path}]. No matches:")
        # sys.exit(-1)
        return []

    return matches


def cli_confirm_files(_list: list):
    print(f"Loading Files [{len(_list)}]: \n")
    print(Tools.col_format(_list))
    if input("Confirm ? [y/N] ").lower() not in ("y", "yes"):
        print("Aborting")
        sys.exit(-1)

    print("Confirmed.")
