import numpy as np
import time       # Measure execution time
import re
import sys
import os


def columnify(iterable):
    # First convert everything to its str
    strings = [str(x) for x in iterable]  # repr seems to be for debugging
    # Now pad all the strings to match the widest
    widest = max(len(x) for x in strings)
    padded = [x.ljust(widest) for x in strings]
    return padded


def colprint(iterable, width=120):
    columns = columnify(iterable)
    colwidth = len(columns[0]) + 2
    perline = (width - 4) // colwidth
    for i, column in enumerate(columns):
        print(column, end='\t')
        if i % perline == perline - 1:
            print('\n', end='')
    print("\n")


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


def load_files(_expression: str) -> list:
    parts = os.path.split(_expression)
    path = "/".join(parts[:-1])  # ToDo not platform independent
    _expression = parts[-1]
    print("parts: ", parts)
    print("path: ", path)
    print("actual regex: ", parts[-1])
    print("All files on dir:")
    print(os.listdir(path))
    # ToDo cheok if it works with just a expression too

    regex = None
    try:
        regex = re.compile(_expression)
    except re.error:
        print("Expression not valid")
        sys.exit()
    print(f"Loading all files matching expression [{regex.pattern}]\n")

    selection = []
    for entry in os.listdir(path):
        if regex.match(entry):
            selection.append(entry)

    print(f"Loading Files [{len(selection)}]: ")
    print(f"{selection}\n")
    # colprint(filtered)
    if input("Confirm ? [y/N] ") not in ("y", "Y"):
        print("Aborting")
        sys.exit()

    return selection