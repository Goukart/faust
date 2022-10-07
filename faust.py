import open3d as o3d
import modules.Tools as Tools      # Custom helpful functions
# import Test
import numpy as np
from matplotlib import pyplot as plt
import subprocess
import sys


# Orchestrate all the Modules
# ToDo: pipe.sh should convert all input images to png

# ToDo wip
# Convert and scale to a (16 bit) unsigned integer and normalize it
import modules.inject as inject
from modules.GUI import Gui


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


depth_images = "PointCloud/depth/"
color_images = "PointCloud/color/"
MICMAC = "mm_out"


def photogrammetry(regex, resolution, output, skip=0):
    """
    This is a wrapper for the actual micmac shell script
    :param regex: Regular expression to select images
    :param resolution: resolution to which each image is down sampled
    :param output: Filename of the point cloud
    :param skip: Step to skip to
    :return: Script output
    """
    # os.popen or os.system runs the given scrip in the current directory NOT the one where the script is in
    prepare = f'cd ./{MICMAC}'
    command = f'./micmac.sh imageRegex="{regex}" resolution={resolution} output="{output}" skipTo={skip}'
    return subprocess.run([f'{prepare} && {command}'], shell=True).returncode


def test(s: str) -> str:
    # requires string returns string
    s = s + "_con"
    return s


def service_pc():
    # pc = PointCloud()
    # pc.call()
    # MiDaS.generate("./images/1", "large", "tmp")

    ####################################################################
    #
    #           Test to see what datatype the depth map has
    #
    ####################################################################
    # quick_dm("l2")
    from PIL import Image


    print("Test custom point cloud from dm")
    # Load image (with PIL)as np array
    image_array = np.asarray(Image.open("PointCloud/depth/z_l1.png"))
    print(image_array.shape)
    print(image_array)
    # plotting
    plt.imshow(image_array, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()
    sys.exit()

    ####################################################################
    #
    #           Photogrammetry
    #
    ####################################################################
    # ToDo: very fragile, idk why it breaks sometimes and sometimes not
    # change to alternative suit? COLMAP, Agisoft, others? maybe not Python?
    # print(photogrammetry(".*.jpg", 1000, "hand"))

    ####################################################################
    #
    #           Generate Depth Map
    #
    ####################################################################

    ####################################################################
    #
    #           Create Point Cloud
    #
    ####################################################################
    # must also reduce points, high-res images have way too many points, convert to layers maybe?

    # color = imageio.imread(f"{color_images}{case}.jpg")
    # Test.depth_map_to_point_cloud("l1")
    Test.depth_map_to_point_cloud("l1")


def tests():
    Tools.load_files("./tmp/render_out/.*")
    Tools.load_files(".*", "./tmp/render_out/")
    sys.exit()


def main():
    gui = Gui()
    gui.build()
    # service_pc()


if __name__ == "__main__":
    main()


# pip freeze > requirements.txt
# https://qavalidation.com/2021/01/how-to-create-use-requirements-txt-for-python-projects.html/
# https://learnpython.com/blog/python-requirements-file/
