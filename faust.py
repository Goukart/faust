# import PointCloud.PointCloud as pcl
import imageio

import MiDaS
import Tools      # Custom helpful functions
# from PointCloud.PointCloud import PointCloud
import Test
# import mic
import numpy as np
from matplotlib import pyplot as plt


# Orchestrate all the Modules
# ToDo: pipe.sh should convert all input images to png

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


depth_images = "PointCloud/depth/"
color_images = "PointCloud/color/"


def quick_dm(name: str):
    # dm = Test.__generate_scale_image(3480, 4640, np.float32)
    dm = MiDaS.generate_dms(f"{color_images}{name}.*", "large")
    key = f"z_{name}"
    # print("Type of depth map: ", type(dm[key]))
    # print(dm)
    Tools.export_bytes_to_image(dm[key], key, depth_images)


def service_pc():
    # pc = PointCloud()
    # pc.call()
    # MiDaS.generate("./images/1", "large", "tmp")

    ####################################################################
    #
    #           Test to see what datatype the depth map has
    #
    ####################################################################
    # dm = MiDaS.generate_dms(f"{color_images}chess.jpg", "large")
    # print("Type of depth map: ", type(dm["z_chess"]))
    # print(dm)

    ####################################################################
    #
    #           Photogrammetry
    #
    ####################################################################
    # mic.main()

    ####################################################################
    #
    #           Generate Depth Map
    #
    ####################################################################
    case = "arm"

    # Test.dry(case)
    #depth_map = Test.__generate_scale_image(5184, 3456, np.float32)  # , _range=None)
    # dms = MiDaS.generate_dms(f"PointCloud/color/face.*", "large", "filename")
    #quick_dm(case)
    # Tools.export_bytes_to_image(depth_map, "z_chess", depth_images)
    # exit()

    ####################################################################
    #
    #           Create Point Cloud
    #
    ####################################################################
    # must also reduce points, high-res images have way too many points, convert to layers maybe?

    # color = imageio.imread(f"{color_images}{case}.jpg")

    Test.depth_map_to_point_cloud(case)

    #plt.subplot(1, 2, 1)
    #plt.title('color image')
    #plt.imshow(color)
    #plt.subplot(1, 2, 2)
    #plt.title('depth image')
    #plt.imshow(depth_map)
    #plt.show()


def main():
    service_pc()


if __name__ == "__main__":
    main()


# pip freeze > requirements.txt
# https://qavalidation.com/2021/01/how-to-create-use-requirements-txt-for-python-projects.html/
# https://learnpython.com/blog/python-requirements-file/
