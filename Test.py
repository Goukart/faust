import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Maybe temp
import modules.MiDaS as MiDaS
import modules.Tools as Tools


def __generate_scale_image(width, height, data_type, _range=None):
    if _range is None:
        _range = range(0, height, 1)
    _from = Tools.limits(data_type).min
    _to = Tools.limits(data_type).max
    # generate gradient image from 0 to image height with given step
    scale = np.zeros((height, width), data_type)
    border_width = 1
    border_value = 1
    for i in range(_range.start, _range.stop):
        # print(i)
        row = np.ones((1, width), data_type) * (i * _range.step)
        row[0][-border_width:width] = np.ones((1, border_width), data_type) * border_value
        row[0][0:border_width] = np.ones((1, border_width), data_type) * border_value
        scale[i] = row
    scale[0:border_width] = np.ones((border_width, width), data_type) * border_value
    scale[-border_width:width] = np.ones((border_width, width), data_type) * border_value
    print(scale)
    return scale


def pick(case, data_type):
    if case == "hand":
        return np.load('z_1_binary.npy')
    elif case == "stool":
        img = Image.open("PointCloud/depth/00000.png")
        return np.array(img)
    elif case == "generate":
        model = "large"
        name = "dry_gen"
        regex = "PointCloud/color/face.jpg"
        MiDaS.generate_dms_regex(regex, model, name)
        exit()
    elif case == "mstool":
        img = Image.open("PointCloud/depth/z_00000.png")
        return np.array(img)
    elif case == "chess":
        img = Image.open("PointCloud/depth/z_dry.png")
        return np.array(img)
    else:
        print("No case in dry run in MiDaS")
        print("Try to generate new")
        img = Image.open(f"PointCloud/color/{case}")
        name = f"dry_{case}"
        depth_map = MiDaS.generate_dms_list([img], "large", name)

        for key in depth_map:
            Tools.export_bytes_to_image(depth_map[key], key, name)
        exit()


def dry(case):
    print("########## Dry run ##########")
    MiDaS.output_name = ""
    data_type = np.float32

    # Load/Generate one of the depth maps
    depth_map = pick(case, data_type)

    print("depth map:")
    print(depth_map)

    # Here a quick interception to test
    # depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    # points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    # __save_result(depth_map, f"dry{__Config.output_file}", "PointCloud/depth/")
    return depth_map
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
    __save_result(scaled, f"dry{__Config.output_file}", "PointCloud/depth/")
