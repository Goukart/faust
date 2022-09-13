# import PointCloud.PointCloud as pcl
import os

import imageio
import open3d as o3d
import MiDaS
import Tools      # Custom helpful functions
# from PointCloud.PointCloud import PointCloud
import Test
import numpy as np
from matplotlib import pyplot as plt
import subprocess
import sys


# Orchestrate all the Modules
# ToDo: pipe.sh should convert all input images to png

# ToDo wip
# Convert and scale to a (16 bit) unsigned integer and normalize it
import inject
from GUI import Gui


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


# ToDo: temp
def quick_dm(name: str):
    # dm = Test.__generate_scale_image(3480, 4640, np.float32)
    dms = MiDaS.generate_dms(f"{color_images}{name}\..*", "large")
    # key = f"z_{name}"
    # print("Type of depth map: ", type(dm[key]))
    # print(dm)
    for key in dms:
        print(type(key))
        Tools.export_bytes_to_image(dms[key], key, depth_images)


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


def ray(vector: list) -> o3d.geometry.Geometry:
    import open3d as o3d
    x, y, z = vector

    print("Let's define some primitives")
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=x,
                                                    height=y,
                                                    depth=z)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

    return mesh_box


# receive a number of points, arrange them in a grid, apply a scale and done:
# get np.random.rand(10000, 3), arrange in a grid and scale z, so it's not 0-1
def grid():
    return 0


def generate_grid(_depths: np.array) -> o3d.geometry.PointCloud:
    wid, hig = _depths.shape
    return generate_gridx(wid, hig, _depths)


def generate_gridx(_width: int, _height: int, _depths: list, _scale=1) -> o3d.geometry.PointCloud:
    points = []
    for i in range(0, _width):
        for j in range(0, _height):
            x = i
            y = j
            z = _depths[i][j]  # math.floor(random.random() * 10)
            # z = math.floor(random.random() * 10)
            # funi: image_array[i % wid][i % hig]

            points.append([x, y, z])
            # invers.append([x, y, -z])
            # test.append([x, y, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def service_pc():
    # pc = PointCloud()
    # pc.call()
    # MiDaS.generate("./images/1", "large", "tmp")

    ####################################################################
    #
    #           Test to see what datatype the depth map has
    #
    ####################################################################
    # ToDo: free up space inbetween generation. GPU is "full" after one image
    # quick_dm("l2")

    from PIL import Image
    import random
    # load file
    # pcd = o3d.io.read_point_cloud("PointCloud/custom.ply")

    # TypeError: __init__(): incompatible constructor arguments. The following argument types are supported:
    # 1. open3d.cuda.pybind.utility.Vector3dVector()
    # 2. open3d.cuda.pybind.utility.Vector3dVector(arg0: numpy.ndarray[numpy.float64])
    # 3. open3d.cuda.pybind.utility.Vector3dVector(arg0: open3d.cuda.pybind.utility.Vector3dVector)
    # 4. open3d.cuda.pybind.utility.Vector3dVector(arg0: Iterable)


    # random
    # pcd = np.random.randint(0, 100, (100, 100))

    # From depth map
    image_array = np.asarray(Image.open("PointCloud/depth/z_l1.png"))
    # inverse = -image_array
    pcd = generate_grid(image_array)

    # Test.depth_map_to_point_cloud("l1")


    # some_pp = some + [0, 10, 0]
    # print(some)
    # print("\n plus shot: \n", some_pp)

    o3d.visualization.draw_geometries([pcd])

    # print("test selection: want all heights")
    # print([row[-1] for row in all_hights])

    # print(image_array) -> x * y Array, not all points in a list
    sys.exit()


    # cast "rays"
    cameras = [[wid/2, hig/2, 50], [wid/2, hig/3, 50]]
    rays = []
    # print(xyz)
    for i in range(len(xyz)):
        c = None
        if i > len(xyz):
            c = 0
        else:
            c = 1
        rays.append([c, i])
    colors = [[1, 0, 0] for i in range(len(rays))]
    points = cameras + xyz
    print(len(points))
    print(points)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        # points=o3d.utility.Vector3dVector([[wid/2, hig/2, 50], [wid/2, hig/3, 50], [10, 10, 5], [100, 100, 5], [50, 70, 2], [0, 80, 4], [80, 0, 8]]),
        lines=o3d.utility.Vector2iVector(rays),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Mask to ignore certain points (primitive like only a threshold unless you get creative)
    # https://github.com/isl-org/Open3D/issues/2291
    # mask = xyz[:, :, 1] < 800 ???
    # pcd.points = o3d.utility.Vector3dVector(points[mask])  # normals and colors are unchanged

    # print(xyz)

    pcd = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd2.points = o3d.utility.Vector3dVector(test)

    vec = ray([0.01, 0.01, 100])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[-0, -0, -0])
    # o3d.visualization.draw_geometries([pcd, pcd2, mesh_frame, line_set])

    o3d.visualization.draw_geometries([pcd])

    sys.exit()

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
    case = "1"

    # Generate a depth map from one image
    # dms = MiDaS.generate_dms(f"PointCloud/color/face.*", "large", "filename")
    # quick_dm(case)


    # Test.dry(case)
    #depth_map = Test.__generate_scale_image(5184, 3456, np.float32)  # , _range=None)
    regex = ""
    file_name = ""
    # dms = MiDaS.generate_dms(regex, "large", file_name)
    # Tools.export_bytes_to_image(depth_map, "z_chess", depth_images)
    # exit()
    # ToDo save image as 16 bit single channel gray scale, is handled in Tools.py

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
