import math
import time

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys

import modules.MiDaS as MiDaS
import modules.Tools as Tools
import open3d as o3d
import random


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

# receive a number of points, arrange them in a grid, apply a scale and done:
# get np.random.rand(10000, 3), arrange in a grid and scale z, so it's not 0-1
def grid():
    return 0


def grayscale_to_point_cloud(_depths: np.array, _scale=1) -> o3d.geometry.PointCloud:
    _width, _height = _depths.shape

    points = []
    # ToDo use numpy for speed
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


def plot_images(_image: str = "l1"):
    # Open 3D stuff
    img = Image.open(f"PointCloud/depth/z_{_image}.png")
    c_img = Image.open(f"PointCloud/color/{_image}.jpg")
    plt.ion()
    plt.subplot(1, 2, 1)
    plt.title('color image')
    plt.imshow(c_img)
    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(img)
    plt.show()


def angle(_vector: np.array, _axis: np.array) -> float:
    if np.linalg.norm(_vector) == 0:
        return 0

    rad = (_vector @ _axis) / \
          (np.linalg.norm(_vector) * np.linalg.norm(_axis))
    # print("axis: ", _axis)
    # print("vec: ", _vector)
    # print("rad: ", rad)
    return math.degrees(math.acos(rad))


#def angle(_radiant: float) -> float:
#    return math.degrees(_radiant)


def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat


def _rotate_to_vector(_origin: np.array, _vector: np.array) -> np.array:
    # Todo might be badly optimized
    """
    Returns a matrix, that rotates points around the origin to point in the vector direction
    :param _vector: The direction which the points should face
    :return: Rotation matrix
    """
    _origin = np.array([1, 0, 0])
    rodriguez = rodrigues_vec_to_rotation_mat(_vector)

    print("vector: ", _vector)
    _rotation = [
        angle(_vector, np.array([1, 0, 0])),
        angle(_vector, np.array([0, 1, 0])),
        angle(_vector, np.array([0, 0, 1])),
    ]
    # projections = a*b/|b|
    # v - (v * plane)*plane
    projectins = np.array([
        _vector, #  * np.array([1, 0, 0]),
        _vector, #  * np.array([0, 1, 0]),
        _vector #  * np.array([0, 0, 1])
    ])
    print("projectins: ", projectins)
    print(f"angle: {angle(projectins[0], np.array([1, 0, 0]))}, {angle(_origin, np.array([1, 0, 0]))}")
    print(f"angle: {angle(projectins[1], np.array([0, 1, 0]))}, {angle(_origin, np.array([0, 1, 0]))}")
    print(f"angle: {angle(projectins[2], np.array([0, 0, 1]))}, {angle(_origin, np.array([0, 0, 1]))}")
    # print(f"angle: {math.acos(_vector.dot(np.array([1, 0, 0])))}, {angle(_vector, np.array([1, 0, 0]))}")
    # print(f"angle: {_vector.dot(np.array([0, 1, 0]))}, {angle(_vector, np.array([0, 1, 0]))}")
    # print(f"angle: {_vector.dot(np.array([0, 0, 1]))}, {angle(_vector, np.array([0, 0, 1]))}")
    _origin = [
        angle(_origin, np.array([1, 0, 0])),
        angle(_origin, np.array([0, 1, 0])),
        angle(_origin, np.array([0, 0, 1]))
    ]
    for i in range(len(_rotation)):
        if _rotation[i] == _origin[i]:
            _rotation[i] = 0
    print("rotation mapd: ", _origin)
    print("rotation: ", _rotation)
    _rotation = [
        math.radians(180 + 45),
        math.radians(45),
        math.radians(45),
    ]
    print("rotation: ", _rotation)

    x_rot = np.array([
        [1, 0, 0],
        [0, math.cos(_rotation[0]), - math.sin(_rotation[0])],
        [0, math.sin(_rotation[0]), math.cos(_rotation[0])]
    ])
    y_rot = np.array([
        [math.cos(_rotation[1]), 0, math.sin(_rotation[1])],
        [0, 1, 0],
        [- math.sin(_rotation[1]), 0, math.cos(_rotation[1])]
    ])
    z_rot = np.array([
        [math.cos(_rotation[2]), - math.sin(_rotation[2]), 0],
        [math.sin(_rotation[2]), math.cos(_rotation[2]), 0],
        [0, 0, 1]
    ])
    rotation = x_rot @ y_rot @ z_rot

    print("matrix: ", rotation)
    print("rodrig: ", rodriguez)
    return rotation


E = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


def rotate_to(_vector: np.array) -> np.array:


    # origin_p = np.array([
    #     [0, 0, 0],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])

    origin_p = np.array([
        [0, 0, 0],
        [0.968740071042690554, 0.043792466156151344, 0.244182093250437438],
        [0.0579026212295622386, -0.997023473972581953, -0.0509065693441429837],
        [0.241225954679318955, 0.0634540168595064541, -0.968392289587977184]
    ])

    origin_l = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
    ])

    rotation = np.array([
        [0.968740071042690554, 0.043792466156151344, 0.244182093250437438],
        [0.0579026212295622386, -0.997023473972581953, -0.0509065693441429837],
        [0.241225954679318955, 0.0634540168595064541, -0.968392289587977184]
    ])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(origin_p),
        lines=o3d.utility.Vector2iVector(origin_l),
    )
    colors = [[0, 1, 1] for i in range(len(origin_p))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


class Camera(object):
    position = np.array([0, 0, 0])  # where is the camera?
    direction = np.array([0, 0, 0])  # What is it looking at
    rotation = 0  # how is it "twisted", is te image upright etc.
    dimension = np.array([0, 0])  # what are the dimensions of the image

    def __init__(self, position: np.array, direction: np.array, rotation: float, dimension: np.array):
        self.position = position  # np.array([1, 1, 4])
        self.direction = direction  # np.array([0, 0, -1])
        self.rotation = rotation  # 45
        self.dimension = dimension  # np.array([2, 1])

    def get_wireframe(self, rotation, scale: float = 0.1) -> o3d.geometry.LineSet:
        points = np.array([
            [-self.dimension[0]/2, self.dimension[1]/2, 3],
            [self.dimension[0]/2, self.dimension[1]/2, 3],
            [-self.dimension[0]/2, -self.dimension[1]/2, 3],
            [self.dimension[0]/2, -self.dimension[1]/2, 3],
            [0, 0, 0],
            [0, 0, 10]
        ])
        # points *= np.array([1, 1, 0])  # flatten in on xy plane

        points *= scale
        # points = points @ _rotate_to_vector(self.direction)
        points = points @ rotation  # _rotate_to_vector(np.array([-1, 0, 0]), self.direction)
        points += self.position

        frame = [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
            [4, 5]
        ]
        colors = [[0, 1, 0] for i in range(len(frame))]
        # print(len(points))
        # print(len(rays))
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            # points=o3d.utility.Vector3dVector([
            # [wid/2, hig/2, 50], [wid/2, hig/3, 50], [10, 10, 5],
            # [100, 100, 5], [50, 70, 2], [0, 80, 4], [80, 0, 8]]),
            lines=o3d.utility.Vector2iVector(frame),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set


# I need a camera position and orientation -> two vectors + rotation
# also an image, that the camera captured with scale and distance from camera, so that a ray cast from
# camera intersecting a pixel hits a point in space
# pos = []
# camera = (pos, )
def cast(a, b):
    #
    return 0


from enum import Enum


class CameraData(object):
    Center: np.array

    def __init__(self, center: np.array):
        self.Center = center


# import xml.dom.minidom
import xml.etree.ElementTree as ET


# XML Structure
CENTER = './OrientationConique/Externe/Centre'
ROTATION_MATRIX = './OrientationConique/Externe/ParamRotation/CodageMatr'


def parse_camera() -> CameraData:
    xml_file = '/home/ben/Workspace/Git/faust/mm_out/Ori-hand/Orientation-IMG_20220307_161951.jpg.xml'
    # document = xml.dom.minidom.parse("college.xml")
    tree = ET.parse(xml_file)

    root = tree.getroot()
    print('Records from XML file:')

    center = np.array([float(i) for i in root.findtext(CENTER).split(" ")])
    val = '0.968740071042690554'
    print(val)
    print(np.float32(val))
    print(np.float64(val))
    print(np.longdouble(val))
    print(float(val))  # float is default 64
    matrix = [
        [np.longfloat(i) for i in root.findtext(f"{ROTATION_MATRIX}/L1").split(" ")],
        [np.longfloat(i) for i in root.findtext(f"{ROTATION_MATRIX}/L2").split(" ")],
        [np.longfloat(i) for i in root.findtext(f"{ROTATION_MATRIX}/L3").split(" ")]
    ]
    val = np.longfloat(val)
    #matrix = [np.longfloat(i) for i in root.findtext(f"{ROTATION_MATRIX}/L1").split(" ")]
    print("Rotation Matrix: ", matrix)
    print("Type: ", type(matrix[0]))
    print("Type: ", type(0.968740071042690554))
    print("Type: ", type(np.longfloat(val)))
    print("Center: ", center)
    print("Matrix: ", matrix)
    matrix = np.array(matrix, dtype=np.float16)
    print("Matrix: ", matrix)

    camera = CameraData(
        np.array([])
    )
    return camera


def some_points():
    points = np.array([
        [-3.291, 1.110, -0.385],
        [-3.300, 1.110, -0.387],
        [-3.310, 1.110, -0.390],
        [-3.319, 1.109, -0.392],
        [-3.329, 1.109, -0.394],
        [-3.338, 1.109, -0.397],
        [-3.348, 1.108, -0.399],
        [-3.358, 1.108, -0.402],
        [-3.367, 1.108, -0.404],
        [-3.377, 1.107, -0.406],
        [-3.386, 1.107, -0.409],
        [-3.396, 1.107, -0.411],
        [-3.405, 1.106, -0.414],
        [-3.415, 1.106, -0.416],
        [-3.424, 1.106, -0.418],
        [-3.434, 1.105, -0.421],
        [-3.444, 1.105, -0.423],
        [-3.453, 1.105, -0.426],
        [-3.463, 1.104, -0.428],
        [-3.472, 1.104, -0.430],
        [-3.482, 1.104, -0.433],
        [-3.491, 1.103, -0.435],
        [-3.501, 1.103, -0.437],
        [-3.511, 1.103, -0.440],
        [-3.520, 1.102, -0.442],
        [-3.530, 1.102, -0.445],
        [-3.539, 1.102, -0.447],
        [-3.549, 1.101, -0.449],
        [-3.549, 1.101, -0.449],
        [-3.546, 1.097, -0.441],
        [-3.543, 1.092, -0.433],
        [-3.540, 1.088, -0.425],
        [-3.537, 1.083, -0.416],
        [-3.534, 1.078, -0.408],
        [-3.531, 1.074, -0.400],
        [-3.528, 1.069, -0.391],
        [-3.525, 1.064, -0.383],
        [-3.522, 1.060, -0.375],
        [-3.519, 1.055, -0.367],
        [-3.516, 1.051, -0.358],
        [-3.513, 1.046, -0.350],
        [-3.510, 1.041, -0.342],
        [-3.507, 1.037, -0.333],
        [-3.504, 1.032, -0.325],
        [-3.501, 1.028, -0.317],
        [-3.498, 1.023, -0.309],
        [-3.496, 1.018, -0.300],
        [-3.493, 1.014, -0.292],
        [-3.490, 1.009, -0.284],
        [-3.487, 1.004, -0.275],
        [-3.484, 1.000, -0.267],
        [-3.481, 0.995, -0.259],
        [-3.478, 0.991, -0.251],
        [-3.475, 0.986, -0.242],
        [-3.472, 0.981, -0.234],
        [-3.469, 0.977, -0.226],
        [-3.466, 0.972, -0.217],
        [-3.463, 0.968, -0.209],
        [-3.460, 0.963, -0.201],
        [-3.457, 0.958, -0.193],
        [-3.454, 0.954, -0.184],
        [-3.451, 0.949, -0.176],
        [-3.448, 0.944, -0.168],
        [-3.445, 0.940, -0.159],
        [-3.442, 0.935, -0.151],
        [-3.439, 0.931, -0.143],
        [-3.436, 0.926, -0.135],
        [-3.433, 0.921, -0.126],
        [-3.430, 0.917, -0.118],
        [-3.427, 0.912, -0.110],
        [-3.299, 0.769, -0.369],
        [-3.299, 0.779, -0.370],
        [-3.299, 0.789, -0.370],
        [-3.299, 0.798, -0.370],
        [-3.298, 0.808, -0.371],
        [-3.298, 0.818, -0.371],
        [-3.298, 0.828, -0.372],
        [-3.298, 0.837, -0.372],
        [-3.297, 0.847, -0.373],
        [-3.297, 0.857, -0.373],
        [-3.297, 0.867, -0.374],
        [-3.297, 0.876, -0.374],
        [-3.296, 0.886, -0.374],
        [-3.296, 0.896, -0.375],
        [-3.296, 0.906, -0.375],
        [-3.296, 0.915, -0.376],
        [-3.295, 0.925, -0.376],
        [-3.295, 0.935, -0.377],
        [-3.295, 0.945, -0.377],
        [-3.295, 0.954, -0.378],
        [-3.294, 0.964, -0.378],
        [-3.294, 0.974, -0.379],
        [-3.294, 0.984, -0.379],
        [-3.294, 0.993, -0.379],
        [-3.293, 1.003, -0.380],
        [-3.293, 1.013, -0.380],
        [-3.293, 1.023, -0.381],
        [-3.293, 1.032, -0.381],
        [-3.292, 1.042, -0.382],
        [-3.292, 1.052, -0.382],
        [-3.292, 1.062, -0.383],
        [-3.292, 1.071, -0.383],
        [-3.291, 1.081, -0.383],
        [-3.291, 1.091, -0.384],
        [-3.291, 1.101, -0.384],
        [-3.291, 1.110, -0.385],
        [-3.291, 1.110, -0.385],
        [-3.294, 1.105, -0.377],
        [-3.298, 1.100, -0.370],
        [-3.302, 1.094, -0.363],
        [-3.305, 1.089, -0.355],
        [-3.309, 1.084, -0.348],
        [-3.313, 1.078, -0.340],
        [-3.316, 1.073, -0.333],
        [-3.320, 1.067, -0.325],
        [-3.324, 1.062, -0.318],
        [-3.328, 1.057, -0.310],
        [-3.331, 1.051, -0.303],
        [-3.335, 1.046, -0.296],
        [-3.339, 1.041, -0.288],
        [-3.342, 1.035, -0.281],
        [-3.346, 1.030, -0.273],
        [-3.350, 1.025, -0.266],
        [-3.353, 1.019, -0.258],
        [-3.357, 1.014, -0.251],
        [-3.361, 1.009, -0.244],
        [-3.365, 1.003, -0.236],
        [-3.368, 0.998, -0.229],
        [-3.372, 0.993, -0.221],
        [-3.376, 0.987, -0.214],
        [-3.379, 0.982, -0.206],
        [-3.383, 0.976, -0.199],
        [-3.387, 0.971, -0.192],
        [-3.390, 0.966, -0.184],
        [-3.394, 0.960, -0.177],
        [-3.398, 0.955, -0.169],
        [-3.402, 0.950, -0.162],
        [-3.405, 0.944, -0.154],
        [-3.409, 0.939, -0.147],
        [-3.413, 0.934, -0.139],
        [-3.416, 0.928, -0.132],
        [-3.420, 0.923, -0.125],
        [-3.424, 0.918, -0.117],
        [-3.427, 0.912, -0.110],
        [-3.539, 0.757, -0.429],
        [-3.529, 0.758, -0.426],
        [-3.519, 0.758, -0.424],
        [-3.510, 0.759, -0.422],
        [-3.500, 0.759, -0.419],
        [-3.491, 0.760, -0.417],
        [-3.481, 0.760, -0.414],
        [-3.472, 0.760, -0.412],
        [-3.462, 0.761, -0.410],
        [-3.452, 0.761, -0.407],
        [-3.443, 0.762, -0.405],
        [-3.433, 0.762, -0.403],
        [-3.424, 0.763, -0.400],
        [-3.414, 0.763, -0.398],
        [-3.405, 0.764, -0.395],
        [-3.395, 0.764, -0.393],
        [-3.385, 0.765, -0.391],
        [-3.376, 0.765, -0.388],
        [-3.366, 0.766, -0.386],
        [-3.357, 0.766, -0.383],
        [-3.347, 0.767, -0.381],
        [-3.338, 0.767, -0.379],
        [-3.328, 0.768, -0.376],
        [-3.318, 0.768, -0.374],
        [-3.309, 0.769, -0.371],
        [-3.299, 0.769, -0.369],
        [-3.299, 0.769, -0.369],
        [-3.303, 0.773, -0.361],
        [-3.307, 0.778, -0.353],
        [-3.311, 0.782, -0.345],
        [-3.315, 0.786, -0.338],
        [-3.319, 0.791, -0.330],
        [-3.323, 0.795, -0.322],
        [-3.326, 0.799, -0.314],
        [-3.330, 0.804, -0.306],
        [-3.334, 0.808, -0.298],
        [-3.338, 0.812, -0.290],
        [-3.342, 0.817, -0.283],
        [-3.346, 0.821, -0.275],
        [-3.350, 0.825, -0.267],
        [-3.354, 0.830, -0.259],
        [-3.358, 0.834, -0.251],
        [-3.361, 0.838, -0.243],
        [-3.365, 0.843, -0.235],
        [-3.369, 0.847, -0.228],
        [-3.373, 0.851, -0.220],
        [-3.377, 0.856, -0.212],
        [-3.381, 0.860, -0.204],
        [-3.385, 0.864, -0.196],
        [-3.389, 0.869, -0.188],
        [-3.392, 0.873, -0.180],
        [-3.396, 0.877, -0.173],
        [-3.400, 0.882, -0.165],
        [-3.404, 0.886, -0.157],
        [-3.408, 0.890, -0.149],
        [-3.412, 0.895, -0.141],
        [-3.416, 0.899, -0.133],
        [-3.420, 0.903, -0.125],
        [-3.424, 0.908, -0.118],
        [-3.427, 0.912, -0.110],
        [-3.549, 1.101, -0.449],
        [-3.548, 1.092, -0.449],
        [-3.548, 1.082, -0.448],
        [-3.548, 1.072, -0.448],
        [-3.548, 1.062, -0.447],
        [-3.547, 1.052, -0.446],
        [-3.547, 1.042, -0.446],
        [-3.547, 1.033, -0.445],
        [-3.546, 1.023, -0.445],
        [-3.546, 1.013, -0.444],
        [-3.546, 1.003, -0.444],
        [-3.546, 0.993, -0.443],
        [-3.545, 0.983, -0.442],
        [-3.545, 0.974, -0.442],
        [-3.545, 0.964, -0.441],
        [-3.544, 0.954, -0.441],
        [-3.544, 0.944, -0.440],
        [-3.544, 0.934, -0.439],
        [-3.544, 0.924, -0.439],
        [-3.543, 0.914, -0.438],
        [-3.543, 0.905, -0.438],
        [-3.543, 0.895, -0.437],
        [-3.542, 0.885, -0.436],
        [-3.542, 0.875, -0.436],
        [-3.542, 0.865, -0.435],
        [-3.541, 0.855, -0.435],
        [-3.541, 0.846, -0.434],
        [-3.541, 0.836, -0.434],
        [-3.541, 0.826, -0.433],
        [-3.540, 0.816, -0.432],
        [-3.540, 0.806, -0.432],
        [-3.540, 0.796, -0.431],
        [-3.539, 0.787, -0.431],
        [-3.539, 0.777, -0.430],
        [-3.539, 0.767, -0.429],
        [-3.539, 0.757, -0.429],
        [-3.539, 0.757, -0.429],
        [-3.536, 0.761, -0.420],
        [-3.533, 0.765, -0.412],
        [-3.530, 0.769, -0.404],
        [-3.527, 0.773, -0.395],
        [-3.524, 0.778, -0.387],
        [-3.521, 0.782, -0.378],
        [-3.518, 0.786, -0.370],
        [-3.515, 0.790, -0.362],
        [-3.512, 0.794, -0.353],
        [-3.509, 0.798, -0.345],
        [-3.506, 0.802, -0.336],
        [-3.503, 0.806, -0.328],
        [-3.501, 0.810, -0.320],
        [-3.498, 0.814, -0.311],
        [-3.495, 0.818, -0.303],
        [-3.492, 0.822, -0.294],
        [-3.489, 0.826, -0.286],
        [-3.486, 0.831, -0.278],
        [-3.483, 0.835, -0.269],
        [-3.480, 0.839, -0.261],
        [-3.477, 0.843, -0.252],
        [-3.474, 0.847, -0.244],
        [-3.471, 0.851, -0.236],
        [-3.468, 0.855, -0.227],
        [-3.465, 0.859, -0.219],
        [-3.463, 0.863, -0.210],
        [-3.460, 0.867, -0.202],
        [-3.457, 0.871, -0.194],
        [-3.454, 0.875, -0.185],
        [-3.451, 0.880, -0.177],
        [-3.448, 0.884, -0.168],
        [-3.445, 0.888, -0.160],
        [-3.442, 0.892, -0.152],
        [-3.439, 0.896, -0.143],
        [-3.436, 0.900, -0.135],
        [-3.433, 0.904, -0.127],
        [-3.430, 0.908, -0.118],
        [-3.427, 0.912, -0.110]
    ])
    return points


def appuis_938():
    points = np.array([
        [-5.4428098529235589, 6.39285025659827166, -11.9415876119580293],
        [0.742909264785465595, -5.14953114688189562, -16.1613093821674525],
        [-6.15829655080877369, 1.5542550581647907, -10.4944708286439052],
        [-1.4477320682253314, 4.6785689890181148, -13.8805487946971624],
        [1.82467522785937475, 4.08673819012450146, -13.5386177104170837],
        [-1.37237110510938276, 2.2573537099158405, -15.883120841959645],
        [-2.44008489351142099, 12.1424925779197856, -16.8028550834303978],
        [-3.94367914935664299, 0.459293857052059051, -9.83141597453896487],
        [-0.0209104756087414856, 0.166883359875661053, -15.1808947962243099],
        [0.978702765791506613, 2.83299140648815007, -8.33950177845214036]
    ])
    return points


def draw_lines(_origin: np.array, _targets: np.array, color: list = [0, 1, 0], offset: np.array = np.array([0, 0, 0])) -> o3d.geometry.LineSet:
    points = np.concatenate((np.array([_origin]), _targets))
    points += offset
    connections = [[0, i] for i in range(1, len(points))]
    colors = [color for i in range(len(connections))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(connections),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def axis_colors(rotation: np.array, offset: np.array = np.array([0, 0, 0])) -> o3d.geometry.LineSet:
    origin = np.array([0, 0, 0])
    line_set = draw_lines(origin, (E @ rotation), offset=offset)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return line_set


def cent_938():
    return [2.65940031031604551, -0.396204596148851951, -1.01447047598784978]


def cent_942():
    return [0, 0, 0]


def stuff_951() -> o3d.geometry.LineSet:
    rotation = np.array([
        [0.968740071042690554, 0.043792466156151344, 0.244182093250437438],
        [0.0579026212295622386, -0.997023473972581953, -0.0509065693441429837],
        [0.241225954679318955, 0.0634540168595064541, -0.968392289587977184]
    ])
    front = np.array([[-3.6709401432089841, 2.84011471603499821, -15.6965221529098002]])  # np.array([0, 0, 1])
    front = -front @ rotation

    points = np.concatenate((
        [center_951],
        front
        ), axis=0)

    frame = [
        [0, 1]
    ]
    colors = [[0, 1, 1] for i in range(len(frame))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(frame),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


# 951 Stuff
center_951 = np.array(
    [-3.42746262763605847, 0.912160136530507537, -0.109722948479107019]
)
pts_951 = np.array([
    [-3.6709401432089841, 2.84011471603499821, -15.6965221529098002],
    [-4.094261555543242, -6.66353816396305731, -18.8282844137882677],
    [-4.77370061721167804, -0.324890206319016994, -13.7509978158501749],
    [-4.21124296186163249, -1.1557354076065014, -8.94352281642436608],
    [0.146862611176203028, -5.69429586843313107, -12.6277588344481853],
    [-5.84499678466673345, -4.68946395056037879, -11.3564036968619551],
    [-3.35681096015382341, 4.12222545332546808, -11.9105107859876647],
    [-2.28933810680366756, -4.65434253642436868, -12.8329009440179611],
    [-1.33654926699804077, -0.838970616826566817, -11.401248804735431],
    [-4.26897453330121301, -3.08593243734781852, -9.93736077508776994]
])


def from_file() -> o3d.geometry.PointCloud:
    points = appuis_938()
    points = np.array([
        cent_938(),
        cent_942()
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("points:")
    print("o3d: ", pcd)
    print("points raw: ", points)
    print("points: ", pcd.points)

    colors = [[1, 0, 0] for i in range(len(points))]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def colmap():
    colmap = o3d.io.read_point_cloud("colmap.ply")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array([
        [5.654400000, 1.754500000, 2.356000000],
        [5.654400000, 1.754500000, 2.356000000],
        [5.654400000, 1.754500000, 2.356000000],
        [3.712000000, 1.740000000, 2.320000000]
    ]))
    o3d.visualization.draw_geometries([colmap])


def micmac():
    cams = o3d.io.read_point_cloud("cams.ply")
    cams2 = o3d.io.read_point_cloud("cams2.ply")

    stuff = draw_lines(pts_951(), center_951)

    o3d.visualization.draw_geometries([cams, stuff])

def correct():
    # o3d.io.write_point_cloud("ascii.ply", cams, True)
    # points = np.asarray(cams.colors)
    # print(points)
    # file = from_file()
    file = stuff_951()

    # colmap()
    # micmac()

    # parse_camera()

    parse_camera()
    sys.exit()

    # Ray casting
    dir = np.array([-1, -1, -1])
    rot = 0.0
    dim = np.array([4, 3])
    camera = Camera(center_951, dir, rot, dim)
    # cameras = [[1, 1, 4]]
    grid3x3 = np.array([
        camera.position,
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0]
    ])
    rays = []
    # points.append(cameras[0])

    #print(points)
    for i in range(len(grid3x3)):
        rays.append([0, i])
    colors = [[1, 0, 0] for i in range(len(rays))]
    # print(len(points))
    # print(len(rays))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid3x3),
        lines=o3d.utility.Vector2iVector(rays),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    _points_grid3x3 = o3d.geometry.PointCloud()
    _points_grid3x3.points = o3d.utility.Vector3dVector(grid3x3)

    test_points = o3d.geometry.PointCloud()
    test_points.points = o3d.utility.Vector3dVector(some_points())
    # [_points_grid3x3, line_set, cams, test_points, file]

    rotation = np.array([
        [0.968740071042690554, 0.043792466156151344, 0.244182093250437438],
        [0.0579026212295622386, -0.997023473972581953, -0.0509065693441429837],
        [0.241225954679318955, 0.0634540168595064541, -0.968392289587977184],
    ])
    wh = np.array([[1, 1, 1]])
    print("norm1: ", np.linalg.norm(wh))
    Ewh = E/np.linalg.norm(wh)
    n = 1/np.linalg.norm(wh)
    Ewh = np.array([
        [1, 0, -1]/np.linalg.norm([1, 0, -1]),
        [0, 1, -1]/np.linalg.norm([0, 1, -1]),
        [1, 1, 1]/np.linalg.norm(wh),
    ])
    print("norm1: ", Ewh)
    zwh = axis_colors(Ewh)

    line_set = camera.get_wireframe(rotation, scale=4)
    # wht = wh @ rotation
    #rotat = draw_lines(np.array([0, 0, 0]), np.concatenate((rotation, wht)), [0, 1, 1], offset=center_951)

    rotat = axis_colors(rotation, center_951)
    # draw_lines(np.array([0, 0, 0]), rotation, [0, 1, 1], offset=center_951)

    # EE = np.concatenate((E, wh))
    # ccs = draw_lines(np.array([0, 0, 0]), EE, [1, 1, 0])
    ccs = axis_colors(E)

    cams = o3d.io.read_point_cloud("cams.ply")
    o3d.visualization.draw_geometries([_points_grid3x3, test_points, line_set, rotat, ccs, cams])
    return


    # load file
    # pcd = o3d.io.read_point_cloud("PointCloud/C3DC_BigMac.ply")

    # Merge
    # compute_point_cloud_distance
    # orient_normals_towards_camera_location
    #o3d.visualization.draw_geometries([pcd, pcd2])

    # Write to file
    # pcd.to_file("l1.ply", internal=["points", "mesh"])
    # o3d.io.write_point_cloud("l1.ply", pcd)
    # o3d.io.write_point_cloud("l2.ply", pcd2)


    # random
    # pcd = np.random.randint(0, 100, (100, 100))

    # Load image (with PIL)as np array
    img = Image.open("PointCloud/depth/z_l1.png")
    print("Some info about the image:")
    print("format: ", img.format)
    print("size: ", img.size)
    print("mode: ", img.mode)

    # plot_images()

    image_array = np.asarray(img)
    print("Image as array info:")
    print("shape: ", image_array.shape)
    print("data: ", image_array)

    inverse = -image_array

    # Visualize
    # pcd = grayscale_to_point_cloud(image_array)
    # o3d.visualization.draw_geometries([pcd])
    nor = grayscale_to_point_cloud(image_array)
    inv = grayscale_to_point_cloud(inverse)
    # pl = image_array * [0]
    arr = np.zeros((100, 100), int)
    plane = grayscale_to_point_cloud(arr)
    # center = grayscale_to_point_cloud(np.array)
    # o3d.visualization.draw_geometries([nor, inv, plane])
    #o3d.visualization.draw_geometries([plane, center])

    # depth_map_to_point_cloud("l1")



    # some_pp = some + [0, 10, 0]
    # print(some)
    # print("\n plus shot: \n", some_pp)


    # print("test selection: want all heights")
    # print([row[-1] for row in all_hights])

    # print(image_array) -> x * y Array, not all points in a list
    return


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
    grid3x3 = cameras + xyz
    print(len(grid3x3))
    print(grid3x3)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid3x3),
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


def pick(case, data_type):
    if case == "hand":
        return np.load('../z_1_binary.npy')
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
        img = Image.open("../PointCloud/depth/z_00000.png")
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
