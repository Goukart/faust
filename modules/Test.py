import math
import time

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys

import modules.MiDaS as MiDaS
import modules.Tools as Tools
import open3d as o3d
from fractions import Fraction
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


def grayscale_to_points(_image: np.array) -> np.array:
    _width, _height = _image.shape

    points = []
    # ToDo use numpy for speed
    for i in range(0, _width):
        for j in range(0, _height):
            x = i
            y = j
            z = _image[i][j]  # math.floor(random.random() * 10)
            # z = math.floor(random.random() * 10)
            # funi: image_array[i % wid][i % hig]

            points.append([x, y, z])
            # invers.append([x, y, -z])
            # test.append([x, y, 0])

    return np.array(points, dtype=float)


def grayscale_to_point_cloud(_depths: np.array, _scale=1) -> o3d.geometry.PointCloud:
    points = grayscale_to_points(_depths)
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


O = np.array([0, 0, 0])
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
    # Attributes
    # private
    # position: where is the camera?
    # rotation_matrix: coordinate system of the camera, i think z is direction? If so direction attr is redundant
    # dimension: what are the dimensions of the image
    #  -> rather (4,3) ? whats the tpye of that?

    # Pending I think?
    # direction: What is it looking at
    # ToDo: rotation?
    # rotation: how is it "twisted", is te image upright etc.
    # supports: some random points i don't know what they are for yet

    # def __init__(self, position: np.array, direction: np.array, rotation: float, dimension: np.array):
    def __init__(self, xml_file: str):
        camera_data = parse_camera(xml_file)
        self.position = camera_data["center"]  # np.array([1, 1, 4])
        # self.direction = direction  # np.array([0, 0, -1])
        # self.rotation = rotation  # 45
        self.dimension = np.array([3480, 4640])  # ToDo: get resolution from file
        ratio = Fraction(self.dimension[0], self.dimension[1])
        # print(f"Image ration is: {ratio.numerator} by {ratio.denominator}")
        self.ratio = (ratio.numerator, ratio.denominator)

        self.supports = camera_data["supports"]
        self.rotation_matrix = camera_data["matrix"]

    def get_wireframe(self, rotation, scale: float = 0.1) -> o3d.geometry.LineSet:
        # ToDo: calculate distance properly
        distance = 5
        points = np.array([
            [-self.ratio[0]/2, self.ratio[1]/2, distance],
            [self.ratio[0]/2, self.ratio[1]/2, distance],
            [-self.ratio[0]/2, -self.ratio[1]/2, distance],
            [self.ratio[0]/2, -self.ratio[1]/2, distance],
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

    def get_position(self):
        return self.position

    def get_rotation(self):
        return self.rotation_matrix

    def get_supports(self):
        return self.supports

    def project_plane(self, points: np.array, distance: int):
        print(points.shape)
        # Rotate to face camera direction and move to origin
        plane_pts = points @ self.rotation_matrix
        # center and move to origin
        plane_pts = plane_pts + self.position
        return plane_pts

# I need a camera position and orientation -> two vectors + rotation
# also an image, that the camera captured with scale and distance from camera, so that a ray cast from
# camera intersecting a pixel hits a point in space
# pos = []
# camera = (pos, )
def cast(a, b):
    #
    return 0


# import xml.dom.minidom
import xml.etree.ElementTree as ET


# XML Structure
XML_PATH = '/home/ben/Workspace/Git/faust/mm_out/Ori-hand/'
CENTER = './OrientationConique/Externe/Centre'
ROTATION_MATRIX = './OrientationConique/Externe/ParamRotation/CodageMatr'
SUPPORTS = './OrientationConique/Verif/Appuis'


def parse_camera(xml_file: str) -> dict:
    xml_file = XML_PATH + xml_file
    # document = xml.dom.minidom.parse("college.xml")
    tree = ET.parse(xml_file)

    root = tree.getroot()

    center = [np.longfloat(i) for i in root.findtext(CENTER).split(" ")]
    matrix = np.array([
        [np.longfloat(i) for i in root.findtext(f"{ROTATION_MATRIX}/L1").split(" ")],
        [np.longfloat(i) for i in root.findtext(f"{ROTATION_MATRIX}/L2").split(" ")],
        [np.longfloat(i) for i in root.findtext(f"{ROTATION_MATRIX}/L3").split(" ")]
    ])
    # some sort of verification?
    supports = [None for i in range(len(root.findall(SUPPORTS)))]
    for i in root.findall(SUPPORTS):
        supports[int(i.findtext("./Num"))] = [np.longfloat(t) for t in i.findtext("./Ter").split(" ")]
        # supports.append([np.longfloat(t) for t in xml_supp.findtext("./Num").split(" ")])

    camera = {
        "center": center,
        "matrix": matrix,
        "supports": supports,
    }

    print('Records from XML file:')
    for key in camera:
        print(f"{key}: {camera[key]}")

    return camera


xml_38 = parse_camera('Orientation-IMG_20220307_161938.jpg.xml')
xml_42 = parse_camera('Orientation-IMG_20220307_161942.jpg.xml')
xml_51 = parse_camera('Orientation-IMG_20220307_161951.jpg.xml')


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


def draw_lines(_origin: np.array, _targets: np.array, color: list = [0, 1, 0], offset: np.array = np.array([0, 0, 0])) -> o3d.geometry.LineSet:
    points = np.concatenate(([_origin], _targets))
    points += offset
    connections = [[0, i] for i in range(1, len(points))]
    colors = [color for i in range(len(connections))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(connections),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def axis_colors(rotation: np.array = E, offset: np.array = np.array([0, 0, 0])) -> o3d.geometry.LineSet:
    origin = np.array([0, 0, 0])
    line_set = draw_lines(origin, (E @ rotation), offset=offset)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return line_set


def stuff_951() -> o3d.geometry.LineSet:
    xml = parse_camera('Orientation-IMG_20220307_161951.jpg.xml')
    center = xml["center"]
    rotation = np.array([
        [0.968740071042690554, 0.043792466156151344, 0.244182093250437438],
        [0.0579026212295622386, -0.997023473972581953, -0.0509065693441429837],
        [0.241225954679318955, 0.0634540168595064541, -0.968392289587977184]
    ])
    front = np.array([[-3.6709401432089841, 2.84011471603499821, -15.6965221529098002]])  # np.array([0, 0, 1])
    front = -front @ rotation

    points = np.concatenate((
        [center],
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


def from_file() -> o3d.geometry.PointCloud:
    center_38 = xml_38["center"]
    center_42 = xml_42["center"]
    points_38 = np.array(xml_38["supports"])
    points = np.array([
        center_38,
        center_42
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

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
    xml = parse_camera('Orientation-IMG_20220307_161951.jpg.xml')
    pts = np.array(xml["supports"])
    center = np.array(xml["center"])
    cams = o3d.io.read_point_cloud("cams.ply")
    cams2 = o3d.io.read_point_cloud("cams2.ply")

    stuff = draw_lines(pts, center)

    o3d.visualization.draw_geometries([cams, stuff])


def cmpr(arr1, arr2):
    print("arr1: ", arr1)
    print("arr2: ", arr2)
    if Tools.array_is_equal(arr1, arr2):
        print("Are equal")
    else:
        print("Are not equal")


def gen_plane(width: int, height: int, scale: float = 1., origin: np.array = np.array([0, 0, 0])) -> np.array:
    origin = origin * scale
    plane = []
    for y in range(height):
        for x in range(width):
            plane.append([x * scale, y * scale, 0] + origin)

    return np.array(plane)


def project_plane(points: np.array, camera: Camera):
    # Rotate to face camera direction and move to origin
    plane_pts = points @ camera.rotation_matrix
    # center and move to origin
    plane_pts = plane_pts + camera.get_position()
    return plane_pts


def pc_from_image(img_path: str) -> o3d.geometry.PointCloud:
    img = Image.open(img_path)
    image_array = np.array(img)
    plane_pts = grayscale_to_point_cloud(image_array)

    return plane_pts


def minimal_correction_example():
    print("minimal_correction_example")
    # Make cube
    size = 3 # half a edge
    points = np.array([
        [-size, -size, -size],
        [-size, size, -size],
        [size, size, -size],
        [size, -size, -size],
        [-size, -size, size],
        [-size, size, size],
        [size, size, size],
        [size, -size, size]
    ])
    frame = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
    ]
    colors = [[0, 1, 0] for i in range(len(frame))]
    cube = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        # points=o3d.utility.Vector3dVector([
        # [wid/2, hig/2, 50], [wid/2, hig/3, 50], [10, 10, 5],
        # [100, 100, 5], [50, 70, 2], [0, 80, 4], [80, 0, 8]]),
        lines=o3d.utility.Vector2iVector(frame),
    )
    cube.colors = o3d.utility.Vector3dVector(colors)

    # Camera
    camera = Camera("test.xml")
    cam = camera.get_wireframe(camera.rotation_matrix)
    # projection = cam.project_plane()
    f = Fraction(1920, 1080)
    print(f"ratio: {f.numerator} by {f.denominator}")

    # wh = draw_lines(O, np.array([[1, 1, 1]]))
    o3d.visualization.draw_geometries([axis_colors(), cube, cam])


def correct():
    minimal_correction_example()
    sys.exit()

    # o3d.io.write_point_cloud("ascii.ply", cams, True)
    # points = np.asarray(cams.colors)
    # print(points)
    # file = from_file()
    file = stuff_951()

    # colmap()
    # micmac()

    # Cameras
    c_51 = Camera('Orientation-IMG_20220307_161951.jpg.xml')
    c_42 = Camera('Orientation-IMG_20220307_161942.jpg.xml')
    c_38 = Camera('Orientation-IMG_20220307_161938.jpg.xml')

    # Original Supports, without any manipulation
    sup_raw = draw_lines(O, c_51.supports, [0, 0, 0], offset=c_51.position)
    rotated = np.array(c_51.supports) @ c_51.rotation_matrix

    # Supports rotated with camera rotation matrix
    sup_rot = draw_lines(O, rotated, [0, 1, 1], offset=c_51.position)
    cams = o3d.io.read_point_cloud("hand.ply")

    # Display axis of rotation matrix
    rotcoord = axis_colors(c_51.rotation_matrix, c_51.position)
    camera_reconstr = c_51.get_wireframe(c_51.rotation_matrix, scale=0.1)

    # Create and do things with plane
    w, h, s = [30, 40, 0.5]
    offset = np.array([-w/2, -h/2, 25.])
    plane_pts = project_plane(gen_plane(w, h, s, offset), c_51)
    plane_pc = o3d.geometry.PointCloud()
    plane_pc.points = o3d.utility.Vector3dVector(plane_pts)

    # Use real depth map
    img = Image.open("PointCloud/depth/z_l1.png")
    image_array = grayscale_to_points(np.array(img))
    image_array *= 0.01
    w, h = [img.width * 0.01, img.height * 0.01]
    offset = np.array([-w / 2, -h / 2, 0.])
    image_array = image_array + offset
    print("img: ", image_array)
    print("shape: ", image_array.shape)
    plane_pc = o3d.geometry.PointCloud()
    plane_pc.points = o3d.utility.Vector3dVector(project_plane(image_array, c_51))

    # cams.points = o3d.utility.Vector3dVector(np.array(cams.points) @ camera_51.rotation_matrix)
    # o3d.visualization.draw_geometries([sup_raw, sup_rot, cams, rotcoord, camera_reconstr, plane_pc])
    # o3d.visualization.draw_geometries([cams, rotcoord, camera_reconstr, plane_pc])
    o3d.visualization.draw_geometries([rotcoord, plane_pc])

    sys.exit()
    # Ray casting
    grid3x3 = gen_plane(3, 3)
    _points_grid3x3 = o3d.geometry.PointCloud()
    _points_grid3x3.points = o3d.utility.Vector3dVector(grid3x3)
    line_set = draw_lines(c_51.position, plane_pts, [1, 0, 0])

    test_points = o3d.geometry.PointCloud()
    test_points.points = o3d.utility.Vector3dVector(some_points())

    # line_set = camera.get_wireframe(rotation, scale=4)
    # wht = wh @ rotation
    #rotat = draw_lines(np.array([0, 0, 0]), np.concatenate((rotation, wht)), [0, 1, 1], offset=center_951)

    rotat = axis_colors(c_51.rotation_matrix, c_51.position)
    # draw_lines(np.array([0, 0, 0]), rotation, [0, 1, 1], offset=center_951)

    # EE = np.concatenate((E, wh))
    # ccs = draw_lines(np.array([0, 0, 0]), EE, [1, 1, 0])
    ccs = axis_colors(E)

    cams = o3d.io.read_point_cloud("hand.ply")
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
