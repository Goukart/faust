# import PointCloud.PointCloud as pcl
import MiDaS
import Tools
from PointCloud.PointCloud import PointCloud
import Test
# import mic
from enum import Enum


class Paths(Enum):
    depth_images = "PointCloud/depth/"
    color_images = "PointCloud/color/"


def service_pc():
    # pc = PointCloud()
    # pc.call()
    # MiDaS.generate("./images/1", "large", "tmp")
    case = "chess"
    # depth_map = Test.dry(case)
    # Tools.export_bytes_to_image(depth_map, case, "PointCloud/depth/")
    Test.run(case)  # must also reduce points, high res images have way too many points, convert to layers maybe?
    # mic.main()


def main():
    service_pc()


if __name__ == "__main__":
    main()


# pip freeze > requirements.txt
# https://qavalidation.com/2021/01/how-to-create-use-requirements-txt-for-python-projects.html/
# https://learnpython.com/blog/python-requirements-file/
