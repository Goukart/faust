# import PointCloud.PointCloud as pcl
import MiDaS
from PointCloud.PointCloud import PointCloud
import Test
from enum import Enum


class Paths(Enum):
    depth_images = "PointCloud/depth/"
    color_images = "PointCloud/color/"


def service_pc():
    # pc = PointCloud()
    # pc.call()
    # MiDaS.generate("./images/1", "large", "tmp")
    MiDaS.dry()
    Test.run()  # must also reduce points, high res images have way too many points, convert to layers maybe?


def main():
    service_pc()


if __name__ == "__main__":
    main()
