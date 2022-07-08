import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image

# Maybe temp
import MiDaS
from MiDaS import __Config
import Tools


def convert_to_1_ch_grayscale(image_path):
    # ToDo, make sure it is gray scale and 16 bit (and if it really has to be)
    img = PIL.Image.open(image_path)

    # img = img.convert('L', colors=2000)
    print(img)
    plt.imshow(img)  # cmap='gray'
    plt.show()

    new = 'PointCloud/depth/z-tmp.png'
    img.save(new)  # is saved as int or float might cause problems

    return new


def depth_map_to_point_cloud(file):
    # Seems like rgb image has to be rgb
    color_img_path = f"PointCloud/color/{file}.jpg"
    depth_img_path = f"PointCloud/depth/z_{file}.png"
    # depth_img_path = "PointCloud/depth/z_dry.png"

    color_raw = o3d.io.read_image(color_img_path)
    depth_raw = o3d.io.read_image(depth_img_path)
    print(depth_raw)

    # loading the image
    img = PIL.Image.open(depth_img_path)
    # fetching the dimensions
    wid, hgt = img.size

    # ############### Test zone ###############
    # new_depth = convert_to_1_ch_grayscale(depth_img_path)
    # depth = o3d.io.read_image(new_depth)
    # print(depth)

    # print(depth_raw)
    # #########################################

    # self, color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=True
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 0.001, 300.)
    # Image has to be spesific resolution (x,y)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 1.0, 65000)
    print(rgbd_image)

    plt.subplot(1, 2, 1)
    plt.title('grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    camera.width, camera.height = img.size

    print("camera properties: ")
    print(camera)
    print("#################################################################")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)
    # print(depth_raw)
    # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, camera)

    #  flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])  # visualize the point cloud


def pick(case, data_type):
    if case == "hand":
        return np.load('z_1_binary.npy')
    elif case == "stool":
        img = Image.open("PointCloud/depth/00000.png")
        return np.array(img)
    elif case == "generate":
        options = {
            __Config.Model: "large",
            __Config.Output: "dry_gen",
            __Config.Images: "PointCloud/color/face.jpg"
        }
        MiDaS.generate(options)
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
        options = {
            __Config.Model: "large",
            __Config.Output: f"dry_{case}",
            __Config.Images: f"PointCloud/color/{case}.jpg"
        }
        img = Image.open(f"PointCloud/color/{case}")
        depth_map = MiDaS.__generate_image(img)
        name = f"PointCloud/depth/{__Config.output_file}"
        Tools.export_bytes_to_image(depth_map, name)
        exit()


def dry(case):
    print("########## Dry run ##########")
    __Config.output_file = ""
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


def redwood(file):
    print("Read Redwood dataset")
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    if file != "":
        print("loading files manually")
        color_raw = o3d.io.read_image(f"/home/ben/open3d_data/extract/SampleRedwoodRGBDImages/color/{file}.jpg")
        depth_raw = o3d.io.read_image(f"/home/ben/open3d_data/extract/SampleRedwoodRGBDImages/depth/{file}.png")
        print(depth_raw)
        print("------------------------------------------------------")
    else:
        color_raw = o3d.io.read_image(redwood_rgbd.color_paths[0])
        depth_raw = o3d.io.read_image(redwood_rgbd.depth_paths[0])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    print(rgbd_image)
    print(depth_raw)

    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])  # visualize the point cloud


def run(case):
    depth_map_to_point_cloud(case)
    # redwood("00000")


    # print("convertion")
    # float_value = 340282346638528859811704183484516925440.0000000000000000 / 2  # 9.232697
    # float_min = 0.0  # -340282346638528859811704183484516925440.0000000000000000
    # float_max = 340282346638528859811704183484516925440.0000000000000000
    # int_min = 0
    # int_max = 65535
    # round_off = False
    #
    # output = (float_value - float_min) * ((int_max - int_min) / (float_max - float_min)) + int_min
    # output = round(output) if round_off else output
    # print(output)
    # print(65535/2)


    # print("Load a ply point cloud, print it, and render it")
    # ply_point_cloud = o3d.data.PLYPointCloud()
    # pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    # print(pcd)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd],
    #                                  zoom=0.3412,
    #                                  front=[0.4257, -0.2125, -0.8795],
    #                                  lookat=[2.6172, 2.0475, 1.532],
    #                                  up=[-0.0694, -0.9768, 0.2024])

if __name__ == "__main__":
    run()
