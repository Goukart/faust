import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image

# Maybe temp
import MiDaS
import Tools


# ToDo; Wip fix border or discard completely
def __generate_scale_image(width, height, data_type, _range=None):
    if _range is None:
            _range = range(0, height, 1)
    _from = Tools.limits(data_type).min
    _to = Tools.limits(data_type).max
    # generate gradient image from 0 to image height with given step
    scale = np.zeros((height, width), data_type)
    border_width = 1
    border_value = 1
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for i in range(_range.start, _range.stop):
        # print(i)
        row = np.ones((1, width), data_type) * (i * _range.step)
        row[0][-border_width:width] = np.ones((1, border_width), data_type) * border_value
        row[0][0:border_width] = np.ones((1, border_width), data_type) * border_value
        scale[i] = row
    scale[0:border_width] = np.ones((border_width, width), data_type) * border_value
    scale[-border_width:width] = np.ones((border_width, width), data_type) * border_value
    print(scale)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # exit()
    return scale


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
    # ToDo; auto detect image extension? or must be jpg? png is black in grayscale display of o3d
    color_img_path = f"PointCloud/color/{file}.jpg"
    depth_img_path = f"PointCloud/depth/z_{file}.png"

    color_raw = o3d.io.read_image(color_img_path)
    depth_raw = o3d.io.read_image(depth_img_path)
    print(depth_raw)
    print("Loaded color file: ", color_img_path)
    print("Loaded depth file: ", depth_img_path)

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
    # Image has to be spesific resolution (x,y)?
    # color and depth image must match in resolution, clor has to be jpg
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 300, 65000)
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
        model = "large"
        name = "dry_gen"
        regex = "PointCloud/color/face.jpg"
        MiDaS.generate_dms(regex, model, name)
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
        depth_map = MiDaS.generate_dms(img, "large", name)

        Tools.export_bytes_to_image(depth_map, name)
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
