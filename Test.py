import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image

# Maybe temp
import MiDaS
import Tools


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


def rgbd_to_pc(file: str) -> o3d.geometry.PointCloud:
    # read image pair
    # Seems like rgb image has to be rgb, or have at least 3 channels?
    # ToDo; auto detect image extension? or must be jpg? png is black in grayscale display of o3d
    # ToDo: decide if either two folders and exact name or same folder but some prefix, change in MiDaS.py as well
    color_img_path = f"PointCloud/color/{file}.jpg"
    depth_img_path = f"PointCloud/depth/z_{file}.png"
    color_raw = o3d.io.read_image(color_img_path)
    depth_raw = o3d.io.read_image(depth_img_path)
    print("Loaded color file: ", color_img_path)
    print("Loaded depth file: ", depth_img_path)

    # Create image with depth from depth map and color image
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 0.001, 300.)
    # Conditions: color and depth image must match in resolution, clor has to be jpg
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 1.0, 65000)
    # ToDo: understand what parameter does what / has which effect and how to determine the optimal ones
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 65000, 0.01)

    # Configure pinhole camera to calculate points in 3d space
    # At this point i could also utilize the image metadata to get more accurate results?; use metadate to configure
    # pinhole camera
    camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    image = PIL.Image.open(depth_img_path)
    # fetching the dimensions
    # wid, hgt = img.size
    camera.width, camera.height = image.size
    # A flat / all black depth map could be, because by depth_scale or depth_trunc being too low. Both seem to have no
    # effect until 0.01, smaller is all black bigger make no difference
    # create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=True)
    print("camera properties: ")
    print(camera)
    print("#################################################################")

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)
    # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, camera)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def depth_map_to_point_cloud(file: str):
    # ############### Test zone ###############
    # new_depth = convert_to_1_ch_grayscale(depth_img_path)
    # depth = o3d.io.read_image(new_depth)
    # print(depth)

    # print(depth_raw)
    # #########################################

    # Load image (with PIL)as np array
    #image = Image.open(f"{file}.png")
    #image_array = np.asarray(image)
    #print(image_array.shape)
    #print(image_array)

    pcd = rgbd_to_pc(file)
    o3d.visualization.draw_geometries([pcd])

    exit(1)

    # Open 3D stuff
    # plt.subplot(1, 2, 1)
    # plt.title('grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    pcd = rgbd_to_pc(file)
    pcd2 = o3d.io.read_point_cloud("PointCloud/C3DC_BigMac.ply")

    # visualize the point cloud
    print(pcd)
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd2])

    # Merge
    # compute_point_cloud_distance
    # orient_normals_towards_camera_location
    o3d.visualization.draw_geometries([pcd, pcd2])

    # Write to file
    # pcd.to_file("l1.ply", internal=["points", "mesh"])
    # o3d.io.write_point_cloud("l1.ply", pcd)
    # o3d.io.write_point_cloud("l2.ply", pcd2)


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
