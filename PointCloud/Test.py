import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import png


def convert_to_1_ch_grayscale(image_path):
    img = PIL.Image.open(image_path)
    image_array = np.asarray(img)  # np.uint16(img)
    image_pil = PIL.Image.fromarray(image_array, "I;16")  # image_array, "I;16"

    # plt.imshow(gray, cmap=plt.get_cmap('gray'))
    plt.imshow(image_pil)
    plt.show()

    new = 'depth/1_z-tmp.png'
    image_pil.save(new)

    with open(new, 'wb') as f:
        writer = png.Writer(width=image_array.shape[1], height=image_array.shape[0], bitdepth=16, greyscale=True)
        zgray2list = image_array.tolist()
        writer.write(f, zgray2list)

    return new


def depth_map_to_point_cloud():
    # color_img_path = "color/4.jpg"
    # depth_img_path = "depth/4_z-tmp.png"
    color_img_path = "PointCloud/color/1.jpg"
    depth_img_path = "PointCloud/depth/z_1_tmp.png"

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

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    plt.subplot(1, 2, 1)
    plt.title('Hand grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Hand depth image')
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
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])  # visualize the point cloud


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


def run():
    depth_map_to_point_cloud()
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