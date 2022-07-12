from pyntcloud import PyntCloud
import numpy as np
import open3d

import ipywidgets as widgets
from ipywidgets import interact, interact_manual


class PointCloud:
    def __init__(self):
        # Load mesh / point cloud from file
        # self.point_cloud = PyntCloud.from_file("PointCloud/ankylosaurus_mesh.ply")
        self.model = PyntCloud.from_file("C3DC_BigMac.ply")

    def call(self):
        print("Call in PointCloud.py")

        self.model

        # Visualization: Mesh;  with the "plot" method
        # Note that we are passing the argument backend="threejs" because mesh visualziation is not currently supported
        # with the default backend.
        # Because of this, the visualization might don't work in some enviroments (i.e. binder). We are working to add
        # support to mesh plotting with the default backend as soon as possible in order to fix these issues
        self.model.plot(mesh=True, backend="threejs")

        # Convert Mesh to Point Cloud using random sampling
        # We can now convert the mesh into a point cloud by sampling 100.000 random points from the surface.
        # You cand learn more about hoy to convert a triangular mesh into a point cloud in the examples/[sampling]
        # notebooks.
        # point_cloud = self.model.get_sample("mesh_random", n=100, rgb=False, normals=False, as_PyntCloud=True)
        point_cloud = self.model
        # Note that we pass the normals=True argument so now our point cloud has normal values for each point.

        # Visualization: Pandas built-in
        # Because PyntCloud.points is a pandas.DataFrame, we can leverage the built-in visualization options of pandas:
        point_cloud.points[["x", "y", "z"]].plot(kind="hist", subplots=True)

        # Visualization: Point Cloud
        # We can visualize the sampled point cloud as follows:
        scene = point_cloud.plot(return_scene=True)
        # We can use the interactive widgets bellow the plot to dynamically adjust the background color and the point
        # size.

        # Custom scalar field with custom color map
        # We can select any of the scalar fields in a PyntCloud (any name in the DataFrame PyntCloud.points) and use it
        # to colorize the point cloud.
        # We do this with the `use_as_color` argument.
        # We can also select any of the [avaliable color maps in Matplotlib]
        # (https://matplotlib.org/examples/color/colormaps_reference.html) adjust the colorization.
        # We do this with the `cmap` argument.
        point_cloud.plot(use_as_color="x", cmap="gray")

        # Visualization: Multiple Point Clouds
        # We can visualize multiple point clouds on the seame scene.
        # First, we generate a new point cloud by randomly sampling 100 points:
        anky_cloud_sample = point_cloud.get_sample("points_random", n=100, as_PyntCloud=True)
        # We set the color of that sample to red for easier visualization:
        anky_cloud_sample.points["red"] = 255
        anky_cloud_sample.points["green"] = 0
        anky_cloud_sample.points["blue"] = 0
        # Now we plot the original point cloud and we use the return_scene=True argument to store the pythreejs scene
        # in a variable.
        scene = point_cloud.plot(initial_point_size=0.01, return_scene=True)
        # Now we can pass that variable using the `scene` argument when we call the plot function on our sampled point
        # cloud:
        anky_cloud_sample.plot(initial_point_size=0.05, scene=scene)


# Fist try to understand point clouds, and create a plane in the middle and delete one side
def remove(plane):
    # Read point cloud from PLY
    pcd1 = open3d.io.read_point_cloud("1.ply")
    points = np.asarray(pcd1.points)

    # Sphere center and radius
    center = np.array([1.586, -8.436, -0.242])
    radius = 0.5

    # Calculate distances to center, set new points
    distances = np.linalg.norm(points - center, axis=1)
    pcd1.points = open3d.utility.Vector3dVector(points[distances <= radius])

    # Write point cloud out
    return