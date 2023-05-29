#%%
import typing
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from pxr import Usd, UsdGeom, Gf, Vt


class SPHGeom:
    def __init__(self):
        self.sources = None
        self.boundaries = None
        self.sinks = None



class CubeGeom(SPHGeom):

    def __init__(self, x_lim=2*np.pi, y_lim=2*np.pi, z_lim=2*np.pi, halfres=None):
        super().__init__()
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim

        if halfres is not None:
            self.init_meshgrid(halfres)

    def init_meshgrid(self, halfres):
        """
        produces grid of 2^halfres x 2^halfres x 2^halfres number of particles
        """
        x_ = np.linspace((self.x_lim/(2**halfres)), self.x_lim, 2**halfres, endpoint=True)
        y_ = np.linspace((self.y_lim/(2**halfres)), self.y_lim, 2**halfres, endpoint=True)
        z_ = np.linspace((self.y_lim/(2**halfres)), self.z_lim, 2**halfres, endpoint=True)

        meshgrid = np.meshgrid(x_, y_, z_, indexing='ij')

        self.meshgrid = np.array(meshgrid).T.reshape(-1,3)
        self.N = self.meshgrid.shape[0]

        return self.meshgrid







MeshPoints = namedtuple('MeshPoints', ['counts', 'faces', 'points', 'normals'])

def transform_coords_xform(prim: Usd.Prim, points:Vt.Vec3fArray, normals:Vt.Vec3fArray) -> typing.Tuple[np.ndarray, np.ndarray]:
    world_transform = get_world_transform_xform(prim)
    newpoints = [world_transform.Transform(points[i]) for i in range(points.__len__())]
    newnormals = [world_transform.TransformDir(normals[i]) for i in range(normals.__len__())]
    return np.array(newpoints), np.array(newnormals)

def get_world_transform_xform(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Get the world transformation of a prim using Xformable.
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_xformable.html
    or https://docs.omniverse.nvidia.com/prod_usd/prod_kit/programmer_ref/usd/transforms/get-world-transforms.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        The global transformation as a Matrix4d
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    return world_transform

def get_local_transform_xform(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Get the local transformation of a prim using Xformable.
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the local transformation.
    Returns:
        The local tranformation of that matrix as a Mtrix4d
    """
    xform = UsdGeom.Xformable(prim)
    local_transformation: Gf.Matrix4d = xform.GetLocalTransformation()
    return local_transformation



class USDAGeom(SPHGeom):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.stage = Usd.Stage.Open(self.filename)
        self.boundaries = self.get_children_meshpoints('/Boundaries')
        self.sources = self.get_children_meshpoints('/Sources')


    def get_children_meshpoints(self, primname: str) -> typing.List[MeshPoints]:
        root = self.stage.GetPrimAtPath(primname)
        children: Usd.Prim = root.GetChildren()
        children_mesh = [child.GetChildren()[0] for child in children if child.GetChildren()[0].IsA(UsdGeom.Mesh)]

        mespoints_list = []
        for child in children_mesh:
            # print(bd.GetPath())
            counts = np.array(child.GetAttribute('faceVertexCounts').Get())
            faces = np.array(child.GetAttribute('faceVertexIndices').Get())
            points = child.GetAttribute('points').Get()
            normals = child.GetAttribute('normals').Get()

            newpoints, newnormals = transform_coords_xform(child, points, normals)
            meshpoints = MeshPoints(counts, faces, newpoints, newnormals)
            mespoints_list.append(meshpoints)

        return mespoints_list



    def visualize(self, how="mpl"):
        """
        visualize the geometry in usdview if available, otherwise use matplotlib
        """
        exit_status = 0

        if how=="usdview":
            import os
            print("Running comand: 'usdview' "+self.filename)
            exit_status = os.system("usdview "+self.filename)

        if how=="matplotlib" or exit_status != 0:
            print("usdview not available, using matplotlib")
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            for meshpoints in self.boundaries + self.sources:
                points = np.array([meshpoints.points[i] for i in meshpoints.faces])
                normals = meshpoints.normals

                ax.scatter(points[:,0], points[:,1], points[:,2], marker='o', label="points", color='r')

                q = ax.quiver(points[:,0], points[:,1], points[:,2], normals[:,0], normals[:,1], normals[:,2], label="normals", normalize=True, alpha=0.5, length=0.05)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # plt.legend()
            # plt.show()
            return ax





if __name__=="__main__":
    geometry = USDAGeom("../demos/data/meshes/glass.usda")
    geometry.visualize()
