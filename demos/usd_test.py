# 





from pxr import Usd, Vt


import typing
from pxr import Usd, UsdGeom, Gf

def get_world_transform_xform(prim: Usd.Prim) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """
    Get the local transformation of a prim using Xformable.
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    print("here is the glass transform:", world_transform)
    print("\n\n")
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale, world_transform


stage = Usd.Stage.Open('data/meshes/glass.usda')
xform = stage.GetPrimAtPath('/Boundaries')
glass = stage.GetPrimAtPath('/Boundaries/Glass/glass3')
# xform = stage.GetPrimAtPath('/Sources')
# glass = stage.GetPrimAtPath('/Sources/Ball/Sphere_002')

print(xform.GetPropertyNames())
print(glass.GetPropertyNames())

print()
translation, rotation, scale, world_transform = get_world_transform_xform(glass)


pointsAttr = glass.GetAttribute('points')
normalsAttr = glass.GetAttribute('normals')
facesAttr = glass.GetAttribute('faceVertexIndices')
countsAttr = glass.GetAttribute('faceVertexCounts')
# extentAttr.Set(extentAttr.Get() / 2)
# print(type(pointsAttr.Get()))

import numpy as np
points = pointsAttr.Get()
# points = np.array(pointsAttr.Get())
normals = normalsAttr.Get()
# normals = np.array(normalsAttr.Get())[:]
faces = np.array(facesAttr.Get())
counts = np.array(countsAttr.Get())


print("counts")
countsidx = np.cumsum(counts)-counts[0]
print(counts.shape, countsidx, counts[:20])

print("faces")
# facesidx = faces[countsidx]
facesidx = faces
print(faces.shape, faces.shape[0]/3, np.max(faces), facesidx)
zeros = np.argwhere(faces==0)

print("points")
mat = np.array(( (0.11517247557640076, 0, 0, 0), (0, 0.11517247557640076, 0, 0), (0, 0, 0.11517247557640076, 0), (-0.28247377276420593, 0.248969167470932, 2.2025229930877686, 1) ))
print("Matrix:", mat)


# points = np.hstack((points, np.zeros((points.shape[0], 1)))) @ mat[:,:-1]
print("type", type(points))
print("äll atributes", dir(points), np.array(points).shape, points.__len__())

points = [world_transform.Transform(points[i]) for i in range(np.array(points).shape[0])]

points = np.array(points)

newpoints = np.array([points[i] for i in facesidx])
print(points.shape)


print("normals")
# normals = np.hstack((normals, np.zeros((normals.shape[0], 1)))) @ mat[:,:-1]


# normals = [world_transform.Transform(normals[i]) for i in range(np.array(normals).shape[0])]      ## this works !

normals = [world_transform.TransformDir(normals[i]) for i in range(np.array(normals).shape[0])]

normals = np.array(normals)

newnormals = normals[:]
# newnormals = normals[facesidx]

print(normals.shape, normals[zeros])
# print(normals.shape[0]/points.shape[0])
# print(faces.shape[0]/3)
print(np.unique(normals[:,0], axis=0).shape)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

print(normals.shape)
print(normals[normals[:,0]<0.0].shape)

# start, end = 800, 2800
# newpoints = newpoints[start:end]
# newnormals = newnormals[start:end]
# points = points[start:end]


# ax.scatter(points[:,0], points[:,1], points[:,2], marker='o', color='r')
ax.scatter(newpoints[:,0], newpoints[:,1], newpoints[:,2], marker='o', color='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')





# ax = fig.add_subplot(projection='3d')
q = ax.quiver(newpoints[:,0], newpoints[:,1], newpoints[:,2], newnormals[:,0], newnormals[:,1], newnormals[:,2], label="normals", normalize=True, alpha=0.5, length = 0.05)
# ax.quiverkey(q, X=0.5, Y=1.1, U=1, label='Normals', labelpos='E')
# ax.scatter(x=coords[:, 0], y=coords[:, 1], c="m", **kwargs)

plt.show()
