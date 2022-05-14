import numpy as np
import trimesh


# =============================================================================================================
# TODO Compute Chamfer distance
# import torch
from sklearn.neighbors import NearestNeighbors

metric='l2'


# # ===== Beard man ====
shape_path = "/scratch/beard_man_related/gt/beard_man.obj"            # GT - Beard man
gt_shape_path = "/scratch/codes/3d_shapes/dhfmagic-main/mesher/cmake-build-debug/verts.obj"     # DeepSDF (256) - Beard man



mesh = trimesh.load(shape_path)
gt_mesh = trimesh.load(gt_shape_path)

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

mesh = scale_to_unit_sphere(mesh)
gt_mesh = scale_to_unit_sphere(gt_mesh)

##### TODO Output normalized gt && output shapes
gt_mesh.export("/scratch/data/david_normalized.ply")
# mesh.export("/scratch/beard_man_related/normalized/gt_normalized.ply")

samples = mesh.sample(1000000)		### Sample 1000000 points for computation
gt_samples = gt_mesh.sample(1000000)		### 

x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(samples)
min_y_to_x = x_nn.kneighbors(gt_samples)[0]
y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(gt_samples)
min_x_to_y = y_nn.kneighbors(samples)[0]
cdist = np.mean(min_y_to_x) + np.mean(min_x_to_y)

print("Chamfer distance: {}".format(cdist))

test = 1
