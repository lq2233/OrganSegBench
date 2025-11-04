import time

import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib as mpl
import math
from collections import Counter


def read_stl_2_cloud(path_stl, N=1000000):
    target_ply = o3d.geometry.TriangleMesh()
    target_ply = o3d.io.read_triangle_mesh(path_stl)
    target_ply.compute_vertex_normals()
    pcd = target_ply.sample_points_uniformly(number_of_points=N)
    # 可视化点云模型
    # o3d.visualization.draw_geometries([pcd])

    V_mesh = np.asarray(target_ply.vertices)
    # F_mesh 为ply网格的面片序列，shape=(m,3)，这里m为此网格的三角面片总数，其实就是对顶点序号（下标）的一种组合，三个顶点组成一个三角形
    F_mesh = np.asarray(target_ply.triangles)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(V_mesh)


    return pcd


if __name__ == "__main__":


    # demo
    Num = 1000000
    path_target_label = "./Sample_Data/STL/4/Label_liver.stl"
    path_target_model = "./Sample_Data/STL/4/Model_liver.stl"

    mesh = o3d.io.read_triangle_mesh(path_target_label)
    o3d.visualization.draw_geometries([mesh])
    target_label_pcd = read_stl_2_cloud(path_target_label, N= Num)
    target_label_pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([target_label_pcd])

    mesh = o3d.io.read_triangle_mesh(path_target_model)
    o3d.visualization.draw_geometries([mesh])
    target_model_pcd = read_stl_2_cloud(path_target_model, N= Num)
    target_model_pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([target_model_pcd])







