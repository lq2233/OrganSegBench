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
import random




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
    # return target_pcd



def color_index(dist, step):
    for i in range(256):
        if dist<(i+1)*step:
            return i
    return 255

def draw_distance(pcd, distance):
    max1 = np.max(distance)
    min1 = np.min(distance)

    color_select = ['red', 'orange', 'yellow', 'green', 'blue']
    len1 = (max1 - min1) / (max1 - min1)
    step = len1 / (len(color_select) - 1)
    red1 = 0

    orange1 = red1 + step

    yellow1 = orange1 + step

    green1 = yellow1 + step

    blue1 = green1 + step
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    color_dict = [(red1, 'blue'), (orange1, 'green'), (yellow1, 'yellow'), (green1, 'orange'), (blue1, 'red')]
    cmap = LinearSegmentedColormap.from_list("roygb", color_dict)  # 256个颜色
    fig, ax = plt.subplots(figsize=(4, 1.5))
    # fig.subplots_adjust(bottom=0.5)   # 设置子图到下边界的距离
    norm = mpl.colors.Normalize(vmin=min1, vmax=max1)

    # font = {'family' : 'serif',
    #         'color'  : 'darkred',
    #         'weight' : 'normal',
    #         'size'   : 16,
    #         }

    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #              cax=ax, orientation='vertical', ticks=np.linspace(min1, max1, 3)) # 竖直方向

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=ax, orientation='horizontal', ticks=np.linspace(min1, max1, 3))  # 竖直方向
    cbar.ax.yaxis.set_ticks_position('left')
    # tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    # cb1.locator = tick_locator
    # cb1.set_ticks([np.min(data), 0.25,0.5, 0.75, np.max(data)])
    # cb1.update_ticks()
    plt.tight_layout()
    plt.show()

    # 初始化颜色
    color_green = [[0.0, 1.0, 0.0]]
    color_green_list = np.tile(color_green, (len(distance), 1))
    pcd.colors = o3d.utility.Vector3dVector(color_green_list)

    print(cmap.N)  # 256
    color_list = [[cmap(i)[0], cmap(i)[1], cmap(i)[2]] for i in range(cmap.N)]
    print(color_list)


    # 设置将dists设置成256个段

    step_distance = (max1 - min1) / 255
    for i in range(len(distance)):
        index = color_index(distance[i], step_distance)
        this_red = cmap(index)[0]
        this_green = cmap(index)[1]
        this_blue = cmap(index)[2]
        # 改变颜色
        this_Color = np.asarray(pcd.colors)

        this_Color[i, 0] = this_red
        this_Color[i, 1] = this_green
        this_Color[i, 2] = this_blue

    pcd.colors = o3d.utility.Vector3dVector(this_Color)
    o3d.visualization.draw_geometries([pcd])




def find_nearest_neighbors(source, target):
    """
    为target点云中的每个点找到source点云中的最近邻索引
    :param source: open3d.geometry.PointCloud 源点云
    :param target: open3d.geometry.PointCloud 目标点云
    :return: 包含最近邻索引的numpy数组（形状为(N,)）
    """
    # 构建源点云的KDTree
    source_kdtree = o3d.geometry.KDTreeFlann(source)

    # 获取目标点云坐标
    target_points = np.asarray(target.points)

    # 存储最近邻索引的数组
    nearest_indices = np.zeros(len(target_points), dtype=int)

    # 遍历所有目标点
    for i, point in enumerate(target_points):
        # 查询最近邻（k=1）

        # print(i)


        [k, idx, _] = source_kdtree.search_knn_vector_3d(point, 1)
        nearest_indices[i] = idx[0]

    return nearest_indices


if __name__ == "__main__":

    path_target = "./Sample_Data/STL/5/Label_liver.stl"
    path_target_label = "./Sample_Data/STL/5/Label_liver.stl"
    path_target_model = "./Sample_Data/STL/5/Model_liver.stl"
    Num = 1000000
    target_label_pcd = read_stl_2_cloud(path_target_label, N= Num)
    target_model_pcd = read_stl_2_cloud(path_target_model, N= Num)
    print(np.asarray(target_label_pcd.points).shape)
    # distance
    dists_target = target_label_pcd.compute_point_cloud_distance(target_model_pcd)  # 计算点云与点云之间的距离
    dists_target = np.asarray(dists_target)

    dists_target_R = target_model_pcd.compute_point_cloud_distance(target_label_pcd)  # 计算点云与点云之间的距离
    dists_target_R = np.asarray(dists_target_R)
    print('负向最大', np.max(dists_target_R))

    # 将反向的映射至正向的
    source_pcd_R = target_model_pcd

    indices_R = find_nearest_neighbors(target_label_pcd, source_pcd_R)

    # 95筛选
    dists_all = np.hstack((dists_target, dists_target_R))
    threshold = np.percentile(dists_all, 95)
    dists_target[dists_target >= threshold] = threshold
    dists_target_R[dists_target_R >= threshold] = threshold

    # print(Counter(indices_left_R))
    counter = dict(Counter(indices_R))
    for i in range(len(indices_R)):  # 相加
        dists_target[indices_R[i]] = (dists_target[indices_R[i]] + dists_target_R[i])
        # if dists_source_left_R[i]>=dists_source_left[indices_left_R[i]]:
        #     dists_source_left[indices_left_R[i]] = dists_source_left_R[i]
    for key in counter:  # 取平均
        dists_target[key] /= counter[key] + 1

    draw_distance(target_label_pcd, dists_target)






