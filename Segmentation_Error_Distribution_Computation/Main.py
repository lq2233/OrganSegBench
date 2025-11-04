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
def find_extreme_points(pcd):
    """找到点云的最左侧、最右侧、最上侧点（基于XYZ坐标）"""
    points = np.asarray(pcd.points)
    left = points[np.argmin(points[:, 0])]  # X最小值
    right = points[np.argmax(points[:, 0])]  # X最大值
    # top = points[np.argmax(points[:, 2])]  # Z最大值（假设垂直方向）
    down = points[np.argmin(points[:, 2])]  # Z最大值（假设垂直方向）
    front = points[np.argmin(points[:, 1])]  # Y最小值（假设垂直方向）
    behind = points[np.argmax(points[:, 1])]  # Y最大值
    # return np.array([left, right, top, front, behind])
    # return np.array([left, right, front, behind])
    return np.array([left, right, down, front, behind])


def compute_transform(src_points, tgt_points):
    """通过对应点计算刚体变换矩阵（SVD方法）"""
    # 计算质心
    src_centroid = np.mean(src_points, axis=0)
    tgt_centroid = np.mean(tgt_points, axis=0)

    # 去中心化
    src_centered = src_points - src_centroid
    tgt_centered = tgt_points - tgt_centroid

    # 计算协方差矩阵
    H = src_centered.T @ tgt_centered

    # SVD分解
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算平移
    t = tgt_centroid - R @ src_centroid

    # 构建变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform

import random

def random_down_sample(cloud, sample_points):
    A = np.asarray(cloud.points)
    sampleA = random.sample(range(A.shape[0]), sample_points)
    sampled_cloud = cloud.select_by_index(sampleA)
    return sampled_cloud


def visualize_registration(source, target, transformed, corr_lines=None):
    """可视化配准结果"""
    source.paint_uniform_color([1, 0, 0])  # 红色：原始源点云
    target.paint_uniform_color([0, 0, 1])  # 蓝色：目标点云
    transformed.paint_uniform_color([0, 1, 0])  # 绿色：变换后的源

    geometries = [source, target, transformed]
    if corr_lines:
        geometries.append(corr_lines)

    # o3d.visualization.draw_geometries(
    #     geometries,
    #     window_name="点云配准结果",
    #     width=800,
    #     height=600
    # )
    return source, target, transformed


def create_correspondence_lines(src_points, tgt_points, color=[1, 0, 0]):
    """创建对应点连线"""
    # print(len(src_points), len(tgt_points))
    assert len(src_points) == len(tgt_points)

    # 合并点坐标
    points = np.vstack([src_points, tgt_points])

    # 创建线段索引（每个对应点对）
    lines = [[i, i + len(src_points)] for i in range(len(src_points))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def manual_icp_registration(source_pcd, target_pcd):
    # 步骤1：查找特征点
    src_corr = find_extreme_points(source_pcd)
    tgt_corr = find_extreme_points(target_pcd)

    # 步骤2：计算变换矩阵
    transform = compute_transform(src_corr, tgt_corr)

    # 步骤3：应用变换
    transformed_pcd = copy.deepcopy(source_pcd)
    transformed_pcd.transform(transform)

    # 步骤4：创建对应点连线（显示变换后的对应关系）
    src_transformed_corr = np.asarray(transformed_pcd.points)[
        [np.argmin(np.asarray(source_pcd.points)[:, 0]),
         np.argmax(np.asarray(source_pcd.points)[:, 0]),
         # np.argmax(np.asarray(source_pcd.points)[:, 2]),
         np.argmin(np.asarray(source_pcd.points)[:, 2]),
         np.argmin(np.asarray(source_pcd.points)[:, 1]),
         np.argmax(np.asarray(source_pcd.points)[:, 1]),
         ]
    ]
    corr_lines = create_correspondence_lines(
        src_transformed_corr,
        tgt_corr,
        color=[1, 0.5, 0]  # 橙色连线
    )

    # 可视化结果
    source, target, transformed = visualize_registration(source_pcd, target_pcd, transformed_pcd, corr_lines)

    return transform, source, target, transformed

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


def registration_pipeline(source_pcd, target_pcd):
    estimated_transform, source_pcd, target_pcd, transformed_pcd = manual_icp_registration(source_pcd, target_pcd)
    x2 = np.asarray(transformed_pcd.points)[:, 0]
    y2 = np.asarray(transformed_pcd.points)[:, 1]
    z2 = np.asarray(transformed_pcd.points)[:, 2]
    # print(np.mean(x2), np.mean(y2), np.mean(z2))

    x1 = np.asarray(target_pcd.points)[:, 0]
    y1 = np.asarray(target_pcd.points)[:, 1]
    z1 = np.asarray(target_pcd.points)[:, 2]
    # print(np.mean(x1), np.mean(y1), np.mean(z1))

    delta_x = np.mean(x1) - np.mean(x2)
    delta_y = np.mean(y1) - np.mean(y2)
    delta_z = np.mean(z1) - np.mean(z2)

    transformed_pcd.translate((delta_x, delta_y, delta_z), relative=True)

    return estimated_transform, source_pcd, target_pcd, transformed_pcd

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

import os

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def self_dist(pcd_target, pcd_source):

    target_points = np.asarray(pcd_target.points)
    index_list = []
    source_points = np.asarray(pcd_source.points)
    count = 0
    # print(len(target_points))
    for target_point in target_points:
        print(count)
        min_dist = 9999999999
        index = -1
        for idx, source_point in enumerate(source_points):
            dist_point2point = euclidean_distance(target_point, source_point)
            if dist_point2point<min_dist:
                min_dist=dist_point2point
                index = idx
        count = count + 1
        print(index)
        index_list.append(index)

    return index_list


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

# def find_nearest_neighbors(source, target):
#     """
#     为target点云中的每个点找到source点云中的最近邻索引
#     :param source: open3d.geometry.PointCloud 源点云
#     :param target: open3d.geometry.PointCloud 目标点云
#     :return: 包含最近邻索引的numpy数组（形状为(N,)）
#     """
#     # 构建源点云的KDTree
#     source_kdtree = o3d.geometry.KDTreeFlann(source)
#
#     # 获取目标点云坐标
#     target_points = np.asarray(target.points)
#
#     # 存储最近邻索引的数组
#     nearest_indices = np.zeros(len(target_points), dtype=int)
#
#     # 遍历所有目标点
#     for i, point in enumerate(target_points):
#         # 查询最近邻（k=1）
#         [k, idx, _] = source_kdtree.search_knn_vector_3d(point, 1)
#         nearest_indices[i] = idx[0]
#
#     return nearest_indices
def mls_upsampling(pcd, radius=0.05, upsampling_ratio=2.0):
    """
    使用移动最小二乘法(MLS)进行点云上采样

    参数:
    pcd: 输入点云
    radius: MLS拟合的邻域半径
    upsampling_ratio: 上采样倍数

    返回:
    上采样后的点云
    """
    # 计算原始点云的点数
    original_points = np.asarray(pcd.points).shape[0]

    # 执行MLS上采样
    upsampled_pcd = pcd.compute_moving_least_square(
        radius=radius,
        upsampling_method="moving_least_squares",
        upsampling_factor=upsampling_ratio
    )

    # 计算上采样后的点数
    new_points = np.asarray(upsampled_pcd.points).shape[0]
    print(f"上采样前点数: {original_points}, 上采样后点数: {new_points}")

    return upsampled_pcd
if __name__ == "__main__":

    path_target = "./reference/label_liver.stl"
    path_target_label = "./reference/label_liver.stl"
    # path_target_label = "C:/Users/acer/Desktop/pointcloud_data/demo_data_liver/01190718V009/LABEL_liver.stl"
    path_target_model ="./reference/label_liver.stl"
    Num = 1000000
    target_label_pcd = read_stl_2_cloud(path_target_label, N= Num)
    target_model_pcd = read_stl_2_cloud(path_target_model, N= Num)
    print(np.asarray(target_label_pcd.points).shape)
    # distance
    dists_target = target_label_pcd.compute_point_cloud_distance(target_model_pcd)  # 计算点云与点云之间的距离
    dists_target = np.asarray(dists_target)

    # draw_distance(target_label_pcd, dists_target)

    # IDs = ['01190626V001', '01200408V001', '01190718V003']

    IDs = sorted(os.listdir('./Sample_Data/STL/'))
    count = 1
    result = []

    yingshe = []
    for id in IDs[:]:
        print(id)
        if id == '0':  # skip
            continue
        # print(len(dists))
        # draw_distance(target_label_pcd, dists_target)

        # source_pcd = read_stl_2_cloud("E:/the_fourth_paper_pointcloud_data/STLData/liver/"+id+"/REF_label_liver.stl")
        # path_source = "E:/the_fourth_paper_pointcloud_data/STLData/liver/"+id+"/REF_label_liver.stl"
        path_source_label = "./Sample_Data/STL/"+id+"/Label_liver.stl"
        path_source_model = "./Sample_Data/STL/"+id+"/Model_liver.stl"

        source_label_pcd = read_stl_2_cloud(path_source_label, N= Num)
        source_model_pcd = read_stl_2_cloud(path_source_model, N= Num)

        dists_source = source_label_pcd.compute_point_cloud_distance(source_model_pcd)  # 计算点云与点云之间的距离
        dists_source = np.asarray(dists_source)
        print('正向最大', np.max(dists_source))

        dists_source_R = source_model_pcd.compute_point_cloud_distance(source_label_pcd)  # 计算点云与点云之间的距离
        dists_source_R = np.asarray(dists_source_R)
        print('负向最大', np.max(dists_source_R))

        # 将反向的映射至正向的
        source_pcd_R = source_model_pcd

        indices_R = find_nearest_neighbors(source_label_pcd, source_pcd_R)


        # 95筛选
        dists_all = np.hstack((dists_source, dists_source_R))
        threshold = np.percentile(dists_all, 95)
        dists_source[dists_source>=threshold]=threshold
        dists_source_R[dists_source_R>=threshold]=threshold

        # print(Counter(indices_left_R))
        counter = dict(Counter(indices_R))
        for i in range(len(indices_R)): # 相加
            dists_source[indices_R[i]] = (dists_source[indices_R[i]] + dists_source_R[i])
            # if dists_source_left_R[i]>=dists_source_left[indices_left_R[i]]:
            #     dists_source_left[indices_left_R[i]] = dists_source_left_R[i]
        for key in counter: # 取平均
            dists_source[key] /= counter[key]+1

        #################################################
        print('per item', np.mean(dists_source), np.max(dists_source))
        result.append(np.mean(dists_source))
        #################################################

        target_pcd = read_stl_2_cloud("./reference/label_liver.stl", N= Num)

        source_pcd = read_stl_2_cloud("./Sample_Data/STL/"+id+"/Label_liver.stl", N= Num)

        # 执行配准
        estimated_transform, source_pcd, target_pcd, transformed_pcd = registration_pipeline(source_pcd, target_pcd)

        # o3d.visualization.draw_geometries([transformed_pcd, target_pcd])

        # 进行点的对应
        time1 = time.time()
        indices = find_nearest_neighbors(transformed_pcd, target_pcd)

        # indices = find_nearest_neighbors(target_pcd, transformed_pcd)
        time2 = time.time()
        print(time2 - time1)

        # yingshe.extend(list(indices))
        for i in range(len(indices)):
            dists_target[i] = dists_target[i] + dists_source[indices[i]]
        count =  count + 1

        # point1 = np.asarray(target_pcd.points)[i]
        # point2 = np.asarray(transformed_pcd.points)[indices[i]]

    # dists_target = dists_target/(len(IDs)+1)
    ###################################################################
    print('10 items', np.mean(result))
    ###################################################################
    dists_target = dists_target/ count

    # counter = dict(Counter(yingshe))
    # for key in counter:  # 取平均
    #     dists_target[key] /= counter[key] + 1

    print(count)
    np.save("./OutPut/AVG_dists_target.npy", dists_target)
    draw_distance(target_pcd, dists_target)

    # index_list = self_dist(target_pcd, transformed_pcd)




