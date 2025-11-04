#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File ：mesh_generate_by_itk.py
@Author ：xiongminghua
@Email ：xiongminghua@zyheal.com
@Date ：2022-03-08 14:03 
"""
import os
import time

import itk
import nibabel as nib
import vtk


def nifti2stl(nifti_file_path: str, stl_file_path:str, thresh: float = 1, mesh_save_path: str = None, reference_path: str = None):
    """
    面片数据生成器
    Args:
        nifti_file_path:
       stl_save_path:

    Returns:
        surface:type vtk.vtkPolyData.
    """
    # 统计时间
    # t_start = time.clock()
    if not os.path.exists(nifti_file_path):
        raise FileNotFoundError(nifti_file_path + ' does not exist.')
    if not mesh_save_path:
        if '.nii.gz' in nifti_file_path:
            mesh_save_path = nifti_file_path.replace('.nii.gz', '.vtk')
        elif '.nii' in nifti_file_path:
            mesh_save_path = nifti_file_path.replace('.nii', '.vtk')
        else:
            raise TypeError(nifti_file_path + '文件类型错误!')
    # 重写方向矩阵(避免产生非正交错误)
    image_info_ref = nib.load(reference_path)
    image_info = nib.load(nifti_file_path)
    qform = image_info_ref.get_qform()
    # print('qform')
    # print(qform)
    image_info.set_qform(qform)
    sform = image_info_ref.get_sform()
    # print('sform')
    # print(sform)
    # print('affine')
    # print(image_info.affine)
    image_info.set_sform(sform)
    nib.save(image_info, nifti_file_path)
    # 读取文件
    pixel_type = itk.UC
    dimensions = 3
    image_type = itk.Image[pixel_type, dimensions]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(nifti_file_path)
    try:
        reader.Update()
    except:
        raise FileExistsError('ITK 读取文件失败!')

    cast = itk.CastImageFilter[image_type, image_type].New(reader)

    # 阈值处理
    threshold = itk.BinaryThresholdImageFilter[image_type, image_type].New()
    threshold.SetInput(cast.GetOutput())
    threshold.SetLowerThreshold(thresh)
    threshold.SetOutsideValue(0)

    # 抽取面片数据
    mesh_type = itk.Mesh[itk.D, dimensions]
    mesh_filter = itk.BinaryMask3DMeshSource[image_type, mesh_type].New()
    mesh_filter.SetInput(threshold.GetOutput())
    mesh_filter.SetObjectValue(255)
    # 写入stl文件
    writer = itk.MeshFileWriter[mesh_type].New()
    writer.SetFileName(mesh_save_path)
    writer.SetInput(mesh_filter.GetOutput())
    # writer.SetFileTypeAsBINARY()
    writer.Update()
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_save_path)
    reader.Update()
    # stl_file_path = mesh_save_path.replace('.vtk', '.stl')
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_file_path)
    writer.SetInputData(reader.GetOutput())
    writer.SetFileTypeToBinary()
    writer.Write()
    # 移除中间文件
    if os.path.exists(mesh_save_path):
        os.remove(mesh_save_path)
    # print('nii2stl successed,', time.clock() - t_start)

    return stl_file_path


if __name__ == '__main__':
    organ = 'liver'
    # nifti_file = 'E:/第四篇文章/Code/Code_正文图像绘制V2/点云可视化/demo_data_liver'
    # stl_path = 'E:/第四篇文章/Code/Code_正文图像绘制V2/点云可视化/demo_data_liver'
    nifti_file = './Sample_Data/NII/'
    stl_path = './Sample_Data/STL/'
    if not os.path.exists(stl_path):
        os.mkdir(stl_path)

    dir_list = sorted(os.listdir(nifti_file))
    count = 0
    for dir in dir_list:
        count = count + 1
        if count>20:
            break
        if os.path.isdir(nifti_file + '/' + dir):

            if not os.path.exists(stl_path + '/' + dir):
                os.mkdir(stl_path + '/' + dir)

            nifti_file_sub = nifti_file + '/' + dir
            reference_path = nifti_file + '/' +dir + '/REF_label_'+organ+'.nii.gz'
            NII_list = os.listdir(nifti_file_sub)
            for NII in NII_list:
                nifti_file_nii = nifti_file_sub+'/'+NII
                name = NII.split('.')[0]
                if not os.path.exists(stl_path + '/' + dir + '/' + name+'.stl'):
                    save_path = stl_path + '/' + dir + '/' + name+'.stl'
                    print('source NII path', nifti_file_nii)
                    print('target STl path', save_path)
                    try:
                        _ = nifti2stl(nifti_file_nii, save_path, reference_path = reference_path)
                    except:
                        continue


