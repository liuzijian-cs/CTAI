from torch.utils import data

import os
import cv2 as cv
import SimpleITK as sitk
import torch
import numpy as np


# Note:SimpleITK 是一个用于医学图像处理的开源库，提供了简单易用的程序接口来访问 ITK
# ITK (Insight Segmentation and Registration Toolkit) 的功能。ITK 是一个广泛用于医学图像处理、分割和注册的高级开源库。


def get_file_list(data_path, get_dice=False):
    """
    根据 data_path 返回每一个数据的路径
    :param data_path: 数据集路径
    :param include_all_phases: 是否使用所有数据进行训练
    :param get_dice: false - 只返回 image_list
    """
    image_list, mask_list, id_list = [], [], []
    dir_list = [os.path.join(data_path, i) for i in os.listdir(data_path)]  # 遍历数据目录获取文件名列表
    for dir_path in dir_list:  # 遍历所有文件名
        phase_path = os.path.join(dir_path, 'arterial phase')  # 访问动脉目录
        if os.path.isdir(phase_path):  # 如果存在当前目录
            for filename in os.listdir(phase_path):  # 遍历当前目录下所有文件
                file_path = os.path.join(phase_path, filename)  # 拼接文件path 如：ct_data/1001/arterial phase/10001.dcm
                if ".dcm" in file_path:
                    patient_id = file_path.split(os.sep)[-3]  # 病人ID，使用 os.sep 来作为路径分隔符，跨平台的方式Linux/windows。
                    image_id = os.path.basename(file_path).replace('.dcm', '')
                    image_list.append((file_path, patient_id, image_id))
                elif "_mask" in file_path:
                    mask_list.append(file_path)
    if get_dice:
        return image_list, mask_list, id_list
    return image_list


def normalize_data(data):
    """
    :param data: 输入数据，进行最大最小值归一化
    """
    if not data.any():  # data.any(): 检查 data 数组中是否至少有一个元素是非零的。
        return data
    return (data - data.min()) / (data.max() - data.min())


def get_dataset(data_path, have=True):
    """
    :param data_path:
    :param have: 是否对遮罩数据 (mask_array) 进行存在性检查
    """
    image_data, mask_data = [], []

    image_list = get_file_list(data_path)  # return image_list - (file_path, patient_id, image_id)

    for image_path in image_list:
        # mask data
        mask_path = image_path[0].replace('.dcm', '_mask.png')
        mask_array = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if have:  # 检查是否有匹配的mask是否存在，如果不存在则跳过当前image
            if not mask_array.any():
                continue
        mask_array = normalize_data(mask_array)  # 最大最小值归一化
        mask_tensor = torch.from_numpy(mask_array).float()  # 将numpy数组转换为张量
        mask_id = os.path.basename(image_path[0]).replace('_mask.png', '')
        mask_data.append((mask_id, mask_tensor))

        # image data:
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path[0]))
        # ROI: ROI（Region of Interest）指的是图像或数据中感兴趣的区域
        roi_mask = np.zeros(shape=image_array.shape)  # 创建了与“image_array”形状相同的零数组
        roi_mask_mini = np.zeros(shape=(1, 160, 100))  # 创建了一个 dim = 1， size =160*100 的零数组
        roi_mask_mini[0] = image_array[0][270:430, 200:300]
        roi_mask_mini = normalize_data(roi_mask_mini)
        roi_mask[0][270:430, 200:300] = roi_mask_mini[0]
        image_tensor = torch.from_numpy(roi_mask).float()
        image_data.append((image_tensor, image_path[1], image_path[2]))

    return image_data, mask_data


class Dataset(data.Dataset):
    """
    数据集类，将数据从数据集路径中读取并保存在类中。
    TODO
    """

    def __init__(self, args, have=True):
        self.imgs = get_dataset(data_path=args.data_path, have=have)

    def __getitem__(self, index):
        image = self.imgs[0][index]
        mask = self.imgs[1][index]
        return image, mask

    def __len__(self):
        return len(self.imgs[0])
