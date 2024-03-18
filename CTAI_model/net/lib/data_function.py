from torch.utils import data

import os
import cv2 as cv
import SimpleITK as sitk


# Note:SimpleITK 是一个用于医学图像处理的开源库，提供了简单易用的程序接口来访问 ITK
# ITK (Insight Segmentation and Registration Toolkit) 的功能。ITK 是一个广泛用于医学图像处理、分割和注册的高级开源库。


def get_train_file_list(data_path, get_dice=False):
    """
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


def get_dataset(data_path, have):
    global test_image, test_mask
    image_list, mask_list, image_data, mask_data = [], [], [], []  # 初始化局部变量

    image_list = get_train_file_list(data_path)  # return image_list - (file_path, patient_id, image_id)
    for image_path in image_list:
        # image data:
        image_data.append(sitk.ReadImage(image_path[0]))
        image_array = sitk.GetArrayFromImage(image_data)
        # mask data
        mask_path = image_path[0].replace('.dcm', '_mask.png')
        mask_array = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)



    for i in image_list:
        image = sitk.ReadImage(i[0])
        image_array = sitk.GetArrayFromImage(image)
        mask = i[0].replace('.dcm', '_mask.png')
        mask_array = cv.imread(mask, cv.IMREAD_GRAYSCALE)

        if have:
            if not mask_array.any():
                continue

        mask_array = data_in_one(mask_array)
        mask_tensor = torch.from_numpy(mask_array).float()
        j = i[0].split('/')[-1].replace('_mask.png', '')
        mask_data.append((j, mask_tensor))

        ROI_mask = np.zeros(shape=image_array.shape)
        ROI_mask_mini = np.zeros(shape=(1, 160, 100))
        ROI_mask_mini[0] = image_array[0][270:430, 200:300]
        ROI_mask_mini = data_in_one(ROI_mask_mini)
        ROI_mask[0][270:430, 200:300] = ROI_mask_mini[0]
        test_image = ROI_mask
        image_tensor = torch.from_numpy(ROI_mask).float()
        image_data.append((image_tensor, i[1], i[2]))

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
