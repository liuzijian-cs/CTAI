import time

import numpy as np
import torch
import torch.nn as nn
import argparse
from lib.base_function import print_log
from lib.data_function import get_dataloader

parser = argparse.ArgumentParser()  # 声明参数解释器
parser.add_argument('--device', type=str, default='cuda:0', help='')  # 设备：默认GPU
parser.add_argument('--data_path', type=str, default='E:\ct_data', help='dataset path')  # 指定使用的数据集路径
parser.add_argument('--save', type=str, default='model_save', help='save path')  # 模型保存路径
parser.add_argument('--log_file', type=str, default='model_save/log.txt', help='log file')  # 日志文件
parser.add_argument('--test_ratio', type=float, default=0.2)  # 测试集比例(本项目未设置开发集)
parser.add_argument('--seed', type=int, default=3407, help='random seed')  # 随机种子，如果设置为None 则完全随机
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # 学习率
parser.add_argument('--epochs', type=int, default=200)  # 训练轮数
parser.add_argument('--rate', type=int, default=0.5)  # 应该是置信度 ？
parser.add_argument('--have', type=bool, default=True)  # 是否检查 image 与 mask的一一对应
parser.add_argument('--print_every', type=int, default=50, help='')  # 每训练多少次迭代输出一次日志信息
parser.add_argument('--batch_size', type=int, default=16)  # batch_size 大小
parser.add_argument('--shuffle', type=bool, default=True)  # shuffle
parser.add_argument('--num_workers', type=int, default=8)  # 多线程 dataloader：！请根据CPU线程数配置

args = parser.parse_args()


def main():
    # Log:
    log = open(args.log_file, 'w')  # 清空 & 创建日志文件

    # Device:
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.backends.cudnn.enabled = True  # 启用 CUDA 库的支持
    print_log(args, "train - Device: {}".format(device))

    # Seed:
    if args.seed is not None:
        if args.device == 'cuda':
            torch.cuda.init()
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True  # 将CUDA的随机数生成器设置为确定性模式
            torch.backends.cudnn.benchmark = False  # 禁用 CUDA 库的基准模式。当启用基准模式时，CUDA 库会根据硬件和输入数据的特性自动选择最适合的卷积算法，以提高性能。然而，自动选择算法可能会导致运行时性能的不稳定性，因此在要求结果可复现性时，禁用基准模式是必要的。
        np.random.seed(args.seed)
    print_log(args, "train - Seed: {}".format(args.seed))

    # Data:
    # 将数据全部与加载到内存（不合理，但暂时不会改，因此保留原作者风格）
    train_loader, test_loader = get_dataloader(args)




if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"Total time cost: {(time_end - time_start):.4f}")
