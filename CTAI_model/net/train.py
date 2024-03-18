import time

import torch
import torch.nn as nn
import argparse
from lib.base_function import print_log


parser = argparse.ArgumentParser() # 声明参数解释器
parser.add_argument('--device', type=str, default='cuda:0', help='') # 设备：默认GPU
parser.add_argument('--data', type=str, default='data/ct_data', help='dataset path') # 指定使用的数据集路径
parser.add_argument('--save', type=str, default='model_save', help='save path') # 模型保存路径
parser.add_argument('--log_file', type=str, default='model_save/log.txt', help='log file') # 日志文件
parser.add_argument('--test_ratio', type=float, default=0.2) # 测试集比例
parser.add_argument('--val_ratio', type=float, default=0) # 开发集比例
parser.add_argument('--seed', type=int, default=3407, help='random seed') # 随机种子，如果设置为-1 则完全随机
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate') # 学习率
parser.add_argument('--epochs', type=int, default=200) # 训练轮数
parser.add_argument('--rate', type=int, default=0.5) # 应该是置信度 ？
parser.add_argument('--print_every', type=int, default=50, help='') # 每训练多少次迭代输出一次日志信息

args = parser.parse_args()


def main():
    log = open(args.log_file, 'w') # 清空 & 创建日志文件

if __name__ == "__main__":
    time_start  = time.time()
    main()
    time_end = time.time()
    print(f"Total time cost: {(time_end-time_start):.4f}")



