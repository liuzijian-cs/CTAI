import os
import time
import numpy as np
import torch
import argparse
from lib.base_function import print_log,draw_picture
from lib.data_function import get_dataloader
from lib.dice_loss import dice
from model.unet import UNet

parser = argparse.ArgumentParser()  # 声明参数解释器
parser.add_argument('--device', type=str, default='cuda:0', help='')  # 设备：默认GPU
parser.add_argument('--data_path', type=str, default='D:\CTAI_source\CTAI_model\ct_data',
                    help='dataset path')  # 指定使用的数据集路径
parser.add_argument('--save', type=str, default='model_save', help='save path')  # 模型保存路径
parser.add_argument('--log_file', type=str, default='model_save/log.txt', help='log file')  # 日志文件
parser.add_argument('--test_ratio', type=float, default=0.2)  # 测试集比例(本项目未设置开发集)
parser.add_argument('--seed', type=int, default=3407, help='random seed')  # 随机种子，如果设置为None 则完全随机
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # 学习率
parser.add_argument('--epochs', type=int, default=2)  # 训练轮数
parser.add_argument('--threshold', type=float, default=0.5)  # 置信度
parser.add_argument('--have', type=bool, default=True)  # 是否检查 image 与 mask的一一对应
parser.add_argument('--batch_size', type=int, default=4)  # batch_size 大小,根据显存合理设置，尽量不要发生内存交换
parser.add_argument('--shuffle', type=bool, default=True)  # shuffle
parser.add_argument('--num_workers', type=int, default=16)  # 多线程 dataloader：！请根据CPU线程数配置

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
    train_loader, test_loader = get_dataloader(args)  # 返回torch.utils.data.Dataloader 对象
    print_log(args, f"Strat Training: Train loader size = {len(train_loader)}, Test loader size = {len(test_loader)}")

    # others:
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    train_loss = []
    train_dice = []
    test_loss = []
    test_dice = []

    print_log(args, str(args))

    for epoch in range(args.epochs):
        epoch_train_loss = 0
        epoch_train_dice = 0
        epoch_test_loss = 0
        epoch_test_dice = 0

        t1_epoch = time.time()

        # Train:
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x[0].to(device), y[1].to(device)
            y = y.unsqueeze(1)

            optimizer.zero_grad()
            y_pred = model(x)

            # train loss:
            loss = criterion(y_pred, y)
            epoch_train_loss += loss.item()
            loss.backward()

            # train dice:
            y_pred_np = (y_pred.detach().cpu().numpy() > 0.5).astype(np.bool_)
            batch_dice = dice(y.detach().cpu().numpy().astype(np.bool_), y_pred_np)
            epoch_train_dice += batch_dice

            optimizer.step()

        t2_epoch = time.time()

        # Test:
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x[0].to(device), y[1].to(device)
                y = y.unsqueeze(1)
                y_pred = model(x)

                # test loss
                loss = criterion(y_pred, y)
                epoch_test_loss += loss.item()

                y_pred_np = (y_pred.detach().cpu().numpy() > 0.5).astype(np.bool_)
                batch_dice = dice(y.detach().cpu().numpy().astype(np.bool_), y_pred_np)
                epoch_test_dice += batch_dice


                # y_pred = model(x)
                # loss = criterion(y,y_pred)
                #
                # y_np = y[1].detach().cpu().squeeze(0).numpy()
                # y_pred_np = torch.squeeze(y_pred).detach().cpu().numpy()
                # y_pred_np[y_pred_np >= args.threshold] = 1
                # y_pred_np[y_pred_np < args.threshold] = 0
                #
                # epoch_test_loss += loss
                # epoch_test_dice += dice(y_np, y_pred_np)

        t3_epoch = time.time()

        train_loss_ = epoch_train_loss / len(train_loader)
        test_loss_ = epoch_test_loss / len(test_loader)
        train_dice_ = epoch_train_dice / len(train_loader)
        test_dice_ = epoch_test_dice/len(test_loader)

        train_loss.append(train_loss_)
        test_loss.append(test_loss_)
        train_dice.append(train_dice_)
        test_dice.append(test_dice_)

        print_log(args, f"epoch: {epoch:03d}, train loss:{train_loss_:.5f}, test loss: {test_loss_:.5f}, train dice: {train_dice_:.5f}, test dice: {test_dice_:.5f}, epoch time cost: {(t3_epoch-t1_epoch):.2f} (train: {(t2_epoch - t1_epoch):.2f}, test: {(t3_epoch-t2_epoch):.2f})")


        torch.save(model.state_dict(),os.path.join(args.save, f"{epoch:03d}.pth")) # 模型保存
    # draw
    draw_picture(args, train_loss, test_loss, train_dice, test_dice)
    log.close()

if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"Total time cost: {(time_end - time_start):.2f}")
