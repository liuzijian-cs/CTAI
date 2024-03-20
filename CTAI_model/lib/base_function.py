import os
import numpy as np
from matplotlib import pyplot as plt


def print_log(args, string):
    """
    Printing log information
    打印日志信息
    """
    log = open(args.log_file, 'a')
    log.write(string + '\n')
    log.flush()
    print(string)



# 绘图并输出
def draw_picture(args,train_loss,train_dice,test_loss,test_dice):
    plt.figure(figsize=(16,12))
    epochs = range(args.epochs)
    plt.plot(epochs,train_loss,label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save, "train_loss.png"))

    plt.figure(figsize=(16,12))
    epochs = range(args.epochs)
    plt.plot(epochs,train_dice,label='Train Dice')
    plt.title('Train Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.savefig(os.path.join(args.save, "train_dice.png"))

    plt.figure(figsize=(16,12))
    epochs = range(args.epochs)
    plt.plot(epochs,test_loss,label='Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save, "test_loss.png"))

    plt.figure(figsize=(16,12))
    epochs = range(args.epochs)
    plt.plot(epochs,test_dice,label='Test Dice')
    plt.title('Test Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.savefig(os.path.join(args.save, "test_dice.png"))

    plt.figure(figsize=(16,12))
    epochs = range(args.epochs)
    plt.plot(epochs,train_loss,"b",label='Train Loss')
    plt.plot(epochs,test_loss,"r",label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save, "loss.png"))

    plt.figure(figsize=(16,12))
    epochs = range(args.epochs)
    plt.plot(epochs,train_dice,"b",label='Train Dice')
    plt.plot(epochs,test_dice,"r",label='Test Dice')
    plt.title('Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.savefig(os.path.join(args.save, "dice.png"))


# 弃用
# def dice_coefficient(y_true, y_pred):
#     smooth = 1e-6 # 避免除以零
#     y_true_flatten = np.asarray(y_true).flatten()
#     y_pred_flatten = np.asarray(y_pred).flatten()
#     intersection = np.sum(y_true_flatten * y_pred_flatten)
#     coefficient = (2. * intersection + smooth) / (np.sum(y_true_flatten)+np.sum(y_pred_flatten)+smooth)
#     return coefficient
