import torch
import torch.nn as nn
from torch.nn import MaxPool2d


class DoubleConv(nn.Module):
    """
    Double Convolution 双重卷积
    第一个卷积层接收输入数据并提取初步的特征；
    二个卷积层进一步处理这些特征。这两个卷积层通常具有相同的滤波器（卷积核）数量和大小。
    这种设计可以加深网络的深度，增强模型的学习能力，而不会立即通过池化（如最大池化）减少空间维度，从而保留更多的特征信息

    # Note: BatchNorm2D
    针对每个特征进行归一化处理，即对小批次中的数据计算均值和方差，并用这些统计量进行归一化。
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-net 模型实现
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.down_convs = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])
        self.last_down_conv = DoubleConv(512, 1024)

        self.pool = MaxPool2d(kernel_size=2)

        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
        ])

        self.up_double_convs = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64)
        ])

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        convs = []  # 初始化一个列表，用于存储每次卷积后的特征图。这些特征图会在解码器部分用于特征融合。
        for conv in self.down_convs:  # 下采样, 遍历down_convs的四个DoubleConv模型
            x = conv(x)
            convs.append(x)
            x = self.pool(x)
        x = self.last_down_conv(x)

        for i, (up_conv, double_conv) in enumerate(zip(self.up_convs, self.up_double_convs)):
            x = up_conv(x)
            x = torch.cat([x, convs[-(i+1)]], dim=1)  # dim = 1 [batch_size, channels, height, width] 相当于沿着channels拼接
            x = double_conv(x)

        return nn.Sigmoid()(self.final_conv(x))

# 测试函数
if __name__ == '__main__':
    # 初始化网络
    # 假设我们的网络输入通道为1（如灰度图），输出通道为1（如二分类问题）
    net = UNet(in_channels=1, out_channels=1)
    # 这里的张量大小为 [batch_size, channels, height, width]
    test_tensor = torch.randn(16, 1, 512, 512)
    # 为了避免任何在训练过程中可能发生的变化（如Dropout），我们将网络设置为评估模式
    net.eval()
    with torch.no_grad():  # 确保不会计算梯度
        output = net(test_tensor)
    # 打印输出张量的大小，确认网络输出的尺寸
    print("Output tensor shape:", output.shape)
