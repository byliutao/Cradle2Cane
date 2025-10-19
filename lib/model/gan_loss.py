import torch
import torch.nn as nn
import torch.nn.functional as F

# 判别器定义，输入为 3 通道 512×512 的图像
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 第一层: 输出尺寸 256x256
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 第二层: 输出尺寸 128x128
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 第三层: 输出尺寸 64x64
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 第四层: 输出尺寸 32x32
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 最后一层: 输出尺寸 31x31 或更小（根据 padding 调整），输出单通道特征图
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0)
        )
        
    def forward(self, x):
        return self.model(x)

# Hinge Loss 定义：
# 对于判别器：
#   对真实图像：loss_real = mean( relu(1 - D(real)) )
#   对伪造图像：loss_fake = mean( relu(1 + D(fake)) )
# 总判别器 loss 为两部分之和
def discriminator_hinge_loss(D, real_images, fake_images, weight=1.0):
    # D 输出无需激活函数，直接计算 hinge loss
    real_preds = D(real_images)
    fake_preds = D(fake_images)
    loss_real = torch.mean(F.relu(1.0 - real_preds))
    loss_fake = torch.mean(F.relu(1.0 + fake_preds))
    d_loss = loss_real + loss_fake
    return d_loss*weight

# 对于生成器：
#   生成器 loss 定义为 -mean(D(fake))
def generator_hinge_loss(all_loss, D, fake_images, weight=1.0):
    fake_preds = D(fake_images)
    g_loss = -torch.mean(fake_preds)
    g_loss *= weight

    if weight > 0:
        all_loss["g_loss"] = g_loss
    return g_loss

# 示例：如何在训练循环中使用
if __name__ == '__main__':
    # 创建判别器实例
    D = Discriminator(in_channels=3).cuda()
    optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # 示例输入：真实图像和生成器生成的假图像
    # 假设真实图像和假图像均为 [B, 3, 512, 512] 的张量
    real_images = torch.randn(4, 3, 512, 512).cuda()
    fake_images = torch.randn(4, 3, 512, 512).cuda()  # 实际中用生成器生成假图像

    # 计算判别器 hinge loss
    d_loss = discriminator_hinge_loss(D, real_images, fake_images)
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()
    
    # 如果需要计算生成器的 loss（生成器网络 G）
    # g_loss = generator_hinge_loss(D, fake_images)