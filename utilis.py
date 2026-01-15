import torch
import torch.nn as nn
from torchvision import models
from pytorch_msssim import MS_SSIM
import torch.fft as fft
from torch.nn import functional as F
import numpy as np
import cv2
import math
from typing import Tuple, Union, Optional

class SSIMLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.5):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, win_size=5, size_average=True, channel=1)

    def forward(self, pred, target):
        # MS-SSIM 损失（值越大表示越相似，需转为损失）
        ms_ssim_loss = 1 - self.ms_ssim(pred, target)
        return ms_ssim_loss

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps).mean()
        return loss


#VGG Loss vascular-7
class VGGFeatureExtractor(nn.Module):
    def __init__(self,layer_index=7):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:layer_index]
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.vgg = vgg.eval()

    def forward(self, x):
        # 灰度图转伪RGB（假设输入为[B,1,H,W]）
        x = torch.cat([x] * 3, dim=1)  # [B,3,H,W]

        # 归一化输入至VGG输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_norm = (x - mean) / std
        # 提取特征
        features = self.vgg(x_norm)
        return features




class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL) 实现
    参考论文: https://arxiv.org/abs/2012.12821v3
    """

    def __init__(self, alpha=1.0):
        """
        参数:
            alpha: 缩放因子，控制权重矩阵的聚焦程度 (原文默认1.0)
            norm: 是否对FFT结果进行归一化（确保梯度稳定）
        """
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        """
        计算Focal Frequency Loss

        参数:
            pred: 生成图像 (shape: [B, C, H, W])
            target: 真实图像 (shape: [B, C, H, W])

        返回:
            loss: 标量损失值
        """
        B, C, H, W = pred.shape

        # 1. 对预测和真实图像进行2D FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))  # [B, C, H, W] (复数张量)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))  # [B, C, H, W]

        # 2. 对FFT结果进行归一化
        pred_fft = pred_fft / (H * W) ** 0.5  # 归一化以确保梯度稳定
        target_fft = target_fft / (H * W) ** 0.5

        # 3. 计算频域距离：|F_pred - F_target|^2
        # 复数差的模平方 = (实部差)^2 + (虚部差)^2
        diff = pred_fft - target_fft
        freq_distance = torch.abs(diff) ** 2  # [B, C, H, W]

        # 4. 计算动态频谱权重矩阵 w(u,v) = |F_pred - F_target|^alpha
        weight = freq_distance ** self.alpha  # [B, C, H, W]
        # 归一化权重到[0,1]（按每个样本的最大值）
        weight_flat = weight.view(B, C, -1)  # [B, C, H*W]
        max_weight = torch.max(weight_flat, dim=2, keepdim=True)[0]  # [B, C, 1]
        max_weight = max_weight.view(B, C, 1, 1)  # 恢复空间维度 [B, C, 1, 1]
        weight = weight / (max_weight + 1e-8) # [B, C, H, W]
        # 5. 计算加权平均：FFL = (1/(B*C*H*W)) * sum(w * freq_distance)
        loss = torch.sum(weight * freq_distance) / (B * C * H * W)

        return loss


class FourierMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.MSELoss()

    def forward(self, pred, target):
        """
        计算傅里叶域的MAE损失
        参数:
            pred: 网络输出的重建图像，形状 [batch, channels, H, W]
            target: 真实图像，形状 [batch, channels, H, W]
        返回:
            fmae_loss: 傅里叶幅度谱的MAE损失
        """

        # 对每个通道计算二维傅里叶变换
        # 傅里叶变换结果为复数，包含实部和虚部
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # 计算幅度谱（复数的模）
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # # 可选：归一化幅度谱（避免因图像亮度差异导致的数值偏差）
        # if self.normalize:
        #     pred_mag = pred_mag / (pred_mag.max() + 1e-8)  # 防止除零
        #     target_mag = target_mag / (target_mag.max() + 1e-8)

        # 计算幅度谱的MAE
        return self.mae(pred_mag, target_mag)


class PSNRCalculator(nn.Module):
    def __init__(self,data_range=1.0):
        super(PSNRCalculator, self).__init__()
        self.data_range = data_range

    def forward(self, pred, target):

        # 计算MSE（每个样本单独计算）
        mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
        # 避免除零错误
        mse = torch.clamp(mse, min=1e-10)
        # 计算PSNR
        psnr = 10.0 * torch.log10((self.data_range ** 2) / mse)  # [B]

        return psnr.mean()

if __name__ == "__main__":
    X = torch.rand(1, 1, 96, 96).to('cuda')  # 示例数据
    Y = torch.rand(1, 1, 96, 96).to('cuda') # 示例数据
    loss_criterion = FocalFrequencyLoss().to('cuda')
    loss = loss_criterion(X, Y)
    print(loss)