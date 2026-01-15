import torch.nn as nn
import torch
import torch.nn.functional as F

class VSRRDN(nn.Module):
    def __init__(
            self,
            upscale_factor=6,
            in_channels=1,
            out_channels=1,
            channels=64,
            num_rdb=16,
            num_rb=8,
            growth_channels=64,
    ):
        super(VSRRDN, self).__init__()
        self.num_rdb = num_rdb

        # Conv Block
        self.conv1 = nn.Conv3d(in_channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1)

        # Residual Dense Blocks
        trunk = []
        for _ in range(num_rdb):
            trunk.append(_ResidualDenseBlock(channels, growth_channels, num_rb))
        self.trunk = nn.Sequential(*trunk)

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(
            nn.Conv3d(int(num_rdb * channels), channels, 1, 1, 0),
            nn.Conv3d(channels, channels, 3, 1, 1),
        )

        self.global_attn = globalAttention()

        # Upscale block
        self.upsampling = _UpsampleBlock(channels, upscale_factor)

        # Output layer
        self.conv3 = nn.Conv3d(channels, out_channels, 3, 1, 1)
    def forward(self, x):
        # 输入x [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        out1 = self.conv1(x) # [B, C, T, H, W]
        out = self.conv2(out1) # [B, C, T, H, W]

        outs = []
        for i in range(self.num_rdb):
            out = self.trunk[i](out)
            outs.append(out)
        out = torch.cat(outs, 1)
        out = self.global_feature_fusion(out)
        out = out.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        out = self.global_attn(out)
        out = out.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        out = torch.add(out1, out)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = torch.mean(out, dim=2) # [batch, C, H, W]
        out = torch.clamp_(out, 0.0, 1.0)
        return out

class _ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int, layers: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        rdb = []
        for index in range(layers):
            rdb.append(_ResidualBlock(channels + index * growth_channels, growth_channels))
        self.rdb = nn.Sequential(*rdb)

        # Local Feature Fusion layer
        self.local_feature_fusion = nn.Conv3d(channels + layers * growth_channels, channels, 1, 1, 0)

    def forward(self, x):
        identity = x

        out = self.rdb(x)
        out = self.local_feature_fusion(out)
        out = torch.add(out, identity)

        return out

class _ResidualBlock(nn.Module):
    def __init__(self, channels, growth_channels):
        super(_ResidualBlock, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv3d(channels, growth_channels, 3, 1, 1),
            nn.ReLU(True),
        )

    def forward(self, x):
        identity = x

        out = self.rb(x)
        out = torch.cat([identity, out], 1)

        return out

class _UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor):
        super(_UpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.PixelShuffle3D_2 = PixelShuffle3D(2)
        self.PixelShuffle3D_3 = PixelShuffle3D(3)

        base_layers = [
            nn.Conv3d(channels, channels * 4, 3, 1, 1),
            nn.LeakyReLU(),
            self.PixelShuffle3D_2,
        ]
        if self.scale_factor == 4:
            num_blocks = 2
        elif self.scale_factor == 2:
            num_blocks = 1
        elif self.scale_factor == 6:
            # 2倍 + 3倍
            num_blocks = 2
        elif self.scale_factor == 12:
            #2倍x2 + 3倍
            num_blocks = 3
        elif self.scale_factor == 16:
            num_blocks = 4
        else:  # scale_factor == 8
            num_blocks = 3
        upsample_layers = []
        for i in range(num_blocks):
            if (scale_factor == 6 or scale_factor == 12) and i == num_blocks - 1:
                upsample_layers += [
                    nn.Conv3d(channels, channels * 9, 3, 1, 1),
                    nn.LeakyReLU(),
                    self.PixelShuffle3D_3,
                ]
            else:
                upsample_layers += base_layers.copy()
        self.upsampling = nn.Sequential(*upsample_layers)
    def forward(self, x):
        return self.upsampling(x)

class PixelShuffle3D(nn.Module):
    def __init__(self, spatial_scale):
        super().__init__()
        self.spatial_scale = spatial_scale

    def forward(self, x):
        B, C, T, H, W = x.shape
        scale = self.spatial_scale
        C_out = C // (scale * scale)
        # Shuffle
        x = x.view(B, C_out, scale, scale, T, H, W)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()  # [B, C_out, T, H, scale, W, scale]
        x = x.view(B, C_out, T, H * scale, W * scale)
        return x

class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=2, heads=1):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * num_feat
        self.hidden_dim = self.dim // heads
        self.patch_size = patch_size

        self.to_q = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_k = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_v = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)


    def forward(self, x):
        b, t, c, h, w = x.shape  # B, T, 64, h, w
        H, D = self.heads, self.dim
        d = self.hidden_dim
        n = (h // self.patch_size) * (w // self.patch_size)

        patch2feat = torch.nn.Fold(output_size=(h, w), kernel_size=self.patch_size, padding=0, stride=self.patch_size)
        q = self.to_q(x.view(-1, c, h, w))  # [B*T, 64, h, w]
        k = self.to_k(x.view(-1, c, h, w))  # [B*T, 64, h, w]
        v = self.to_v(x.view(-1, c, h, w))  # [B*T, 64, h, w]]

        unfold_q = self.feat2patch(q)  # [B*T, 2*2*64, n]
        unfold_k = self.feat2patch(k)  # [B*T, 2*2*64, n]
        unfold_v = self.feat2patch(v)  # [B*T, 2*2*64, n]

        unfold_q = unfold_q.view(b, t, H, d, n)  # [B, T, H, 2*2*64/H, n]
        unfold_k = unfold_k.view(b, t, H, d, n)  # [B, T, H, 2*2*64/H, n]
        unfold_v = unfold_v.view(b, t, H, d, n)  # [B, T, H, 2*2*64/H, n]

        unfold_q = unfold_q.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 2*2*64/H, T, n]
        unfold_k = unfold_k.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 2*2*64/H, T, n]
        unfold_v = unfold_v.permute(0, 2, 3, 1, 4).contiguous()  # [B, H, 2*2*64/H, T, n]

        unfold_q = unfold_q.view(b, H, d, t * n)  # [B, H, 2*2*64/H, T*n]
        unfold_k = unfold_k.view(b, H, d, t * n)  # [B, H, 2*2*64/H,T*n]
        unfold_v = unfold_v.view(b, H, d, t * n)  # [B, H, 2*2*64/H, T*n]

        attn = torch.matmul(unfold_q.transpose(2, 3), unfold_k)  # [B, H, T*n, T*n]
        attn = attn * (d ** (-0.5))  # [B, H, T*n, T*n]
        attn = F.softmax(attn, dim=-1)  # [B, H, T*n, T*n]

        attn_x = torch.matmul(attn, unfold_v.transpose(2, 3))  # [B, H, T*n, 2*2*64/H]
        attn_x = attn_x.view(b, H, t, n, d)  # [B, H, T, n, 8*8*64/H]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()  # [B, T, H, 2*2*64/H, n]
        attn_x = attn_x.view(b * t, D, n)  # [B*5, 2*2*64, n]
        feat = patch2feat(attn_x)  # [B*T, 64, h, w]

        out = self.conv(feat).view(x.shape)  # [B, T, 64, h, w]
        out += x  #  [B, T, 64, h, w]

        return out

class Discriminator(nn.Module):
    def __init__(self, __patch_size__):
        super(Discriminator, self).__init__()
        # features
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.patch_size = __patch_size__ // 8
        # 1x192x192
        self.conv1 = nn.Conv2d(1, 64, 3, 2, 1)
        # 64x96x96
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        # 128x48x48
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        # 256x24x24
        # classifier
        self.linear0 = nn.Linear(256 * self.patch_size * self.patch_size, 100)
        self.linear1 = nn.Linear(100, 1)

    def forward(self, x):
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.lrelu(self.bn2(self.conv2(out)))
        out = self.lrelu(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.lrelu(self.linear0(out))
        out = self.linear1(out)
        return out

if __name__ == '__main__':
    sequence = torch.randn(2,3,1,78,252)
    net=VSRRDN(upscale_factor=4)
    a = net(sequence)
    print(a.size())