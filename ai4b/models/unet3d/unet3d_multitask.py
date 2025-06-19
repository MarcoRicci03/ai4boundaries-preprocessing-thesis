import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
#             Core Blocks              #
########################################

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diffH = skip.size(-2) - x.size(-2)
        diffW = skip.size(-1) - x.size(-1)
        x = F.pad(x, [0, diffW, 0, diffH])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

########################################
#             2D Modules               #
########################################

def get_norm2d(name, channels, num_groups):
    if name == 'BatchNorm':
        return nn.BatchNorm2d(channels)
    elif name == 'GroupNorm':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        raise ValueError(f"Unsupported norm type: {name}")

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=(1, 1),
                 padding=(0, 0), dilation=(1, 1), norm_type='BatchNorm',
                 num_groups=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=strides, padding=padding, dilation=dilation,
                              bias=False, groups=groups)
        self.norm = get_norm2d(norm_type, out_channels, num_groups)

    def forward(self, x):
        return self.norm(self.conv(x))

class HeadSingleTask(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes,
                 depth=2, norm_type='BatchNorm', norm_groups=None):
        super().__init__()
        layers = [ConvBlock2D(in_channels, out_channels, kernel_size=(3, 3),
                              padding=(1, 1), norm_type=norm_type, num_groups=norm_groups)]
        for _ in range(depth - 1):
            layers.append(ConvBlock2D(out_channels, out_channels, kernel_size=(3, 3),
                                      padding=(1, 1), norm_type=norm_type, num_groups=norm_groups))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, n_classes, kernel_size=1, padding=0))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

class CrispSigmoid(nn.Module):
    def __init__(self, smooth=1e-2):
        super().__init__()
        self.smooth = smooth
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        scaling = torch.reciprocal(self.smooth + torch.sigmoid(self.gamma))
        return torch.sigmoid(x * scaling)

class UNet2DMultitaskHead(nn.Module):
    def __init__(self, in_channels, n_classes, embed_channels=32,
                 spatial_size=256, norm_type='BatchNorm', norm_groups=None,
                 segm_act='softmax'):
        super().__init__()

        self.dist_head = HeadSingleTask(embed_channels, embed_channels, n_classes, norm_type=norm_type, norm_groups=norm_groups)
        self.dist_eq = ConvBlock2D(n_classes, embed_channels, kernel_size=1, norm_type=norm_type, num_groups=norm_groups)

        self.bound_comb = ConvBlock2D(embed_channels * 2, embed_channels, kernel_size=1, norm_type=norm_type, num_groups=norm_groups)
        self.bound_head = HeadSingleTask(embed_channels * 2, embed_channels, n_classes, norm_type=norm_type, norm_groups=norm_groups)
        self.bound_eq = ConvBlock2D(n_classes, embed_channels, kernel_size=1, norm_type=norm_type, num_groups=norm_groups)

        self.final_head = HeadSingleTask(embed_channels * 2, embed_channels, n_classes, norm_type=norm_type, norm_groups=norm_groups)

        if n_classes == 1:
            self.channel_act = CrispSigmoid()
            self.segm_act = CrispSigmoid()
        else:
            self.channel_act = nn.Softmax(dim=1)
            self.segm_act = nn.Softmax(dim=1) if segm_act == 'softmax' else CrispSigmoid()

        self.bound_act = CrispSigmoid()

    def forward(self, x):
        dist_logits = self.dist_head(x)
        dist = self.channel_act(dist_logits)
        dist_eq = F.relu(self.dist_eq(dist))

        bound_input = torch.cat([x, dist_eq], dim=1)
        bound_logits = self.bound_head(bound_input)
        bound = self.bound_act(bound_logits)
        bound_eq = F.relu(self.bound_eq(bound))

        comb_features = torch.cat([bound_eq, dist_eq], dim=1)
        comb_features = F.relu(self.bound_comb(comb_features))

        segm_input = torch.cat([comb_features, x], dim=1)
        segm_logits = self.final_head(segm_input)
        segm = self.segm_act(segm_logits)

        return torch.cat([segm, bound, dist], dim=1)

########################################
#         Full Model Assembly          #
########################################

class UNet3DMultitask(nn.Module):
    def __init__(self, in_channels=5, n_classes=3, base_filters=32, spatial_size=512,
                 segm_act='softmax', norm_type='BatchNorm', norm_groups=None, time_steps=6):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.enc2 = ConvBlock3D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.enc3 = ConvBlock3D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.enc4 = ConvBlock3D(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool3d((1, 2, 2))

        self.enc5 = ConvBlock3D(base_filters * 8, base_filters * 16)
        self.pool5 = nn.MaxPool3d((1, 2, 2))

        self.bottleneck = ConvBlock3D(base_filters * 16, base_filters * 32)

        self.up5 = UpBlock3D(base_filters * 32, base_filters * 16, base_filters * 16)
        self.up4 = UpBlock3D(base_filters * 16, base_filters * 8, base_filters * 8)
        self.up3 = UpBlock3D(base_filters * 8, base_filters * 4, base_filters * 4)
        self.up2 = UpBlock3D(base_filters * 4, base_filters * 2, base_filters * 2)
        self.up1 = UpBlock3D(base_filters * 2, base_filters, base_filters)

        self.temporal_proj = nn.Conv3d(base_filters, base_filters, kernel_size=(time_steps, 1, 1))

        self.head = UNet2DMultitaskHead(
            in_channels=base_filters,
            n_classes=n_classes,
            embed_channels=base_filters,
            spatial_size=spatial_size,
            norm_type=norm_type,
            norm_groups=norm_groups,
            segm_act=segm_act
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        b = self.bottleneck(self.pool5(e5))

        d5 = self.up5(b, e5)
        d4 = self.up4(d5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        d1_proj = self.temporal_proj(d1)  # [B, C, 1, H, W]
        d1_2d = d1_proj.squeeze(2)  # [B, C, H, W]

        return self.head(d1_2d)

########################################
#              Main Test               #
########################################

if __name__ == "__main__":
    model = UNet3DMultitask(in_channels=5, n_classes=1, base_filters=32)
    dummy_input = torch.randn(2, 6, 5, 512, 512)  # [B, T, C, H, W]
    output = model(dummy_input)
    print("Output shape:", output.shape)