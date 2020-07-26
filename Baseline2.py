import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet_CMs_low_rank(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm, rank):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(UNet_CMs_low_rank, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        #
        # if class_no > 2:
        #     #
        #     self.final_in = class_no
        # else:
        #     #
        #     self.final_in = 1
        #
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders_noisy_layers = nn.ModuleList()
        #

        for i in range(self.depth):

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)
        #
        for i in range(self.noisy_labels_no):
            #
            self.decoders_noisy_layers.append(low_rank_cm_layers(in_channels=width, norm=norm, class_no=self.final_in, rank=rank))

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        y_noisy = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
                #
                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            #
            y = self.decoders[-(i+1)](y)
        #
        for i in range(self.noisy_labels_no):
            #
            y_noisy_label = self.decoders_noisy_layers[i](y)
            y_noisy.append(y_noisy_label)
        #
        y = self.conv_last(y)
        #
        return y, y_noisy


class CMNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(CMNet, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        #
        # if class_no > 2:
        #     #
        #     self.final_in = class_no
        # else:
        #     #
        #     self.final_in = 1
        #
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders_noisy_layers = nn.ModuleList()
        #

        for i in range(self.depth):

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        for i in range(self.noisy_labels_no):
            #
            self.decoders_noisy_layers.append(cm_layers(in_channels=width, norm=norm, class_no=self.final_in))

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        y_noisy = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
                #
                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            #
            y = self.decoders[-(i+1)](y)
        #
        for i in range(self.noisy_labels_no):
            #
            y_noisy_label = self.decoders_noisy_layers[i](y)
            y_noisy.append(y_noisy_label)
            #
        return y_noisy


class AnnotatorNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm, low_rank=False, rank=0):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(AnnotatorNet, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        #
        # if class_no > 2:
        #     #
        #     self.final_in = class_no
        # else:
        #     #
        #     self.final_in = 1
        #
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders_noisy_layers = nn.ModuleList()
        #

        for i in range(self.depth):

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)
        #
        for i in range(self.noisy_labels_no):
            #
            if low_rank is False:
                #
                assert rank == 0
                #
                self.decoders_noisy_layers.append(cm_layers(in_channels=width, norm=norm, class_no=self.final_in))
            else:
                #
                assert rank <= class_no
                #
                self.decoders_noisy_layers.append(low_rank_cm_layers(in_channels=width, norm=norm, class_no=self.final_in, rank=rank))

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        y_noisy = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
                #
                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            #
            y = self.decoders[-(i+1)](y)
        #
        for i in range(self.noisy_labels_no):
            #
            y_noisy_label = self.decoders_noisy_layers[i](y)
            y_noisy.append(y_noisy_label)
        #
        # y = self.conv_last(y)
        #
        return y_noisy


class UNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(UNet, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        #
        # if class_no > 2:
        #     #
        #     self.final_in = class_no
        # else:
        #     #
        #     self.final_in = 1
        #
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders_noisy_layers = nn.ModuleList()
        #

        for i in range(self.depth):

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        y_noisy = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
                #
                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            #
            y = self.decoders[-(i+1)](y)
            #
        y = self.conv_last(y)
        #
        return y


class UNet_Implicit_CMs(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(UNet_Implicit_CMs, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        #
        # if class_no > 2:
        #     #
        #     self.final_in = class_no
        # else:
        #     #
        #     self.final_in = 1
        #
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders_noisy_layers = nn.ModuleList()
        #

        for i in range(self.depth):

            if i == 0:
                #
                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))
                #
            elif i < (self.depth - 1):
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
            else:
                #
                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
                #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)
        #
        for i in range(self.noisy_labels_no):
            #
            self.decoders_noisy_layers.append(cm_layers(in_channels=width, norm=norm, class_no=self.final_in))

    def forward(self, x):
        #
        y = x
        #
        encoder_features = []
        y_noisy = []
        #
        for i in range(len(self.encoders)):
            #
            y = self.encoders[i](y)
            encoder_features.append(y)
        # print(y.shape)
        for i in range(len(encoder_features)):
            #
            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]
            #
            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
                #
                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            #
            y = self.decoders[-(i+1)](y)
        #
        for i in range(self.noisy_labels_no):
            #
            y_noisy_label = self.decoders_noisy_layers[i](y)
            y_noisy.append(y_noisy_label)
        #
        y = self.conv_last(y)
        #
        return y, y_noisy


class noisy_output_layers(nn.Module):

    def __init__(self, in_channels, norm, class_no):
        super(noisy_output_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_last = nn.Conv2d(in_channels, class_no, 1, bias=True)

    def forward(self, x):
        #
        y = self.conv_last(self.conv_2(self.conv_1(x)))
        #
        return y


class cm_layers(nn.Module):

    def __init__(self, in_channels, norm, class_no):
        super(cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_last = nn.Conv2d(in_channels, class_no**2, 1, bias=True)
        self.relu = nn.Softplus()

    def forward(self, x):
        #
        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))
        #
        return y


class low_rank_cm_layers(nn.Module):
    #
    def __init__(self, in_channels, norm, class_no, rank):
        super(low_rank_cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        if rank == 1:
            self.conv_last = nn.Conv2d(in_channels, rank * class_no * 2 + 1, 1, bias=True)
        else:
            self.conv_last = nn.Conv2d(in_channels, rank*class_no*2 + 1, 1, bias=True)
        self.relu = nn.Softplus()

    def forward(self, x):
        #
        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))
        #
        return y


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, step, norm):
        super(conv_block, self).__init__()
        #
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=step, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_1 = nn.PReLU()
        self.activation_2 = nn.PReLU()
        #
        if norm == 'bn':
            self.smooth_1 = nn.BatchNorm2d(out_channels, affine=True)
            self.smooth_2 = nn.BatchNorm2d(out_channels, affine=True)
        elif norm == 'in':
            self.smooth_1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.smooth_2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'ln':
            self.smooth_1 = nn.GroupNorm(out_channels, out_channels, affine=True)
            self.smooth_2 = nn.GroupNorm(out_channels, out_channels, affine=True)
        elif norm == 'gn':
            self.smooth_1 = nn.GroupNorm(out_channels // 8, out_channels, affine=True)
            self.smooth_2 = nn.GroupNorm(out_channels // 8, out_channels, affine=True)

    def forward(self, inputs):
        output = self.activation_1(self.smooth_1(self.conv_1(inputs)))
        output = self.activation_2(self.smooth_2(self.conv_2(output)))
        return output


def double_conv(in_channels, out_channels, step, norm):
    #
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===================
    if norm == 'in':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'bn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'ln':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'gn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU()
        )