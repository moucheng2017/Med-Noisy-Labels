import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet_Direct_CMs(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm, b, h, w):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(UNet_Direct_CMs, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        #
        if class_no > 2:
            #
            self.final_in = class_no
        else:
            #
            self.final_in = 1
        #
        # self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.confusion_matrices = nn.ModuleList()
        #
        if self.dropout is True:

            self.dropout_layers = nn.ModuleList()

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
            self.confusion_matrices.append(confusion_matrix_layer(b=b, c=class_no, h=h, w=w))
            #
        # self.confusion_matrix_1 = torch.nn.Parameter(torch.cat(b * h * w * [torch.eye(class_no, class_no)]).reshape(b*h*w, class_no, class_no))
        # self.confusion_matrix_2 = torch.nn.Parameter(torch.cat(b * h * w * [torch.eye(class_no, class_no)]).reshape(b*h*w, class_no, class_no))
        # self.confusion_matrix_3 = torch.nn.Parameter(torch.cat(b * h * w * [torch.eye(class_no, class_no)]).reshape(b*h*w, class_no, class_no))
        # self.confusion_matrix_4 = torch.nn.Parameter(torch.cat(b * h * w * [torch.eye(class_no, class_no)]).reshape(b*h*w, class_no, class_no))

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
        for i in range(self.noisy_labels_no):
            #
            y_noisy_label = self.confusion_matrices[i](y)
            y_noisy.append(y_noisy_label)
        #
        return y, y_noisy
        # return y, \
        #       F.relu(self.confusion_matrix_1, inplace=False), \
        #       F.relu(self.confusion_matrix_2, inplace=False), \
        #       F.relu(self.confusion_matrix_3, inplace=False), \
        #       F.relu(self.confusion_matrix_4, inplace=False)


class confusion_matrix_layer(nn.Module):

    def __init__(self, b, c, h, w):
        super(confusion_matrix_layer, self).__init__()
        self.batch = b
        self.class_no = c
        self.height = h
        self.width = w
        self.confusion_layer = torch.nn.Parameter(torch.cat(b * h * w * [torch.eye(c, c)]).reshape(b*h*w, c, c))

    def forward(self, x):
        cm = F.relu(self.confusion_layer, inplace=False)
        cm = cm.reshape(self.batch*self.height*self.width, self.class_no, self.class_no)
        #
        bb, cc, hh, ww = x.size()
        #
        if cc == 1:
            x_positive = torch.sigmoid(x)
            x_negative = torch.ones_like(x_positive) - x_positive
            x = torch.cat([x_positive, x_negative], dim=1).reshape(self.batch*self.height*self.width, self.class_no, 1)
        else:
            x = nn.softmax(dim=1)(x).reshape(self.batch*self.height*self.width, self.class_no, 1)
        #
        cm = cm / cm.sum(1, keepdim=True)
        #
        y = torch.bmm(cm, x).reshape(self.batch, self.class_no, self.height, self.width)
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