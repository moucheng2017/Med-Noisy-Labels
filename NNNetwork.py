import torch
import torch.nn as nn
import torch.nn.functional as F


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


class segnet_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super(segnet_encoder, self).__init__()
        self.convs_block = conv_block(in_channels, out_channels, step=1, norm=mode)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.convs_block(inputs)
        unpooled_size = outputs.size()
        outputs, indices = self.maxpool(outputs)
        return outputs, indices, unpooled_size


class segnet_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super(segnet_decoder, self).__init__()
        self.convs_block = conv_block(in_channels, out_channels, step=1, norm=mode)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.convs_block(outputs)
        return outputs

# ========================================================
# SegNet
# ========================================================


class SegNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, norm, n_classes, dropout, apply_last_layer):
        # depth: down-sampling stages
        super(SegNet, self).__init__()
        #
        self.apply_last_layer = apply_last_layer
        self.dropout_mode = dropout
        self.depth = depth
        #
        if n_classes == 2:

            output_channel = 1

        else:

            output_channel = n_classes

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        if self.dropout_mode is True:

            self.dropout_layers = nn.ModuleList()

        for i in range(self.depth):

            if i == 0:

                self.encoders.append((segnet_encoder(in_ch, width, norm)))
                self.decoders.append((segnet_decoder(width, width, norm)))

            else:

                self.encoders.append((segnet_encoder(width*(2**(i - 1)), width*(2**i), norm)))
                self.decoders.append((segnet_decoder(width*(2**i), width*(2**(i - 1)), norm)))

            if self.dropout_mode is True:

                self.dropout_layers.append(nn.Dropout2d(0.4))

        self.classification_layer = nn.Conv2d(width, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        encoder_features = []
        encoder_indices = []
        encoder_pool_shapes = []

        # print(x.shape)

        for i in range(len(self.encoders)):
            #
            x, indice, shape = self.encoders[i](x)
            #
            encoder_features.append(x)
            encoder_indices.append(indice)
            encoder_pool_shapes.append(shape)
            #
        # print(x.shape)
        for i in range(len(encoder_features)):
            #
            x = self.decoders[len(encoder_features) - i - 1](x, encoder_indices[len(encoder_features) - i - 1], encoder_pool_shapes[len(encoder_features) - i - 1])
            #
            if self.dropout_mode is True:
                #
                x = self.dropout_layers[i](x)
        #
        if self.apply_last_layer is True:
            y = self.classification_layer(x)
        else:
            y = x
        #
        return y


# =====================
# Unet
# =====================
class UNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm, dropout=False, apply_last_layer=True):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(UNet, self).__init__()
        #
        self.apply_last_layer = apply_last_layer
        self.depth = depth
        self.dropout = dropout
        #
        if class_no > 2:
            #
            self.final_in = class_no
        else:
            #
            self.final_in = 1
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()
        #
        if self.dropout is True:

            self.dropout_layers = nn.ModuleList()

        for i in range(self.depth):

            if self.dropout is True:

                self.dropout_layers.append(nn.Dropout2d(0.4))

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

    def forward(self, x):
        #
        y = x
        # print(x.shape)
        encoder_features = []
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
            diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
            diffX = torch.tensor([y_e.size()[3] - y.size()[3]])
            #
            y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            y = torch.cat([y_e, y], dim=1)
            y = self.decoders[-(i+1)](y)
            #
            if self.dropout is True:
                #
                y = self.dropout_layers[i](y)
        #
        if self.apply_last_layer is True:
            y = self.conv_last(y)
        return y
