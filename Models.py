import torch
import torch.nn as nn
import torch.nn.functional as F

from Utilis import init_weights, init_weights_orthogonal_normal, l2_regularisation
from torch.distributions import Normal, Independent, kl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UNet_CMs(nn.Module):
    """ Proposed method containing a segmentation network and a confusion matrix network.
    The segmentation network is U-net. The confusion  matrix network is defined in cm_layers

    """
    def __init__(self, in_ch, width, depth, class_no, norm='in', low_rank=False):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # rank: False
        # ===============================================================================
        super(UNet_CMs, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        self.lowrank = low_rank
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
            if self.lowrank is False:
                self.decoders_noisy_layers.append(cm_layers(in_channels=width, norm=norm, class_no=self.final_in))
            else:
                self.decoders_noisy_layers.append(low_rank_cm_layers(in_channels=width, norm=norm, class_no=self.final_in, rank=1))

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


class UNet_GlobalCMs(nn.Module):
    """ Baseline with trainable global confusion matrices.

    Each annotator is modelled through a class_no x class_no matrix, fixed for all images.
    """
    def __init__(self, in_ch, width, depth, class_no, input_height, input_width, norm='in'):
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # rank: False
        # input_height: Height of the input image
        # input_width: Width of the input image
        # ===============================================================================
        super(UNet_GlobalCMs, self).__init__()
        #
        self.depth = depth
        self.noisy_labels_no = 4
        self.final_in = class_no
        #
        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()

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

        # Define a list of global confusion matrices:
        # self.decoders_noisy_layers = []
        self.decoders_noisy_layers = nn.ModuleList()
        for i in range(self.noisy_labels_no):
            # self.decoders_noisy_layers.append(global_cm_layers(class_no, input_height, input_width))
            self.decoders_noisy_layers.append(gcm_layers(class_no, input_height, input_width))

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

        # Return the confusion matrices:
        for i in range(self.noisy_labels_no):
            # Copy the confusion matrix over the batch: (1, c, c, h , w) => (b, c, c, h, w)
            # batch_size = x.size(0)
            # y_noisy_label = self.decoders_noisy_layers[i].repeat(batch_size, 1, 1, 1, 1)
            # y_noisy.append(y_noisy_label.to(device='cuda', dtype=torch.float32))
            y_noisy.append(self.decoders_noisy_layers[i](x))
        #
        y = self.conv_last(y)
        #
        return y, y_noisy


class cm_layers(nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)

    """

    def __init__(self, in_channels, norm, class_no):
        super(cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_last = nn.Conv2d(in_channels, class_no**2, 1, bias=True)
        self.relu = nn.Softplus()

    def forward(self, x):

        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))

        return y


class gcm_layers(nn.Module):
    """ This defines the global confusion matrix layer. It defines a (class_no x class_no) confusion matrix, we then use unsqueeze function to match the
    size with the original pixel-wise confusion matrix layer, this is due to convenience to be compact with the existing loss function and pipeline.

    """

    def __init__(self, class_no, input_height, input_width):
        super(gcm_layers, self).__init__()
        self.class_no = class_no
        self.input_height = input_height
        self.input_width = input_width
        self.global_weights = nn.Parameter(torch.eye(class_no))
        self.relu = nn.Softplus()

    def forward(self, x):

        all_weights = self.global_weights.unsqueeze(0).repeat(x.size(0), 1, 1)
        all_weights = all_weights.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, self.input_height, self.input_width)
        y = self.relu(all_weights)

        return y


class low_rank_cm_layers(nn.Module):
    """ This class defines the low-rank version of the annotator network, which models the confusion matrix at low-rank approximation.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)

    """
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

        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))

        return y

# =========================
# U-net:
# =========================


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
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===========================================
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


class UNet(nn.Module):
    #
    def __init__(self, in_ch, width, depth, class_no, norm, dropout=False, apply_last_layer=True):
        """

        Args:
            in_ch:
            width:
            depth:
            class_no:
            norm:
            dropout:
            apply_last_layer:
        """

        # ============================================================================================================
        # This UNet is our own implementation, it is an enhanced version of the original UNet proposed at MICCAI 2015.
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ============================================================================================================
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


# ===============================
# Probablistic U-net
# ===============================
class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []

        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding), bias=False))
            layers.append(nn.InstanceNorm2d(output_dim, affine=True))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'

        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')

        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm

            # print(input.shape)
            # print(segm.shape)

            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        # print(input.shape)

        encoding = self.encoder(input)

        self.show_enc = encoding

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        # dist = Independent(Normal(loc=mu, scale=log_sigma), 1)
        return dist


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'

        if self.num_classes == 2:
            self.num_classes = 1

        if self.use_tile:
            layers = []

            # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0] + self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb - 2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels, num_classes, num_filters=[32, 64, 128], latent_dim=6, no_convs_fcomb=4, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 2
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 4

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=False).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, False)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)

        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + self.beta * self.kl), self.reconstruction_loss, self.beta * self.kl


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            # nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            # nn.init.normal_(self.last_layer.bias)

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        del blocks

        # Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding), bias=False))
        layers.append(nn.InstanceNorm2d(output_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding), bias=False))
        layers.append(nn.InstanceNorm2d(output_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, pool=False)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        else:
            up = self.upconv_layer(x)

        if up.shape[3] != bridge.shape[3]:
            #
            diffY = torch.tensor([bridge.size()[2] - up.size()[2]])
            diffX = torch.tensor([bridge.size()[3] - up.size()[3]])
            #
            up = F.pad(up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            #
            # print(up.shape)
            # print(bridge.shape)
            #
        # print(up.shape)
        # print(bridge.shape)

        # assert up.shape[3] == bridge.shape[3]

        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out




# def global_cm_layers(class_no, height, width):
#     """ Define (unnormalised) global confusion matrix model.
#
#     This function defines an image-level (not pixel wise) global confusion matrix for each annotator.
#     Currently, it first defines a class_no x class_no confusion matrix, and then copy this over to all
#     pixels, so this function can be more readily integrated into the existing pipeline.
#
#     Args:
#         width (int): width of the image
#         height (int): height of the image
#         class_no (int): number of classes
#
#     Returns:
#         confusion_matrix (parameter tensor): unnormalised confusion matrix of size (1, c, c, h, w).
#             The elements are ensured to be positive via a softplus function, but not normalised.
#
#     """
#     # Define global confusion matrix: (1, c, c, 1, 1)
#     weights = nn.Parameter(torch.randn(1, class_no, class_no, 1, 1))
#
#     # Broadcast to shape (1, c, c, h, w) by adding a zero tensor.
#     confusion_matrix = torch.zeros(1, class_no, class_no, height, width) + F.softplus(weights)
#     return confusion_matrix