import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from .sty2_model import Blur, fused_leaky_relu, FusedLeakyReLU, ScaledLeakyReLU

import math

try:
    from models.blocks import FRN, ActFirstResBlk
except:
    from blocks import FRN, ActFirstResBlk


class Discriminator(nn.Module):
    """Discriminator: (image x, domain y) -> (logit out)."""
    def __init__(self, image_size=256, num_domains=2, max_conv_dim=1024):
        super(Discriminator, self).__init__()
        dim_in = 64 if image_size < 256 else 32
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(image_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ActFirstResBlk(dim_in, dim_in, downsample=False)]
            blocks += [ActFirstResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

        self.apply(weights_init('kaiming'))

    def forward(self, x, y):
        """
        Inputs:
            - x: images of shape (batch, 3, image_size, image_size).
            - y: domain indices of shape (batch).
        Output:
            - out: logits of shape (batch).
        """
        out = x
        for net in self.main:
            out = net(out)
        # out = self.main(x)
        feat = out
        out = out.view(out.size(0), -1)                          # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]                                         # (batch)
        return out, feat

    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

class EqualConv2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input,
                       self.weight * self.scale,
                       bias=self.bias,
                       stride=self.stride,
                       padding=self.padding)
        return out

    def __repr__(self):
        return(
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f'{self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )




class EqualConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super(EqualConvTranspose2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channel, out_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, *input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding
        )
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )



class ConvLayer(nn.Sequential):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 upsample=False,
                 downsample=False,
                 blur_kernel=(1, 3, 3, 1),
                 bias=True,
                 activate=True,
                 padding='zero'):
        layers = []
        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate
                )
            )
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
        else:
            if not downsample:
                if padding == 'zero':
                    self.padding = (kernel_size -1) // 2
                elif padding == 'reflect':
                    padding = (kernel_size - 1) // 2
                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))
                    self.padding = 0
                elif padding != 'valid':
                    raise ValueError('padding should be in (zero, reflect, valid)')

            layers.append(
                EqualConv2d(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate
                )
            )
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 downsample,
                 padding='zero',
                 blur_kernel=(1, 3, 3,1)):
        super().__init__()
        self.conv1 = ConvLayer(in_channel=in_channel,
                               out_channel=out_channel,
                               kernel_size=3,
                               padding=padding)
        self.conv2 = ConvLayer(in_channel=out_channel,
                               out_channel=out_channel,
                               kernel_size=3,
                               downsample=downsample,
                               padding=padding,
                               blur_kernel=blur_kernel)
        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel=in_channel,
                out_channel=out_channel,
                kernel_size=1,
                downsample=downsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False
            )
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input
        return (out + skip) / math.sqrt(2)

class EqualLinear(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1,
                 activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


# class ConvLayer(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=1,  stride=1, padding=0, relu=False):
#         model = [nn.Conv2d(in_channel=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
#         if relu:
#             model.append(nn.LeakyReLU(negative_slope=0.2))
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         out = self.model(input)
#         return out




class StyleDiscriminator(nn.Module):
    def __init__(self, channel=32, img_size=256):
        super(StyleDiscriminator, self).__init__()

        encoder = [ConvLayer(in_channel=3, out_channel=channel, kernel_size=1)]
        ch_multiplier = (2, 4, 8, 12, 12)
        downsample = (True, True, True, True, True)
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_channel=in_ch, out_channel=channel * ch_mul, downsample=down))
            in_ch = channel * ch_mul

        if img_size > 128:
            k_size = 2
            feat_size = 2 * 2
        else:
            k_size = 1
            feat_size = 1 * 1

        encoder.append(ConvLayer(in_channel=in_ch, out_channel=channel * 12, kernel_size=1, padding='valid'))
        self.encoder = nn.Sequential(*encoder)
        self.linear = nn.Sequential(
            EqualLinear(
                in_dim=channel * 12 * 2 * feat_size,
                out_dim=channel * 32,
                activation='fused_lrelu'
            ),
            EqualLinear(
                in_dim=channel * 32,
                out_dim=channel * 32,
                activation='fused_lrelu'
            ),
            EqualLinear(
                in_dim=channel * 32,
                out_dim=channel * 16,
                activation='fused_lrelu'
            ),
            EqualLinear(
                in_dim=channel * 16,
                out_dim=1,
            ),
        )

    def forward(self, input, ref=None, ref_batch=None, ref_input=None):
        out_input = input
        for net in self.encoder:
            out_input = net(out_input)
        # out_input = self.encoder(input)
        if ref_input is None:
            ref_input = self.encoder(ref)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, channel, height, width)
            ref_input = ref_input.mean(1)

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out, ref_input

def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

    return patches

if __name__ == '__main__':
    # D = Discriminator(64, 10)
    # x_in = torch.randn(4, 3, 64, 64)
    # y_in = torch.randint(0, 10, size=(4, ))
    # out, feat = D(x_in, y_in)
    #
    # print(out.shape, feat.shape)

    D = StyleDiscriminator(channel=32, img_size=128).cuda()
    input = torch.randn(4, 3, 128, 128).cuda()
    patch_1 = patchify_image(input, 8)
    patch_2 = patchify_image(input, 32)
    out = D(patch_1, ref=patch_2, ref_batch=4)
    print('1')