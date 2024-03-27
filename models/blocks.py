import torch
import torch.nn.functional as F
from torch import nn


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, act, pad_type, use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = nn.ModuleList()
        for i in range(num_blocks):
            self.model.append(ResBlock(dim, norm=norm, act=act, pad_type=pad_type, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', act='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act=act,
                                               pad_type=pad_type, use_sn=use_sn),
                                   Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act='none',
                                               pad_type=pad_type, use_sn=use_sn))

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + 0.1 * residual
        return out


class AttnBlock(nn.Module):

    def __init__(self, img_size, dim, num_heads=8):

        super(AttnBlock, self).__init__()
        self.pos_embbedding = nn.Parameter(torch.randn(1, img_size ** 2, dim))  # (1, 256, 512)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.mat = torch.matmul
        self.proj = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim * 2, dim, (1, 1))

    def forward(self, style_feat):  # (Q, (KV))
        B, C, H, W = style_feat.shape
        style = style_feat.reshape(B, H * W, C)  # (4, 256, 512)

        style_token = style + self.pos_embbedding[:, :(H * W)]  # (4, 256, 512)
        style_token_ = self.q(style_token)  # (4, 256, 512)

        query = style_token_.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (4, 8, 256, 64)
        kv = self.kv(style_token).reshape(B, H*W, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()  # (2, 4, 8, 256, 64)
        key, value = kv[0], kv[1]  # (4, 8, 256, 64)
        attention_map = (self.mat(query, key.transpose(-2, -1))) * self.scale  # (4, 8, 256, 256)
        attention_map = attention_map.softmax(dim=-1)  # (4, 8, 256, 256)

        result = self.mat(attention_map, value).transpose(1, 2).reshape(B, H*W, C)  # (4, 256, 512)
        # s = result + content_token
        s = self.proj(result)
        s = s.reshape(B, C, H, W)
        feat_c_s = torch.cat((s, style_feat), dim=1)
        feat_c_s = self.conv(feat_c_s)

        return feat_c_s, attention_map


class spital_attention_v2(nn.Module):
    def __init__(self, in_channel=512):
        super(spital_attention_v2, self).__init__()
        self.sp_att = nn.Conv2d(in_channel, 2, kernel_size=1, padding=0)
        # self.softmax = torch.nn.Softmax()

    def forward(self, input):  # [n, 512, 16, 16]
        sp_w = self.sp_att(input)  # [n, 2, 16, 16]
        sp_w = sp_w.transpose(1, 2).transpose(2, 3)  # [n, 16, 16, 2]
        sp_w = sp_w.softmax(dim=-1)[:, :, :, 0]  # [n, 16, 16]
        sp_w = sp_w.unsqueeze(1)  # [n, 1, 16, 16]
        left_w = sp_w.repeat(1, input.size(1), 1, 1)  # [n, 512, 16, 16]
        out = left_w * input  # [n, 512, 16, 16]
        return out, sp_w

class ActFirstResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, downsample=True):
        super(ActFirstResBlk, self).__init__()
        self.norm1 = FRN(dim_in)
        self.norm2 = FRN(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.norm2(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        tmp = torch.rsqrt(torch.tensor(2.0)).to(x.device)
        shortcut = self._shortcut(x)
        residual = self._residual(x)
        return tmp * shortcut + tmp * residual
        # return torch.rsqrt(torch.tensor(2.0)) * self._shortcut(x) + torch.rsqrt(torch.tensor(2.0)) * self._residual(x)


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', act='relu', use_sn=False):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', act='relu', pad_type='zero',
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaIN2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)
        if use_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) + self.eps)
        return torch.max(self.gamma * x + self.beta, self.tau)


class AdaIN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaIN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "AdaIN params are None"
        N, C, H, W = x.size()
        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)
        x_ = x.contiguous().view(1, N * C, H * W)
        normed = F.batch_norm(input=x_, running_mean=running_mean, running_var=running_var, weight=self.weight, bias=self.bias, training=True, momentum=self.momentum, eps=self.eps)
        return normed.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ')'


if __name__ == '__main__':
    print("CALL blocks.py")
