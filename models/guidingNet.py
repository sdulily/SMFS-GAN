from torch import nn
import torch.nn.functional as F

try:
    from models.blocks import Conv2dBlock, FRN, AttnBlock, spital_attention_v2
except:
    from blocks import Conv2dBlock, FRN, AttnBlock, spital_attention_v2


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19cut': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'N'],
}

attn_layer = {
    'vgg11': 20,
    'vgg13': 26,
    'vgg16': 32,
    'vgg19': 38,
}


class GuidingNet(nn.Module):
    def __init__(self, img_size=64, output_k={'cont': 128, 'disc': 10}, self_attention=False, vgg_version="vgg11"):
        super(GuidingNet, self).__init__()
        # network layers setting
        self.features = make_layers(cfg[vgg_version], True)

        self.disc = nn.Linear(512, output_k['disc'])
        self.cont = nn.Linear(512, output_k['cont'])

        self._initialize_weights()
        self.vgg_version = vgg_version
        self.self_attention = self_attention
        # if vgg_version != 'vgg11' :
        #     self.self_attention = False
        if self.self_attention:
            # self.attn_block = AttnBlock(img_size=16, dim=512, num_heads=8)
            self.attn_block = spital_attention_v2(in_channel=512)

    def forward(self, x, sty=False):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        if sty:
            return cont
        disc = self.disc(flat)
        return {'cont': cont, 'disc': disc}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def moco(self, x, get_attn_map=False):
        sty_feats = []
        for i, l in enumerate(self.features):
            x = l(x)
            if i == attn_layer[self.vgg_version] and self.self_attention:
                before_att_feat = x
                x, attn_map = self.attn_block(x)
                after_att_feat = x
            if i in [2, 10, 17, 28]:
                sty_feats.append(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        if self.self_attention and get_attn_map:
            return cont, attn_map, before_att_feat, after_att_feat, sty_feats
        else:
            return cont, sty_feats

    def iic(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        disc = self.disc(flat)
        return disc


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    import torch
    C = GuidingNet(64)
    x_in = torch.randn(4, 3, 64, 64)
    sty = C.moco(x_in)
    cls = C.iic(x_in)
    print(sty.shape, cls.shape)
