import os
from collections import OrderedDict

import torch
import random
from torch.nn import functional as F
from models.guidingNet import GuidingNet
from models.generator import Generator
from models.discriminator import Discriminator, StyleDiscriminator

class Logger(object):
    def __init__(self, log_dir):
        self.last = None

    def scalar_summary(self, tag, value, step):
        if self.last and self.last['step'] != step:
            print(self.last)
            self.last = None
        if self.last is None:
            self.last = {'step':step,'iter':step,'epoch':1}
        self.last[tag] = value

    def images_summary(self, tag, images, step, nrow=8):
        """Log a list of images."""
        self.viz.images(
            images,
            opts=dict(title='%s/%d' % (tag, step), caption='%s/%d' % (tag, step)),
            nrow=nrow
        )


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_model(args):
    networks = {}
    opts = {}

    networks['C'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k}, args.with_attn, args.vgg_version)
    networks['G'] = Generator(args.img_size, args.sty_dim, use_sn=False, use_attn=args.with_attn)
    if not args.no_moco:
        networks['C_EMA'] = GuidingNet(args.img_size, {'cont': args.sty_dim, 'disc': args.output_k}, args.with_attn, args.vgg_version)
        networks['G_EMA'] = Generator(args.img_size, args.sty_dim, use_sn=False, use_attn=args.with_attn)
        networks['C_EMA'].load_state_dict(networks['C'].state_dict())
    if not args.validation:
        networks['D'] = Discriminator(args.img_size, num_domains=args.output_k)
        networks['Ds'] = StyleDiscriminator(channel=32, img_size=128)
        opts['C'] = torch.optim.Adam(networks['C'].parameters(), 1e-4, weight_decay=0.001)
        opts['D'] = torch.optim.RMSprop(networks['D'].parameters(), 1e-4, weight_decay=0.0001)
        opts['Ds'] = torch.optim.RMSprop(networks['Ds'].parameters(), 1e-4, weight_decay=0.0001)
        opts['G'] = torch.optim.RMSprop(networks['G'].parameters(), 1e-4, weight_decay=0.0001)

    if args.gpu is not None:
        print('Put networks to gpu %d'%args.gpu)
        for name, net in networks.items():
            networks[name] = net.cuda(args.gpu)
            print('put %s'%name)
    else:
        for name, net in networks.items():
            networks[name] = torch.nn.DataParallel(net).cuda()
    if args.validation:
        return networks
    else:
        return networks, opts

def load_model(args, networks, opts=None):
    if args.load_model is not None:
        if args.load_epoch is not None:
            to_restore = 'model_%d.ckpt'%args.load_epoch
        else:
            check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
            to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print("=> loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            for name, net in networks.items():
                if name in ['inceptionNet', 'C', 'G'] and not args.no_moco:
                    continue
                if name in ['C_EMA', 'G_EMA'] and args.no_moco:
                    continue
                try:
                    tmp_keys = next(iter(checkpoint[name + '_state_dict'].keys()))
                except:
                    tmp_keys = next(iter(checkpoint[name + '_EMA_state_dict'].keys()))
                if 'module' in tmp_keys:
                    tmp_new_dict = OrderedDict()
                    for key, val in checkpoint[name + '_state_dict'].items():
                        tmp_new_dict[key[7:]] = val
                    net.load_state_dict(tmp_new_dict)
                    networks[name] = net
                else:
                    try:
                        net.load_state_dict(checkpoint[name + '_state_dict'])
                    except:
                        net.load_state_dict(checkpoint[name + '_EMA_state_dict'])
                    networks[name] = net
            if not args.validation:
                for name, opt in opts.items():
                    opt.load_state_dict(checkpoint[name.lower() + '_optimizer'])
                    opts[name] = opt
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.log_dir))



def save_checkpoint(state, check_list, log_dir, epoch=0):
    check_file = os.path.join(log_dir, 'model_{}.ckpt'.format(epoch))
    torch.save(state, check_file)
    check_list.write('model_{}.ckpt\n'.format(epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def add_logs(args, logger, tag, value, step):
    logger.add_scalar(tag, value, step)


def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(30, height - c_h - 30)
        c_x = random.randrange(30, width - c_w - 30)

        cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

    return patches

