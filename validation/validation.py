"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import os
import random

import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import random
import torch.nn.functional as F

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from scipy import linalg

from tools.utils import *


def validateUN(data_loader, networks, args):
    # set nets
    try:
        C_EMA = networks['C_EMA']
        G_EMA = networks['G_EMA']
    except:
        C_EMA = networks['C']
        G_EMA = networks['G']
    C_EMA.eval()
    G_EMA.eval()
    # data loader
    val_dataset = data_loader.dataset

    x_each_cls = []
    ref_each_cls = []
    sketch_name_each_cls = []
    ref_name_each_cls = []

    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()[:]
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True, drop_last=False)

            tmp_iter = iter(tmp_dl)
            if args.dataset == 'sketchy125':
                len_tmp_iter = 10
            else:
                len_tmp_iter = len(tmp_iter)
            tmp_sample = None
            tmp_name = None
            tmp_ref_name = None
            tmp_ref_sample = None
            for sample_idx in range(len_tmp_iter):
                batch = next(tmp_iter)
                imgs = batch['sketch']
                sample_ref = batch['ref'][0]
                names = batch['sketch_name']
                ref_names = batch['ref_name']
                sketch_path = batch['sketch_path']
                x_ = imgs[0]
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)

                if tmp_ref_sample is None:
                    tmp_ref_sample = sample_ref.clone()
                else:
                    tmp_ref_sample = torch.cat((tmp_ref_sample, sample_ref), 0)

                if tmp_name is None:
                    tmp_name = names
                else:
                    tmp_name = tmp_name + names

                if tmp_ref_name is None:
                    tmp_ref_name = ref_names
                else:
                    tmp_ref_name = tmp_ref_name + ref_names

            x_each_cls.append(tmp_sample)
            ref_each_cls.append(tmp_ref_sample)
            sketch_name_each_cls.append(tmp_name)
            ref_name_each_cls.append(tmp_ref_name)

        # 草图生成图像
        total_i = 0
        with torch.no_grad():
            for src_idx in range(len(args.att_to_use)):
                src_i = x_each_cls[src_idx]
                ref_i = ref_each_cls[src_idx]
                name_i = sketch_name_each_cls[src_idx]
                ref_name_i = ref_name_each_cls[src_idx]
                total_num = src_i.size()[0]
                if total_num % 10 == 0:
                    ll = 0
                else:
                    ll = 1
                for i in range(int(total_num / 10) + ll):
                    last_len = src_i.size()[0]
                    if last_len <= 10:
                        src_tmp = src_i[0: last_len, :, :, :].cuda(args.gpu, non_blocking=True)
                        ref_tmp = ref_i[0: last_len, :, :, :].cuda(args.gpu, non_blocking=True)
                        name_tmp = name_i[0: last_len]
                        ref_name_i = ref_name_i[0: last_len]
                        ref_name_tmp = ref_name_i
                    else:
                        src_tmp = src_i[0:10, :, :, :].cuda(args.gpu, non_blocking=True)
                        ref_tmp = ref_i[0:10, :, :, :].cuda(args.gpu, non_blocking=True)
                        name_tmp = name_i[0:10]
                        ref_name_tmp = ref_name_i[0:10]
                        src_i = src_i[10:, :, :, :]
                        ref_i = ref_i[10:, :, :, :]
                        name_i = name_i[10:]
                        ref_name_i = ref_name_i[10:]
                    c_src = G_EMA.cnt_encoder(src_tmp)
                    s_ref, _ = C_EMA.moco(ref_tmp)

                    blank = torch.ones((1, 3, 128, 128)).cuda(args.gpu)
                    result = torch.cat((blank, src_tmp), dim=0)
                    fake_B_res = G_EMA.decode(c_src, s_ref, with_ref=True)
                    # output_noref = G_EMA.decode(c_src, s_ref, with_ref=False)
                    # result = torch.cat((result, blank, output_noref))
                    for s_ref_i in range(ref_tmp.size(0)):
                        ref_tmp_tmp = ref_tmp[s_ref_i, :, :, :].unsqueeze(0)
                        result = torch.cat((result, ref_tmp_tmp), dim=0)
                        s_ref_tmp_tmp = s_ref[s_ref_i, :].unsqueeze(0)
                        for s_src_i in range(src_tmp.size(0)):
                            c_src_tmp_tmp = c_src[s_src_i, :, :, :].unsqueeze(0)
                            output = G_EMA.decode(c_src_tmp_tmp, s_ref_tmp_tmp, with_ref=True)
                            result = torch.cat((result, output), dim=0)

                    for sub_dir in args.att_to_use:
                        if not os.path.exists(os.path.join(args.res_dir, 'fakeB', str(sub_dir).zfill(2))):
                            os.makedirs(os.path.join(args.res_dir, 'fakeB', str(sub_dir).zfill(2)))
                        if not os.path.exists(os.path.join(args.res_dir, 'ref', str(sub_dir).zfill(2))):
                            os.makedirs(os.path.join(args.res_dir, 'ref', str(sub_dir).zfill(2)))
                    vutils.save_image(result,
                                      os.path.join(args.res_dir, '{}_{}.jpg'.format(src_idx, i)),
                                      normalize=True,
                                      nrow=fake_B_res.size(0) + 1)
                    f = open(os.path.join(args.res_dir, '{}_{}.txt'.format(src_idx, i)), 'w')
                    for ref_name in ref_name_tmp:
                        f.write(ref_name + '\n')
                    f.close()
                    # 生成fakeB，方便计算指标
                    for res_single_id in range(fake_B_res.size()[0]):
                        vutils.save_image(fake_B_res[res_single_id, :, :, :],
                                          os.path.join(args.res_dir, 'fakeB', str(src_idx).zfill(2),
                                                       '%s' % name_tmp[res_single_id]),
                                          normalize=True)
                        vutils.save_image(ref_tmp[res_single_id, :, :, :],
                                          os.path.join(args.res_dir, 'ref', str(src_idx).zfill(2),
                                                       '%s' % name_tmp[res_single_id]),
                                          normalize=True)
                        total_i += 1
                        print(total_i)
                    # 同时生成风格参考




def validateOne(networks, sketch, ref, target_path, args, getALL=False):
    sketch = sketch.cuda(args.gpu)
    ref = ref.cuda(args.gpu)
    try:
        G = networks['G_EMA'].cuda(args.gpu)
        C = networks['C_EMA'].cuda(args.gpu)
    except:
        G = networks['G'].cuda(args.gpu)
        C = networks['C'].cuda(args.gpu)

    G.eval()
    C.eval()

    s_ref, _ = C.moco(ref)
    c_src = G.cnt_encoder(sketch)
    output = G.decode(c_src, s_ref, with_ref=True)
    if getALL:
        output = torch.cat((sketch, ref, output), dim=0)
    vutils.save_image(output, target_path, normalize=True)


def validateInterpolation(networks, sketch, ref1, ref2, ratio, target_path, args):
    G = networks['G_EMA'].cuda(args.gpu)
    C = networks['C_EMA'].cuda(args.gpu)

    G.eval()
    C.eval()

    s_ref1, _ = C.moco(ref1)
    s_ref2, _ = C.moco(ref2)
    c_src = G.cnt_encoder(sketch)
    s_ref = ratio * s_ref1 + (1 - ratio) * s_ref2
    output = G.decode(c_src, s_ref, with_ref=True)
    vutils.save_image(output, target_path, normalize=True)


def get_style_feature(networks, ref, args):
    ref = ref.cuda(args.gpu)
    try:
        C = networks['C_EMA'].cuda(args.gpu)
    except:
        C = networks['C'].cuda(args.gpu)

    C.eval()

    s_ref, _ = C.moco(ref)
    return s_ref
