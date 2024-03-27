from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss, calc_l1_loss, d_logistic_loss, \
    g_nonsaturating_loss, attn_to_rgb, calc_sty_loss
import torchvision.utils as vutils


def train_no_moco(data_loader, networks, opts, epoch, args, logger):
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()

    D = networks['D']
    Ds = networks['Ds']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module

    if args.with_tom:
        TOM = networks['TOM']
        TOM_VGG = networks['TOM_VGG']

    d_opt = opts['D']
    ds_opt = opts['Ds']
    g_opt = opts['G']
    c_opt = opts['C']
    # switch to train mode
    D.train()
    G.train()
    C.train()

    train_it = iter(data_loader)
    logger = logger

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            batch = next(train_it)
            imgs = batch['sketch']
            y_org = batch['target']
            x_ref = batch['ref']
        except:
            train_it = iter(data_loader)
            batch = next(train_it)
            imgs = batch['sketch']
            y_org = batch['target']
            x_ref = batch['ref']

        x_org = imgs[0]

        x_ref_tf = x_ref[1]
        x_ref = x_ref[0]


        # 自己的训练方法
        x_org = x_org.cuda(args.gpu)
        y_org = y_org.cuda(args.gpu)
        x_ref_idx = [i for i in range(x_org.size(0))]
        x_ref = x_ref.cuda(args.gpu)
        x_ref_tf = x_ref_tf.cuda(args.gpu)

        with_ref = True
        with torch.no_grad():  # 更新判别器不需要梯度
            y_ref = y_org.clone()
            y_ref = y_ref[x_ref_idx]
            s_ref, _ = C.moco(x_ref)
            c_src = G.cnt_encoder(x_org)
            x_fake = G.decode(c_src, s_ref, with_ref)
        x_ref.requires_grad_()

        d_real_logit, _ = D(x_ref, y_ref)
        d_fake_logit, _ = D(x_fake.detach(), y_ref)

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        d_adv = d_adv_real + d_adv_fake

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        if args.distributed:
            average_gradients(D)
        d_opt.step()

        ###################
        # Style Discriminator
        ###################
        fake_patch = patchify_image(x_fake, args.n_crop)
        real_patch = patchify_image(x_ref, args.n_crop)
        ref_patch = patchify_image(x_ref, args.n_crop * args.ref_crop)
        fake_patch_pred, ref_input = Ds(fake_patch, ref_patch, ref_batch=args.ref_crop)
        real_patch_pred, _ = Ds(real_patch, ref_input=ref_input)
        Ds_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)

        ds_opt.zero_grad()
        Ds_loss.backward()
        ds_opt.step()

        # Train G
        if args.with_attn:
            s_ref, attn_map, before_att_feat, after_att_feat, sty_feats = C.moco(x_ref, get_attn_map=True)

            before_att_feat = torch.mean(before_att_feat, 1, keepdim=True)
            before_att_feat = attn_to_rgb(before_att_feat)[:, :3, :, :]  # (n, 3, 16, 16)
            before_att_feat = torch.nn.functional.interpolate(before_att_feat, (128, 128), mode='bilinear').cuda(args.gpu).to(torch.float32)


            attn_map = torch.nn.functional.interpolate(attn_map, size=(128, 128), mode='bilinear')
            attn_map = attn_to_rgb(attn_map)[:, :3, :, :].cuda(args.gpu).to(torch.float32)

            after_att_feat = torch.mean(after_att_feat, 1, keepdim=True)
            after_att_feat = attn_to_rgb(after_att_feat)[:, :3, :, :]
            after_att_feat = torch.nn.functional.interpolate(after_att_feat, (128, 128), mode='bilinear').cuda(args.gpu).to(torch.float32)

        else:
            s_ref, sty_feats = C.moco(x_ref, get_attn_map=False)

        c_src = G.cnt_encoder(x_org)
        x_fake = G.decode(c_src, s_ref, with_ref)

        # fakeB提取边缘，和边缘图做损失
        if args.with_tom:
            rec_edge_feat = TOM_VGG(x_fake)
            rec_edge = TOM(rec_edge_feat[2])
            fakeB_cnt_feat = G.cnt_encoder(rec_edge)
        else:
            fakeB_cnt_feat = G.cnt_encoder(x_fake)

        # 边缘重建余弦相似度
        realA_cnt_feat = G.cnt_encoder(x_org)
        N_, C_, H_, W_ = fakeB_cnt_feat.size()
        cos_x = realA_cnt_feat.view(N_, -1)
        cos_y = fakeB_cnt_feat.view(N_, -1)

        cos_one = torch.ones((1)).cuda(args.gpu)
        g_edge_loss = (cos_one - torch.cosine_similarity(cos_x, cos_y, dim=1).squeeze().mean(0)).squeeze().cuda(args.gpu)

        if i % 200 == 0:
            if args.with_attn:
                train_result = torch.cat((x_org, x_ref, x_fake, before_att_feat, attn_map, after_att_feat), dim=0)
            else:
                train_result = torch.cat((x_org, x_ref, x_fake), dim=0)
            vutils.save_image(train_result, os.path.join(args.log_dir, 'train_results', 'epoch%d_iter%d.png' % (epoch, i)),
                              normalize=True, nrow=args.batch_size)
        fake_patch = patchify_image(x_fake, args.n_crop)
        ref_patch = patchify_image(x_ref, args.n_crop * args.ref_crop)
        fake_patch_pred, _ = Ds(fake_patch, ref_patch, ref_batch=args.ref_crop)
        g_ds_loss = g_nonsaturating_loss(fake_patch_pred)

        g_fake_logit, _ = D(x_fake, y_ref)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')

        g_adv = g_adv_fake

        # 重建
        if args.w_rec > 0.0:
            s_src, _ = C.moco(x_org)
            x_rec = G.decode(c_src, s_src)
            g_rec_logit, _ = D(x_rec, y_ref)
            g_adv_rec = calc_adv_loss(g_rec_logit, 'g')
            g_adv_fake += g_adv_rec
            g_imgrec = calc_recon_loss(x_rec, x_org)
        else:
            g_imgrec = torch.zeros([1]).cuda(args.gpu)

        g_sty_recon = torch.zeros([1]).cuda(args.gpu)
        if args.w_sty_recon > 0.0:
            s_fake, style_feats_fake = C.moco(x_fake)
            with torch.no_grad():
                s_ref_ema, style_feat_ref = C.moco(x_ref)
            for fake_feat_i, ref_feat_i in zip(style_feats_fake, style_feat_ref):
                g_sty_recon += calc_sty_loss(fake_feat_i, ref_feat_i)
        g_loss = args.w_adv * (g_adv + g_ds_loss) + args.w_edge_loss * g_edge_loss \
                 + args.w_sty_recon * g_sty_recon + args.w_rec * g_imgrec

        g_opt.zero_grad()
        c_opt.zero_grad()
        g_loss.backward()
        if args.distributed:
            average_gradients(G)
            average_gradients(C)
        c_opt.step()
        g_opt.step()

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0:
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)
                add_logs(args, logger, 'D/DS', Ds_loss, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/DS', g_ds_loss, summary_step)
                add_logs(args, logger, 'G/edgeREC', g_edge_loss, summary_step)
                add_logs(args, logger, 'G/imgRec', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/styREC', g_sty_recon, summary_step)

                print('Epoch: [{}/{}] [{}/{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}]'.format(epoch + 1, args.epochs, i + 1, args.iters,
                                                        d_losses=d_losses, g_losses=g_losses))
