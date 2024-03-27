from torch import autograd
import torch
import torch.distributed as dist
from torch.nn import functional as F
from matplotlib.pylab import cm


def compute_grad_gp(d_out, x_in, is_patch=False):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum() if not is_patch else d_out.mean(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.sum() / batch_size
    return reg



def compute_grad_gp_wgan(D, x_real, x_fake, gpu):
    alpha = torch.rand(x_real.size(0), 1, 1, 1).cuda(gpu)

    x_interpolate = ((1 - alpha) * x_real + alpha * x_fake).detach()
    x_interpolate.requires_grad = True
    d_inter_logit = D(x_interpolate)
    grad = torch.autograd.grad(d_inter_logit, x_interpolate,
                               grad_outputs=torch.ones_like(d_inter_logit), create_graph=True)[0]

    norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)

    d_gp = ((norm - 1) ** 2).mean()
    return d_gp


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def copy_norm_params(model_tgt, model_src):
    with torch.no_grad():
        src_state_dict = model_src.state_dict()
        tgt_state_dict = model_tgt.state_dict()
        names = [name for name, _ in model_tgt.named_parameters()]
        for n in names:
            del src_state_dict[n]
        tgt_state_dict.update(src_state_dict)
        model_tgt.load_state_dict(tgt_state_dict)


def calc_iic_loss(x_out, x_tf_out, lamb=1.0, EPS=1e-10):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                             k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def calc_recon_loss(predict, target):
    return torch.mean(torch.abs(predict - target))

def calc_l1_loss(predict, target):
    l1loss = torch.nn.L1Loss()
    return l1loss(predict, target)


def calc_contrastive_loss(args, query, key, queue, temp=0.07):
    N = query.shape[0]
    K = queue.shape[0]

    zeros = torch.zeros(N, dtype=torch.long).cuda(args.gpu)
    key = key.detach()
    logit_pos = torch.bmm(query.view(N, 1, -1), key.view(N, -1, 1))
    logit_neg = torch.mm(query.view(N, -1), queue.t().view(-1, K))

    logit = torch.cat([logit_pos.view(N, 1), logit_neg], dim=1)

    loss = F.cross_entropy(logit / temp, zeros)

    return loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_sty_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    mse_loss = torch.nn.MSELoss()
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)


def calc_adv_loss(logit, mode):
    assert mode in ['d_real', 'd_fake', 'g']
    if mode == 'd_real':
        loss = F.relu(1.0 - logit).mean()
    elif mode == 'd_fake':
        loss = F.relu(1.0 + logit).mean()
    else:
        loss = -logit.mean()

    return loss


def queue_data(data, k):
    return torch.cat([data, k], dim=0)


def dequeue_data(data, K=128):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def attn_to_rgb(feat):
    feat = cm.jet(feat.squeeze().data.cpu().numpy())  # (n, 16, 16, 4)
    feat = torch.from_numpy(feat)  # (n, 16, 16, 4)
    feat = feat.transpose(2, 3).transpose(1, 2)  # (n, 4, 16, 16)
    return feat


def initialize_queue(model_k, device, train_loader, feat_size=128):
    queue = torch.zeros((0, feat_size), dtype=torch.float)
    queue = queue.to(device)

    for _, batch in enumerate(train_loader):
        x_k = batch['ref'][0]
        x_k = x_k.cuda(device)
        outs = model_k(x_k)
        k = outs['cont']
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K=128)
        break
    return queue


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        # Handle unused parameters
        if param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
