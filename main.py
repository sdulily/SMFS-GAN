import argparse
from datetime import datetime
from glob import glob
from shutil import copyfile

import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models.tom_model import TomGenerator, VGGSimple

from train.train_smfs import trainGAN_SUP
from train.train_ablation import train_no_moco

from validation.validation import validateUN

from tools.utils import *
from datasets.multiclassdataset import MutiClassDataset
from torch.utils.data import DataLoader
from tools.ops import initialize_queue

from tensorboardX import SummaryWriter

# Configuration
parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument('--dataset', default='qmul_new', help='Dataset name to use',
                    choices=['qmul', 'qmul_new', 'sketchyCOCO', 'sketchy125'])
parser.add_argument('--data_path', type=str, default='./data',
                    help='Dataset directory. Please refer Dataset in README.md')
parser.add_argument('--model_name', type=str, default='SMFS-GAN',
                    help='Prefix of logs and results folders. '
                         'ex) --model_name=ABC generates ABC_20230101-140517 in logs and results')

parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to run. Not actual epoch.')
parser.add_argument('--iters', default=1000, type=int, help='Total number of iterations per epoch')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--val_batch', default=10, type=int,
                    help='Batch size for validation. '
                         'The result images are stored in the form of (val_batch, val_batch) grid.')
parser.add_argument('--log_step', default=100, type=int)

parser.add_argument('--sty_dim', default=128, type=int, help='The size of style vector')
parser.add_argument('--output_k', default=1, type=int, help='Total number of classes to use')
parser.add_argument('--img_size', default=128, type=int, help='Input image size')



parser.add_argument('--load_model', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--load_epoch', default=None, type=int)
parser.add_argument('--validation', dest='validation', action='store_true', default=False,
                    help='Call for valiation only mode')

parser.add_argument('--with_tom',  action='store_true', default=False, help='use TOM Edge detector')
parser.add_argument('--with_attn',  action='store_true', default=False, help='use self-attn block')
parser.add_argument('--tom_ckpt', type=str, default='./models/4999_model.pth', help='path of TOM')
parser.add_argument('--no_moco', action='store_true', default=False, help='use no moco mode')
parser.add_argument('--n_crop', type=int, default=8, help='crop of local discriminator')
parser.add_argument('--ref_crop', type=int, default=4, help='crop of local discriminator')
parser.add_argument('--negative_num', type=int, default=0, help='number of negative samples, if set to 0, use the queue as negative samples')

parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')

parser.add_argument('--vgg_version', default='vgg11', type=str, help="version of vgg, like vgg11, vgg13, vgg16 and vgg19")
parser.add_argument('--w_gp', default=10.0, type=float, help='Coefficient of GP of D')
parser.add_argument('--w_adv', default=1.0, type=float, help='Coefficient of Adv. loss of G')
parser.add_argument('--w_vec', default=0.1, type=float, help='Coefficient of Style vector rec. loss of G')
parser.add_argument('--w_edge_loss', default=10.0, type=int, help='Coefficient of L1loss of rec edge')
parser.add_argument('--w_sty', default=1.0, type=float, help='local discriminator loss')
parser.add_argument('--w_Dg', default=1.0, type=float, help='global discriminator loss')
parser.add_argument('--w_sty_recon', default=1.0, type=float, help='sty reconstruction loss')
parser.add_argument('--w_rec', default=0.0, type=float, help='tunit-origin rec loss')

def main():
    ####################
    # Default settings #
    ####################
    args = parser.parse_args()
    print("PYTORCH VERSION", torch.__version__)
    args.data_dir = args.data_path
    args.start_epoch = 0

    ngpus_per_node = torch.cuda.device_count()

    # Logs / Results
    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.event_dir = os.path.join(args.log_dir, 'events')
    if args.load_epoch is None:
        args.res_dir = os.path.join('./logs', args.model_name)
    else:
        args.res_dir = os.path.join('./logs', args.model_name, str(args.load_epoch))

    makedirs(args.log_dir)
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.idea', '.git', 'logs', 'results', '.gitignore', '.nsmlignore', 'resrc']
    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))
    makedirs(args.res_dir)
    makedirs(os.path.join(args.log_dir, 'train_results'))

    if args.load_model is None:
        pyfiles = glob("./*.py")
        for py in pyfiles:
            copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

        for to_make in dirs_to_make:
            if to_make in not_dirs:
                continue
            tmp_files = glob(os.path.join('./', to_make, "*.py"))
            for py in tmp_files:
                copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))


    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):

    if args.gpu is not None:
        args.gpu = int(args.gpu)
        print("Use GPU: {} for training".format(args.gpu))

    # # of GT-classes
    args.num_cls = args.output_k

    # Classes to use
    if args.dataset == 'qmul' or args.dataset == 'qmul_new' or args.dataset == 'QMUL_NEW':
        args.att_to_use = [0, 1, 2]
        args.output_k = 3
    elif args.dataset == 'sketchyCOCO':
        args.att_to_use = [i for i in range(14)]
        args.output_k = 14
    elif 'sketchy125' in args.dataset:
        args.att_to_use = [i for i in range(125)]
        args.output_k = 125

    # Logging
    if not args.validation:
        logger = SummaryWriter(args.event_dir)

    # build model - return dict
    if args.validation:
        networks = build_model(args)
    else:
        networks, opts = build_model(args)

    print('Network Build Completed')

    # load model if args.load_model is specified
    load_model(args, networks)

    # image to sketch model
    if args.with_tom:
        networks['TOM'] = TomGenerator(infc=256, nfc=128)
        tom_checkpoint = torch.load(args.tom_ckpt, map_location=lambda storage, loc: storage)
        networks['TOM'].load_state_dict(tom_checkpoint['g'])
        networks['TOM'].cuda(args.gpu)
        networks['TOM'].eval()
        for p in networks['TOM'].parameters():
            p.requires_grad = False

        networks['TOM_VGG'] = VGGSimple()
        networks['TOM_VGG'].load_state_dict(torch.load('./models/vgg-feature-weights.pth', map_location=lambda a, b: a))
        networks['TOM_VGG'].cuda(args.gpu)
        networks['TOM_VGG'].eval()
        for p in networks['TOM_VGG'].parameters():
            p.requires_grad = False


    cudnn.benchmark = True

    train_dataloader = DataLoader(MutiClassDataset(args, 'train'),
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=0)

    trainFunc, validationFunc = map_exec_func(args)

    queue_loader = train_dataloader

    if not args.no_moco:
        queue = initialize_queue(networks['C_EMA'], args.gpu, queue_loader, feat_size=args.sty_dim)

    # print all the argument
    print_args(args)

    # All the test is done in the training - do not need to call
    if args.validation:
        test_dataloader = DataLoader(MutiClassDataset(args, 'test'),
                                     shuffle=True,
                                     batch_size=args.batch_size,
                                     num_workers=0)
        validationFunc(test_dataloader, networks, args)
        return

    # For saving the model
    record_txt = open(os.path.join(args.log_dir, "record.txt"), "a+")
    for arg in vars(args):
        record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
    record_txt.close()

    save_model(args, 0, networks, opts)
    print('save model 0')
    for epoch in range(args.start_epoch, args.epochs):
        print("START EPOCH[{}]".format(epoch+1))
        if (epoch + 1) % (args.epochs // 10) == 0:
            save_model(args, epoch, networks, opts)

        networks['G_EMA'].load_state_dict(networks['G'].state_dict())
        if args.no_moco:
            train_no_moco(train_dataloader, networks, opts, epoch, args, logger)
        else:
            trainFunc(train_dataloader, networks, opts, epoch, args, {'logger': logger, 'queue': queue})

        # Write logs
        if (epoch + 1) % 5 == 0:
            save_model(args, epoch, networks, opts)

#################
# Sub functions #
#################
def print_args(args):
    for arg in vars(args):
        print('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))

def map_exec_func(args):
    trainFunc = trainGAN_SUP
    validationFunc = validateUN
    return trainFunc, validationFunc

def save_model(args, epoch, networks, opts):
    check_list = open(os.path.join(args.log_dir, "checkpoint.txt"), "a+")
    # if (epoch + 1) % (args.epochs//10) == 0:
    with torch.no_grad():
        save_dict = {}
        save_dict['epoch'] = epoch + 1
        for name, net in networks.items():
            if name in ['D', 'Ds', 'TOM', 'TOM_VGG']:
                continue
            save_dict[name+'_state_dict'] = net.state_dict()
        print("SAVE CHECKPOINT[{}] DONE".format(epoch+1))
        save_checkpoint(save_dict, check_list, args.log_dir, epoch + 1)
    check_list.close()

if __name__ == '__main__':
    main()
