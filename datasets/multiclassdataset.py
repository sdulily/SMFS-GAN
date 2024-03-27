"""
# File       : multiclassdataset.py
# Time       : 2023/8/13 12:43
# Author     : czw
# software   : PyCharm , Python3.7
# Description: 
"""
import os
import random
import sys

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


class MutiClassDataset(data.Dataset):
    def __init__(self, args, mode='train'):
        root = os.path.join(args.data_dir, args.dataset, mode)
        classes, class_to_idx = self._find_classes(root)
        sketches, edges = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(sketches) == 0:
            raise (RuntimeError('Found 0 imgs in subfolders of:%s' % root))
        else:
            print('Use %s dataset, total %d imgs' % (args.dataset, len(sketches)))

        if args.dataset == 'qmul' or args.dataset == 'qmul_new' or args.dataset == 'QMUL_NEW':
            args.att_to_use = [0, 1, 2]
        elif args.dataset == 'sketchyCOCO':
            args.att_to_use = [i for i in range(14)]
        elif args.dataset == 'sketchy125':
            args.att_to_use = [i for i in range(125)]
        else:
            raise RuntimeError('unsupported dataset of %s' % args.dataset)

        remap_table = {}
        i = 0
        for k in args.att_to_use:
            remap_table[k] = i
            i += 1

        self.negative_nums = args.negative_num
        self.root = root
        self.extensions = IMG_EXTENSIONS
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.sketches = sketches
        self.edges = edges
        self.targets = [s[1] for s in sketches]
        self.class_table = remap_table
        if args.dataset == 'sketchy125':
            self.imgs_ref = get_ref(os.path.join(self.root, 'image', 'tx_000000000000'), self.classes)
        else:
            self.imgs_ref = get_ref(os.path.join(self.root, 'image'), self.classes)
        self.transform = get_transform(args, mode)
        print('dataset init completed')

    def __getitem__(self, index):
        sketch_path, target = self.sketches[index]

        # edge_path, _ = self.edges[index]
        ref_path = random.choice(self.imgs_ref[target])
        if len(sketch_path.split('\\')) > 1:
            sketch_name = sketch_path.split('\\')[-1]
            ref_name = ref_path.split('\\')[-1]
        else:
            sketch_name = sketch_path.split('/')[-1]
            ref_name = ref_path.split('/')[-1]


        ref = Image.open(ref_path).convert('RGB')
        ref = self.transform(ref)
        negative_ref = []
        if not self.negative_nums == 0:
            negative_ref_paths = []
            for i in range(self.negative_nums):
                tmp_ref_path = random.choice(self.imgs_ref[target])
                while tmp_ref_path in negative_ref_paths or tmp_ref_path == ref_path:
                    tmp_ref_path = random.choice(self.imgs_ref[target])
                tmp_ref = Image.open(tmp_ref_path).convert('RGB')
                tmp_ref = self.transform(tmp_ref)[0]
                negative_ref.append(tmp_ref)

        sketch = Image.open(sketch_path).convert('RGB')
        sketch = self.transform(sketch)


        target = self.class_table[target]
        return {'sketch': sketch,  'target': target, 'ref': ref, 'negative_ref': negative_ref, 'sketch_name': sketch_name, 'ref_name': ref_name, 'sketch_path': sketch_path}

    def __len__(self):
        return len(self.sketches)



    def _find_classes(self, dir):
        dir = os.path.join(dir, 'sketch')
        if sys.version_info >= (3, 5):
            if 'sketchy125' in dir:
                type_list = os.listdir(dir)
                dir = os.path.join(dir, type_list[0])
                classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            else:
                classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx



def make_dataset(dir, class_to_idx, extensions):
    sketches = []
    edges = []
    dirA = os.path.join(os.path.expanduser(dir), 'sketch')
    dirC = os.path.join(os.path.expanduser(dir), 'edge')

    if 'sketchy125' in dir:
        type_list = os.listdir(dirA)
        type = type_list[0]
        for target in sorted(class_to_idx.keys()):
            # print(target)
            d_A = os.path.join(dirA, type, target)
            for root, _, fnames in sorted(os.walk(d_A)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        sketches.append(item)

    else:
        for target in sorted(class_to_idx.keys()):
            d_A = os.path.join(dirA, target)
            d_C = os.path.join(dirC, target)
            if not os.path.isdir(d_A):
                continue
            if not os.path.isdir(d_C):
                continue

            for root, _, fnames in sorted(os.walk(d_A)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        sketches.append(item)
            for root, _, fnames in sorted(os.walk(d_C)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        edges.append(item)
    return sketches, edges


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def get_ref(root_path, classes):
    ref = []
    dir = root_path
    for class_ in classes:
        ref_ = []
        dir_ = os.path.join(dir, class_)
        names = os.listdir(dir_)
        for i in names:
            ref_.append(os.path.join(dir_, i))
        ref.append(ref_)
    return ref



class DuplicatedCompose(object):
    def __init__(self, tf1, tf2):
        self.tf1 = tf1
        self.tf2 = tf2

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t1 in self.tf1:
            img1 = t1(img1)
        for t2 in self.tf2:
            img2 = t2(img2)
        return img1, img2

def get_transform(args, mode):
    if mode == 'train':
        transform = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])],
                                  [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                   transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                ratio=(0.9, 1.1), interpolation=2),
                                   transforms.ColorJitter(0.1, 0.1, 0.1, 0.025),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:
        transform = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                        ratio=(0.9, 1.1), interpolation=2),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    return transform