import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pickle as pkl
import pandas as pd
import os
import collections
from PIL import Image

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

head_to_class = {'cifar10': {8: 0}, 'cifar100': {78: 18, 79: 44, 88: 50, 89: 8, 98: 2, 99: 61}}
tail_to_class = {'cifar10': {0: 8}, 'cifar100': {18: 78, 44: 79, 50: 88, 8: 89, 2: 98, 61: 99}}


class CUSTUMCIFAR100():
    """CIFAR-100 Long Tail Dataset"""

    def __init__(self, root_dir, split='train'):
        self.file_list = pd.read_csv(os.path.join(root_dir, 'cifar100_lt/data/cifar100_lt.csv'))
        self.root_dir = root_dir
        self.split = split
        if self.split == 'train':
            self.transform = transform_train
        else:
            self.transform = transform_val
        self.read_csv()

    def __len__(self):
        return len(self.img_list)

    def read_csv(self):
        if self.split == 'test':
            self.file_list = self.file_list[-9999:]
        else:
            self.file_list = self.file_list[:-9999]

        self.file_list.columns = ['img_paths', 'labels']
        self.file_list.sort_values('labels')
        self.img_list = self.file_list['img_paths']
        label_list = self.file_list['labels']
        self.sorted_count = collections.Counter(label_list).most_common()

        self.label_idx = {label[0]: index for index, label in enumerate(self.sorted_count)}
        self.idx_label = {i: k for i, k in enumerate(self.label_idx)}

        self.file_list['class_index'] = self.file_list['labels'].map(self.label_idx)
        self.file_list.sort_values(['class_index'], ascending=[True], inplace=True)
        self.file_list.drop('class_index', 1, inplace=True)

        self.img_list = []
        self.label_list = []
        if self.split == 'test':
            self.img_list = self.file_list['img_paths'].values
            return
        count = 0
        for label, total in self.sorted_count:
            train_num = int(np.around(total * 0.8))
            if self.split == 'train':
                self.img_list = [*self.img_list, *self.file_list[count:count + train_num]['img_paths'].values]
                self.label_list = [*self.label_list, *self.file_list[count:count + train_num]['labels'].values]
            elif self.split == 'val':
                self.img_list = [*self.img_list, *self.file_list[count + train_num:count + total]['img_paths'].values]
                self.label_list = [*self.label_list, *self.file_list[count + train_num:count + total]['labels'].values]
            count += total
        self.label_list = [self.label_idx[label] for label in self.label_list]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.root_dir, self.img_list[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        if self.split == 'test':
            return img, self.img_list[idx]

        label = self.label_list[idx]

        return img, label

    def get_cls_num_list(self):
        cls_num_list = []
        for item in self.sorted_count:
            cls_num_list.append(item[1])
        return cls_num_list


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 target_transform=None, download=False, one_class=-1, t_as_h=-1):
        if train:
            transform = transform_train
        else:
            transform = transform_val

        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.one_class = one_class
        self.t_as_h = t_as_h

        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.head_to_class = head_to_class['cifar%d' % (len(img_num_list))]
        self.tail_to_class = tail_to_class['cifar%d' % (len(img_num_list))]
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        if self.one_class != -1:
            the_img_num = img_num_per_cls[self.one_class]
            self.num_per_cls_dict[self.one_class] = the_img_num
            idx = np.where(targets_np == self.one_class)[0]
            np.random.shuffle(idx)
            self.selec_idx = idx[:the_img_num]
            new_data.append(self.data[self.selec_idx, ...])
            new_targets.extend([self.one_class, ] * the_img_num)
            new_data = np.vstack(new_data)
            self.data = new_data
            self.targets = new_targets
        else:

            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                new_data.append(self.data[selec_idx, ...])
                if (self.t_as_h == 0) and (the_class in self.head_to_class.keys()):
                    print('tail to head %d -> %d' % (the_class, self.head_to_class[the_class]))
                    new_targets.extend([self.head_to_class[the_class], ] * the_img_num)
                elif (self.t_as_h == 1) and (the_class in self.tail_to_class.keys()):
                    print('tail to head %d -> %d' % (the_class, self.tail_to_class[the_class]))
                    new_targets.extend([self.tail_to_class[the_class], ] * the_img_num)
                else:
                    new_targets.extend([the_class, ] * the_img_num)
            new_data = np.vstack(new_data)
            self.data = new_data
            self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    # head_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    # head_dataset.
    trainset = IMBALANCECIFAR10(root='./data', train=False, download=True, t_as_h=True)

    trainloader = iter(trainset)
    data, label = next(trainloader)
    print(type(data), type(label))
#    import pdb; pdb.set_trace()
