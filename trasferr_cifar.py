import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pickle as pkl
import os
from os.path import join
head_to_class = {'cifar10': {7:4, 8:0, 9:0},'cifar100':{78:18, 79:44, 88:50, 89:8, 98:2, 99:61}}
new_h_to = {'cifar10': {7:4, 8:0},'cifar100':{78:18, 79:44, 88:50, 89:8, 98:2, 99:61}}

class TRANSCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False, change_portion=0.1, t_h_file=None):
        super(TRANSCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)

        new_data, new_targets = self.gen_tail_class_first(img_num_list, t_h_file, change_portion=change_portion)

        self.gen_imbalanced_data(new_data, new_targets, img_num_list)

    def gen_tail_class_first(self, img_num_list, t_h_file, change_portion=0.1):
        new_data = []
        new_targets = []
        with open(t_h_file, 'rb') as f:
            t_h_list = pkl.load(f)

        for item in t_h_list:
            tail = item['tail']
            head = item['head']
            idx = item['samples'][0]
            print(head, tail, len(idx))

            num_to_change = (int)(np.round(img_num_list[head] * change_portion))
            head_idx = [i for i, item in enumerate(self.targets) if item == head]
            selec_idx = idx[:num_to_change].astype(np.int64)
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([tail, ] * num_to_change)
            for idx in selec_idx:
                self.targets[idx] = tail

        return new_data, new_targets

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

    def gen_imbalanced_data(self, new_data, new_targets, img_num_per_cls):
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            print(the_class, the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class TRANSCIFAR100(TRANSCIFAR10):
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
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = TRANSCIFAR10(root='./data', train=True,
                                download=True, transform=transform, change_portion=0.1,
                                t_h_file='./data/cifar10_resnet32_CE_None_exp_0.1_0.pickle')
    with open('./data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle', 'rb') as f:
        t_h_list = pkl.load(f)

    trainloader = iter(trainset)
    data, label = next(trainloader)
    print(data.shape, label)
#    import pdb; pdb.set_trace()
