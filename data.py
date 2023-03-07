import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def load_data(opt):
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    # 有验证集的用这段代码
    # splits = ['train', 'test', 'validate']
    # drop_last_batch = {'train': True, 'test': False, 'validate': False}
    # shuffle = {'train': True, 'test': True, 'validate': True}

    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': True}

    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x],
                                                 worker_init_fn=(None if opt.manual_seed == -1 else lambda x: np.random.seed(opt.manual_seed)))
                  for x in splits}
    return dataloader
