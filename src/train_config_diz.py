"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

DATASET_DIR='/home/avarfolomeev/Data/Instance'

args = dict(

    cuda=True,
    display=True,
    display_it=8,

    save=True,
    save_dir=DATASET_DIR + '/exp2',
    resume_path= DATASET_DIR + '/exp2/best_iou_model.pth', 

    train_dataset = {
        'name': 'diz_instances',
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type': 'train',
            'size': -1,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'size': (704, 1024),
                    }
                },
                   
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 2,
        'workers': 4
    }, 

    val_dataset = {
        'name': 'diz_instances',
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'size': (704, 1024),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 2,
        'workers': 4
    }, 

    model = {
        'name': 'branched_erfnet', 
        'kwargs': {
            'num_classes': [4,1]
        }
    }, 

    lr=5e-4,
    n_epochs=200,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': 2,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
)


def get_args():
    return copy.deepcopy(args)
