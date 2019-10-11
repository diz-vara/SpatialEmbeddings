"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
import random

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import relabel_sequential

import torch
from torch.utils.data import Dataset
import sys


sys.path.append('../../../pyVoxels/segm-net')

from read_ontology import read_ontology

class DizInstanceDataset(Dataset):

# =============================================================================
#     class_names = ('person', 'rider', 'car', 'truck',
#                    'bus', 'train', 'motorcycle', 'bicycle')
#     class_ids = (24, 25, 26, 27, 28, 31, 32, 33)
# =============================================================================

    def __init__(self, root_dir='./', type="train", class_id=26, size=None, transform=None):

        self.ontology = read_ontology(root_dir + '/Ontology.csv')[0]
        self.class_ids = [ont.id for ont in self.ontology]
        self.class_names = [ont.name for ont in self.ontology]
        self.class_colors = [ont.color for ont in self.ontology]
        self.has_objects = [ont.has_objects for ont in self.ontology]
        
        print('Instances Dataset created from {}, type = {}, {} classes, {} with instances'.format(root_dir, 
              type, len(self.class_ids), sum(self.has_objects)))
        inst_classes = [(ont.id,ont.name) for ont in self.ontology if ont.has_objects is not 0]
        self.class_ids = [cls[0] for cls in inst_classes]
        for c in inst_classes:
            print(' [{:2d}] - "{}"'.format(c[0], c[1]))

        
        # get image and instance list
        image_dir = os.path.join(root_dir, '{}/images/*.jpg'.format(type))
        image_list = glob.glob(image_dir)
        image_list.sort()
        self.image_list = image_list

        labels_dir = os.path.join(root_dir, '{}/labels/*.png'.format(type))
        print("Labels:", labels_dir)
        labels_list = glob.glob(labels_dir)
        labels_list.sort()
        self.labels_list = labels_list



        self.class_id = class_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        print('Got {} images, {} labels'.format(self.real_size,
              len(self.labels_list)))
        assert (len(self.labels_list) == self.real_size)

    def __len__(self):

        return self.real_size

    def __getitem__(self, index):

        #index = random.randint(0, self.real_size-1)
        sample = {}

        # load image
        image = Image.open(self.image_list[index])
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        # load instances
        label = Image.open(self.labels_list[index])
        instance_map, class_map = self.decode_instance(label)
        sample['instance_map'] = instance_map
        sample['class_map'] = class_map

        # transform
        if(self.transform is not None):
            return self.transform(sample)
        else:
            return sample
    
    #@classmethod
    def decode_instance(self, pic):
        
        pic = np.array(pic, copy=False)
        instance_map = pic[:,:,1]
        class_map = pic[:,:,0] #(instance_map != 0).astype(np.uint8)

        return Image.fromarray(instance_map), Image.fromarray(class_map)
