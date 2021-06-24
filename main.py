#!/usr/bin/env python
# coding: utf-8

import argparse

parser = argparse.ArgumentParser('Vox2Vox training and validation script', add_help=False)

## training parameters
parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU position')
parser.add_argument('-nc', '--num_classes', default=4, type=int, help='number of classes')
parser.add_argument('-bs', '--batch_size', default=4, type=int, help='batch size')
parser.add_argument('-a', '--alpha', default=5, type=int, help='alpha weight')
parser.add_argument('-ne', '--num_epochs', default=200, type=int, help='number of epochs')

args = parser.parse_args()
gpu = args.gpu
n_classes = args.num_classes
batch_size = args.batch_size
alpha = args.alpha
n_epochs = args.num_epochs


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import nibabel as nib
import glob
import time
from tensorflow.keras.utils import to_categorical
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from scipy.ndimage.interpolation import affine_transform
from sklearn.model_selection import train_test_split

from utils import *
from augmentation import *
from losses import *
from models import *
from train_v2v import *

Nclasses = 4
classes = np.arange(Nclasses)

# images lists
t1_list = sorted(glob.glob('../BRATS_2020/Training/*/*t1.nii'))
t2_list = sorted(glob.glob('../BRATS_2020/Training/*/*t2.nii'))
t1ce_list = sorted(glob.glob('../BRATS_2020/Training/*/*t1ce.nii'))
flair_list = sorted(glob.glob('../BRATS_2020/Training/*/*flair.nii'))
seg_list = sorted(glob.glob('../BRATS_2020/Training/*/*seg.nii'))

# create the training and validation sets
Nim = len(t1_list)
idx = np.arange(Nim)

idxTrain, idxValid = train_test_split(idx, test_size=0.25)
sets = {'train': [], 'valid': []}

for i in idxTrain:
    sets['train'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
for i in idxValid:
    sets['valid'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
    
train_gen = DataGenerator(sets['train'], batch_size=batch_size, n_classes=n_classes, augmentation=True)
valid_gen = DataGenerator(sets['valid'], batch_size=batch_size, n_classes=n_classes, augmentation=True)
    
# train the vox2vox model
h = fit(train_gen, valid_gen, alpha, n_epochs)
