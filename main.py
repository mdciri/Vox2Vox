import os
import numpy as np
import tensorflow as tf
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
    
train_gen = DataGenerator(sets['train'], augmentation=True)
valid_gen = DataGenerator(sets['valid'], augmentation=True)
    
# train the vox2vox model
h = fit(train_gen, valid_gen, 200)
