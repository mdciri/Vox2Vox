import numpy as np
import tensorflow as tf
import nibabel as nib
import glob
import time
import concurrent.futures
from tensorflow.keras.utils import to_categorical
from sys import stdout

from image_loader import *
from data_generator import *
from vox2vox_architecture import *

Nclasses = 7
classes = np.arange(Nclasses)

# images lists
# here you should change your path
t1_list = sorted(glob.glob('/nobackup/data/marci30/*/*/*t1.nii'))
t2_list = sorted(glob.glob('/nobackup/data/marci30/*/*/*t2.nii'))
t1ce_list = sorted(glob.glob('/nobackup/data/marci30/*/*/*t1ce.nii'))
flair_list = sorted(glob.glob('/nobackup/data/marci30/*/*/*flair.nii'))
seg_list = sorted(glob.glob('/nobackup/data/marci30/SEG_unzipped/*/*completeSeg.nii'))

# -----------------------
# split the data
idxTrain, idxValid, idxTest = np.load('idxTrain.npy'), np.load('idxValid.npy'), np.load('idxTest.npy')
print('Training, validation and testing set have lenghts: {}, {} and {} respectively.'.format(len(idxTrain), len(idxValid), len(idxTest)))

sets = {'train': [], 'valid': [], 'test': []}

for i in idxTrain:
    sets['train'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
for i in idxValid:
    sets['valid'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
for i in idxTest:
    sets['test'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
    
# -----------------------
# data generators
train_gen = DataGenerator(sets['train'], augmentation=True)
valid_gen = DataGenerator(sets['valid'], augmentation=True)

# -----------------------
# load class weights
class_weights = np.load('class_weights.npy')
print(class_weights)

# -----------------------
# Vox2Vox model
imShape = (64, 64, 64, 4)
gtShape = (64, 64, 64, 7)
gan = vox2vox(imShape, gtShape, class_weights, depth=4, batch_size=32, LAMBDA=10) 

# -----------------------
# Training the model
trends_train, trends_valid = gan.train(train_gen, valid_gen, 200)


    
    
