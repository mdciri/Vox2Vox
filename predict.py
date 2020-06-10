#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tensorflow as tf
import nibabel as nib
import glob

Nclasses = 4
classes = np.arange(Nclasses)

# images lists
t1_list = sorted(glob.glob('/nobackup/data/marci30/BRATS_2020/*/*/*t1.nii'))
t2_list = sorted(glob.glob('/nobackup/data/marci30/BRATS_2020/*/*/*t2.nii'))
t1ce_list = sorted(glob.glob('/nobackup/data/marci30/BRATS_2020/*/*/*t1ce.nii'))
flair_list = sorted(glob.glob('/nobackup/data/marci30/BRATS_2020/*/*/*flair.nii'))
seg_list = sorted(glob.glob('/nobackup/data/marci30/BRATS_2020/*/*/*seg.nii'))

sets = {'train': []}

for i in range(369):
    sets['train'].append([t1_list[i], t2_list[i], t1ce_list[i], flair_list[i], seg_list[i]])
    
def load_img(img_files):
    ''' Load one image and its target form file
    '''
    N = len(img_files)
    # target
    y = nib.load(img_files[N-1])
    
    # get image info
    info = {'affine': y.affine,
            'header': y.header,
            'file_map': y.file_map, 
            'extra': y.extra}
    
    y = y.get_fdata(dtype='float32', caching='unchanged')
    y = y[40:200,34:226,8:136]
    y[y==4]=3
      
    X_norm = np.empty((240, 240, 155, 4))
    for channel in range(N-1):
        X = nib.load(img_files[channel]).get_fdata(dtype='float32', caching='unchanged')
        brain = X[X!=0] 
        brain_norm = np.zeros_like(X)
        norm = (brain - np.mean(brain))/np.std(brain)
        brain_norm[X!=0] = norm
        X_norm[:,:,:,channel] = brain_norm        
        
    X_norm = X_norm[40:200,34:226,8:136,:]    
    del(X, brain, brain_norm)
    
    return X_norm, y, info


## LOAD THE MODEL
from architecture import vox2vox

ALPHA = 5
gan = vox2vox(imShape, gtShape, class_weights, depth=4, batch_size=1, LAMBDA=ALPHA)
gan.generator.load_weights('./RESULTS/alpha_{}/Generator.h5'.format(ALPHA))


## SAVE PREDICTIONS
for i in range(len(t1_list)):
    X, y_true, info = load_img(sets['train'][i])
    
    y_pred = gan.generator.predict(np.expand_dims(X, axis=0))
    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_pred = np.squeeze(y_pred)
    
    # check shape
    if y_pred.shape != (240, 240, 155):
        y_pred_pad = np.zeros((240, 240, 155))
        y_pred_pad[40:200,34:226,8:136] = y_pred
    else:
        y_pred_pad = y_pred
    
    idx = i+1
    if idx<10:
        ID = '00'+str(idx)
    if idx>=10 and idx<100:
        ID = '0'+str(idx)
    if idx>=100:
        ID = str(idx)
    
    nifti_im = nib.Nifti1Image(y_pred_pad, affine=info['affine'], header=info['header'], file_map=info['file_map'], extra=info['extra'])
    
    filename = 'BraTS20_Training_' + ID + '_pred'
    
    nib.save(nifti_im, filename)