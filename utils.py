import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from augmentation import *

def load_img(img_files):
    ''' Load one image and its target form file
    '''
    N = len(img_files)
    # target
    y = nib.load(img_files[N-1]).get_fdata(dtype='float32', caching='unchanged')
    y = y[40:200,34:226,8:136]
    y[y==4]=3
      
    X_norm = np.empty((240, 240, 155, 4))
    for channel in range(N-1):
        X = nib.load(img_files[channel]).get_fdata(dtype='float32', caching='unchanged')
        brain = X[X!=0] 
        brain_norm = np.zeros_like(X) # background at -100
        norm = (brain - np.mean(brain))/np.std(brain)
        brain_norm[X!=0] = norm
        X_norm[:,:,:,channel] = brain_norm        
        
    X_norm = X_norm[40:200,34:226,8:136,:]    
    del(X, brain, brain_norm)
    
    return X_norm, y
    
def visualize(X):
    """
    Visualize the image middle slices for each axis
    """
    a,b,c = X.shape
    
    plt.figure(figsize=(15,15))
    plt.subplot(131)
    plt.imshow(np.rot90(X[a//2, :, :]), cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(np.rot90(X[:, b//2, :]), cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(X[:, :, c//2], cmap='gray')
    plt.axis('off')
    
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=4, dim=(160,192,128), n_channels=4, n_classes=4, shuffle=True, augmentation=False, patch_size=128, n_patches=1):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data     
        X, y = self.__data_generation(list_IDs_temp)
        if self.augmentation == True:
            X, y = self.__data_augmentation(X, y)
        
        if index == self.__len__()-1:
            self.on_epoch_end()
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            # Store sample
            X[i], y[i] = load_img(IDs)
            
        if self.augmentation == True:
            return X.astype('float32'), y
        else:
            return X.astype('float32'), to_categorical(y, self.n_classes)

    def __data_augmentation(self, X, y):
        'Apply augmentation'
        X_aug, y_aug = patch_extraction(X, y, sizePatches=self.patch_size, Npatches=self.n_patches)
        X_aug, y_aug = combine_aug(X_aug, y_aug)        
                
        return X_aug, to_categorical(y_aug, self.n_classes)
