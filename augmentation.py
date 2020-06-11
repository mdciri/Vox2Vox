import numpy as np
from scipy.ndimage.interpolation import affine_transform
import elasticdeform
from scipy import stats
import multiprocessing as mp

def patch_extraction(Xb, yb, sizePatches=128, Npatches=1):
    """
    3D patch extraction
    """
    
    batch_size, rows, columns, slices, channels = Xb.shape
    X_patches = np.empty((batch_size*Npatches, sizePatches, sizePatches, sizePatches, channels))
    y_patches = np.empty((batch_size*Npatches, sizePatches, sizePatches, sizePatches))
    i = 0
    for b in range(batch_size):
        for p in range(Npatches):
            x = np.random.randint(rows-sizePatches+1) 
            y = np.random.randint(columns-sizePatches+1)
            z = np.random.randint(slices-sizePatches+1) 

            X_patches[i] = Xb[b, x:x+sizePatches, y:y+sizePatches, z:z+sizePatches, :]
            y_patches[i] = yb[b, x:x+sizePatches, y:y+sizePatches, z:z+sizePatches]
            i += 1
                               
    return X_patches, y_patches

def flip3D(X, y):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
    choice = np.random.randint(3)
    if choice == 0: # flip on x
        X_flip, y_flip = X[::-1, :, :, :], y[::-1, :, :]
    if choice == 1: # flip on y
        X_flip, y_flip = X[:, ::-1, :, :], y[:, ::-1, :]
    if choice == 2: # flip on z
        X_flip, y_flip = X[:, :, ::-1, :], y[:, :, ::-1]
        
    return X_flip, y_flip


def rotation3D(X, y):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    alpha, beta, gamma = np.random.randint(0, 31, size=3)/180*np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    R = np.dot(np.dot(Rx, Ry), Rz)
    
    X_rot = np.empty_like(X)
    for channel in range(X.shape[-1]):
        X_rot[:,:,:,channel] = affine_transform(X[:,:,:,channel], R, offset=0, order=3, mode='constant')
    y_rot = affine_transform(y, R, offset=0, order=0, mode='constant')
    
    return X_rot, y_rot

def brightness(X, y):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.
    
    Gain chosen between [0.9 - 1.1]
    Gamma chosen between [0.4 - 2.5]
    
    new_im = gain * im^gamma
    """
    
    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:,:,:,c] - np.min(X[:,:,:,c]) # all positive values
        
        choice = np.random.random_sample((2,))
        gain = (1.1 - 0.9) * np.random.random_sample() + 0.9
        gamma = (2.5-1)*choice[0] + 1
        
        if choice[1]<0.5:
            gamma = 1/gamma
        
        im_new = gain*(im**gamma)
        
        # the mode of new_im is where the background lie, so we must set again these values to 0
        X_new[:,:,:,c] = im_new - stats.mode(im_new, axis=None)[0][0]
    
    return X_new, y

def elastic(X, y):
    """
    Elastic deformation on a image and its target
    """
    
    [Xel, yel] = elasticdeform.deform_random_grid([X, y], sigma=5, axis=[(0, 1, 2), (0, 1, 2)], order=[3, 0])
    
    return Xel, yel

def random_decisions(N):
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """
    
    decisions = np.zeros((N, 4)) # 4 is number of aug techniques to combine
    for n in range(N):
        decisions[n] = np.random.randint(2, size=4)
        
    return decisions

def combine_aug(X, y, do):
    """
    Combine randomly the different augmentation techniques written above
    """
    Xnew, ynew = X, y
    
    if do[0] == 1:
        Xnew, ynew = flip3D(Xnew, ynew)
        
    if do[1] == 1:
        Xnew, ynew = brightness(Xnew, ynew)   
        
    if do[2] == 1:
        Xnew, ynew = rotation3D(Xnew, ynew)
        
    if do[3] == 1:
        Xnew, ynew = elastic(Xnew, ynew)
        
    return Xnew, ynew

def aug_batch(Xb, Yb):
    """
    Generate a augmented image batch 
    """
    batch_size = len(Xb)
    newXb, newYb = np.empty_like(Xb), np.empty_like(Yb)
    
    decisions = random_decisions(batch_size)
    
    inputs = [(X, y, do) for X, y, do in zip(Xb, Yb, decisions)]
    pool = mp.Pool(processes=8)
    multi_result = pool.starmap(combine_aug, inputs)
    pool.close()
    
    for i in range(len(Xb)):
        newXb[i], newYb[i] = multi_result[i][0], multi_result[i][1]
        
    return newXb, newYb  