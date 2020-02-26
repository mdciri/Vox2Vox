def load_img(img_files):
    ''' Load one image and its target form file
    '''
    X = np.zeros((240, 240, 155, 4))
    X[:,:,:,0] = nib.load(img_files[0]).get_fdata(dtype='float32')
    X[:,:,:,1] = nib.load(img_files[1]).get_fdata(dtype='float32')
    X[:,:,:,2] = nib.load(img_files[2]).get_fdata(dtype='float32')
    X[:,:,:,3] = nib.load(img_files[3]).get_fdata(dtype='float32')
    # image normalization
    X = X[43:196,39:221,0:144,:]
    X = X-np.mean(X, axis=(0,1,2))/(np.std(X, axis=(0,1,2))+1e-8)
    X = 2*(X-np.amin(X, axis=(0,1,2)))/(np.amax(X, axis=(0,1,2))-np.amin(X, axis=(0,1,2)))-1
    # target
    y = nib.load(img_files[4]).get_fdata(dtype='float32')
    y = y[43:196,39:221,0:144]
    
    return X, y
