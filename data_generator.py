class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=4, dim=(153, 182, 144), n_channels=4, n_classes=7, shuffle=True, augmentation=False, patch_size=64, n_patches=8):
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
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        
        X, y = self.__data_generation(list_IDs_temp)
        if self.augmentation == True:
            X, y = self.__data_3Dpatch_augmentation(X, y)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            # Store sample
            X[i], y[i] = load_img(IDs)
        
        return X, to_categorical(y, self.n_classes)

    def __data_3Dpatch_augmentation(self, X, Y):
        X_patches, y_patches = [], []
 
        for Nim in range(self.batch_size):
            for Np in range(self.n_patches):
                x = np.random.randint(self.dim[0]-self.patch_size) 
                y = np.random.randint(self.dim[1]-self.patch_size)
                z = np.random.randint(self.dim[2]-self.patch_size)
            
                X_patches.append(X[Nim, x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size, :])
                y_patches.append(Y[Nim, x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size, :])
                
        return np.asarray(X_patches), np.asarray(y_patches)
