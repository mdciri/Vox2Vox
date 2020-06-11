import numpy as np
import tensorflow as tf

import time
from tensorflow.keras.utils import to_categorical
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib.image as mpim

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

class vox2vox():
    def __init__(self, img_shape, seg_shape, class_weights, Nfilter_start=64, depth=4, batch_size=3, LAMBDA=5):
        self.img_shape = img_shape
        self.seg_shape = seg_shape
        self.class_weights = class_weights
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size
        self.LAMBDA = LAMBDA
        
        def diceLoss(y_true, y_pred, w=self.class_weights):
            y_true = tf.convert_to_tensor(y_true, 'float32')
            y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)

            num = tf.math.reduce_sum(tf.math.multiply(w, tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0,1,2,3])))
            den = tf.math.reduce_sum(tf.math.multiply(w, tf.math.reduce_sum(tf.math.add(y_true, y_pred), axis=[0,1,2,3])))+1e-5

            return 1-2*num/den

        # Build and compile the discriminator
        self.discriminator = self.Discriminator()
        self.discriminator.compile(loss='mse', optimizer=Adam(2e-4, beta_1=0.5), metrics=['accuracy'])

        # Construct Computational Graph of Generator
        # Build the generator
        self.generator = self.Generator()

        # Input images and their conditioning images
        seg = Input(shape=self.seg_shape)
        img = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        seg_pred = self.generator(img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([seg_pred, img])

        self.combined = Model(inputs=[seg, img], outputs=[valid, seg_pred])
        self.combined.compile(loss=['mse', diceLoss], loss_weights=[1, self.LAMBDA], optimizer=Adam(2e-4, beta_1=0.5))
    
    def Generator(self):
        '''
        Generator model
        '''

        inputs = Input(self.img_shape, name='input_image')     

        def encoder_step(layer, Nf, inorm=True):
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            if inorm:
                x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            
            return x
        
        def bottlenek(layer, Nf):
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            for i in range(4):
                y = Conv3D(Nf, kernel_size=4, strides=1, kernel_initializer='he_normal', padding='same')(x)
                x = InstanceNormalization()(y)
                x = Dropout(0.2)(x)
                x = LeakyReLU()(x)
                x = Concatenate()([x, y])
                
            return x

        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv3DTranspose(Nf, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(layer)
            x = InstanceNormalization()(x)
            x = ReLU()(x)
            x = Concatenate()([x, layer_to_concatenate])
            return x

        layers_to_concatenate = []
        x = inputs

        # encoder
        for d in range(self.depth-1):
            if d==0:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d), False)
            else:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(x)
        
        # bottlenek
        x = bottlenek(x, self.Nfilter_start*np.power(2,self.depth-1))

        # decoder
        for d in range(self.depth-2, -1, -1): 
            x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,d))

        # classifier
        last = Conv3DTranspose(self.seg_shape[-1], kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', activation='softmax', name='output_generator')(x)

       # Create model
        return Model(inputs=inputs, outputs=last, name='Generator')

    def Discriminator(self):
        '''
        Discriminator model
        '''
        
        inputs = Input(self.img_shape, name='input_image')
        targets = Input(self.seg_shape, name='target_image')

        def encoder_step(layer, Nf, inorm=True):
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            if inorm:
                x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            return x

        x = Concatenate()([inputs, targets])

        for d in range(self.depth):
            if d==0:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d), False)
            else:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d))


        last = tf.keras.layers.Conv3D(1, 4, strides=1, padding='same', kernel_initializer='he_normal', name='output_discriminator')(x) 

        return Model(inputs=[targets, inputs], outputs=last, name='Discriminator')
