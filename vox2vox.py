import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, ZeroPadding3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, Loss
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

from sklearn.model_selection import KFold
tf.keras.backend.set_floatx('float32')

class vox2vox():
    def __init__(self, img_shape, tar_shape, Nclasses, model_name=['myGenerator.h5', 'myDiscriminator.h5'], Nfilter_start=64, depth=4, batch_size=3, LAMBDA=100):
        self.img_shape = img_shape
        self.tar_shape = tar_shape
        self.Nclasses = Nclasses
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size
        self.model_name = model_name
        self.LAMBDA = LAMBDA
        
        # define generator and discriminator
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        
        # define optimizers
        self.gen_optimizer = Adam(2e-4, beta_1=0.5)
        self.disc_optimizer = Adam(2e-4, beta_1=0.5)
        
    def GeneratorLoss(self, y_true, y_pred):
        msa = tf.reduce_mean(tf.abs(y_true - y_pred)) # mean absolute error
        # rmse = tf.math.sqrt(tf.reduce_mean(tf.math.square(y_true - y_pred)))

        return msa
    
    def DiscriminatorLoss(self, disc_real_pred, disc_gen_pred):
        disc_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(disc_real_pred), disc_real_pred)
        disc_gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(disc_gen_pred), disc_gen_pred)
        disc_loss = tf.reduce_mean(disc_real_loss + disc_gen_loss)
        
        return disc_loss
    
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

        def decoder_step(layer, layer_to_concatenate, Nf, drop=True):
            x = Conv3DTranspose(Nf, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(layer)
            if drop:
                x = Dropout(0.2)(x)
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
        x = encoder_step(x, self.Nfilter_start*np.power(2,self.depth-1))
        layers_to_concatenate.append(x)
        x = encoder_step(x, self.Nfilter_start*np.power(2,self.depth-1))
        layers_to_concatenate.append(x)
        x = encoder_step(x, self.Nfilter_start*np.power(2,self.depth-1))
        layers_to_concatenate.append(x)
        x = encoder_step(x, self.Nfilter_start*np.power(2,self.depth-1))
        #layers_to_concatenate.append(x)
        x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,self.depth-1))
        x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,self.depth-1))
        x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,self.depth-1))

        # decoder
        for d in range(self.depth-2, -1, -1): 
            x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,d), False)

        # classifier
        last = Conv3DTranspose(1, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', activation='tanh', name='output_generator')(x)

       # Create model
        return Model(inputs=inputs, outputs=last, name='Generator')

    def Discriminator(self):
        '''
        Discriminator model
        '''
        
        inputs = Input(self.img_shape, name='input_image')
        targets = Input(self.tar_shape, name='target_image')

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
                if d == self.depth-1:
                    x = ZeroPadding3D()(x)
                    x = encoder_step(x, self.Nfilter_start*np.power(2,d))             
                else:
                    x = encoder_step(x, self.Nfilter_start*np.power(2,d))
        
        x = ZeroPadding3D()(x)
        last = tf.keras.layers.Conv3D(1, 4, strides=1, kernel_initializer='he_normal', name='output_discriminator')(x) 

        return Model(inputs=[inputs, targets], outputs=last, name='Discriminator')
    
    def train_step(self, Xbatch, Ybatch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator prediction
            gen_pred = self.generator(Xbatch, training=True)

            # Discriminator predictions
            disc_real_pred = self.discriminator([Xbatch, Ybatch], training=True)   
            disc_gen_pred = self.discriminator([Xbatch, gen_pred], training=True) 

            # Discriminator loss
            disc_loss = self.DiscriminatorLoss(disc_real_pred, disc_gen_pred)

            # Generator loss
            gen_loss = self.GeneratorLoss(Ybatch, gen_pred)

            # Total Generator loss
            gen_gan_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(disc_gen_pred), disc_gen_pred))
            total_gen_loss = gen_gan_loss+ self.LAMBDA * gen_loss
        
        # calculating the gradiends
        gen_grads = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # apply gradients to the optimizers
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return total_gen_loss, gen_gan_loss, gen_loss, disc_loss
    
    def valid_step(self, Xbatch, Ybatch):
        # Generator prediction
        gen_pred = self.generator(Xbatch, training=False)
        
        # Discriminator predictions
        disc_real_pred = self.discriminator([Xbatch, Ybatch], training=False)
        disc_gen_pred = self.discriminator([Xbatch, gen_pred], training=False) 
        
        # Discriminator loss        
        disc_loss = self.DiscriminatorLoss(disc_real_pred, disc_gen_pred)

        # Generator loss
        iou_loss = self.GeneratorLoss(Ybatch, gen_pred)

        # Total Generator loss
        gen_gan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(disc_gen_pred), disc_gen_pred))
        total_gen_loss = gen_gan_loss+ self.LAMBDA * gen_loss
        
        return total_gen_loss, gen_gan_loss, iou_loss, disc_loss   
    
    def fit(self, Xtrain, Ytrain, Xvalid, Yvalid, nEpochs):
        print('Training process:')
        print('Training on {} images and validating on {} images.\n'.format(Xtrain.shape[0], Xvalid.shape[0]))
        # we save in a dictionary the histories obtained after each epoch
        history = {}
        history['total_gen_loss'] = []
        history['gen_gan_loss'] = []
        history['gen_loss'] = []
        history['disc_loss'] = []
        
        Nbatches_train = int(np.ceil(Xtrain.shape[0]/self.batch_size)) # number of batches in the training set
        Nbatches_valid = int(np.ceil(Xvalid.shape[0]/self.batch_size)) # number of batches in the validation set
        valid_loss_prev = np.inf      
        
        for e in range(nEpochs):
            print('Epoch {}/{}'.format(e+1,nEpochs))
            start_time = time.time()
            kf = KFold(n_splits=Nbatches_train, shuffle=True)
            b = 0
            for _, batch_index in kf.split(Xtrain):
                Xbatch, Ybatch = Xtrain[batch_index], Ytrain[batch_index]
                train_losses = self.train_step(Xbatch, Ybatch)
                b += len(batch_index)
                stdout.write('\rBatch: {}/{} - gan_loss: {:.4f} - gen_gan_loss: {:.4f} - rmse: {:.4f} - disc_loss: {:.4f}'.format(b, Xtrain.shape[0], train_losses[0], train_losses[1], train_losses[2], train_losses[3]))
                stdout.flush()
                
            kf = KFold(n_splits=Nbatches_valid, shuffle=True)
            for _, batch_index in kf.split(Xvalid):
                Xbatch, Ybatch = Xvalid[batch_index], Yvalid[batch_index]
                valid_losses = self.valid_step(Xbatch, Ybatch)
         
            stdout.write(' - gan_loss_valid: {:.4f}'.format(valid_losses[0]))
            elapsed_time = datetime.timedelta(seconds=time.time() - start_time)
            stdout.write('\nElapsed time: {} h:mm:ss'.format(elapsed_time))
            stdout.flush()
                
            # saving the loss values
            history['total_gen_loss'].append([train_losses[0], valid_losses[0]])
            history['gen_gan_loss'].append([train_losses[1], valid_losses[1]])
            history['gen_loss'].append([train_losses[2], valid_losses[2]])
            history['disc_loss'].append([train_losses[3], valid_losses[3]])
      
            
            if valid_losses[0]<valid_loss_prev:
                self.generator.save_weights(self.model_name[0])
                self.discriminator.save_weights(self.model_name[1])
                print('\nval_loss decreased from {:.4f} to {:.4f}. Hence, the models are now updated and saved.'.format(valid_loss_prev, valid_losses[0]))
                valid_loss_prev = valid_losses[0]
            else:
                print('\nval_loss did not decreased.')
            print('\n ')
            
            # printing a model predition every 3 epochs
            if (e + 1) % 1 == 0:
                pred = self.generator(Xbatch, training=True)

                fig = plt.figure(10, figsize=(50,10))
                plt.subplot(1, 4, 1)  
                plt.imshow(Xbatch[0,:,:,Xvalid.shape[3]//2,0]*0.5+0.5, cmap='gray') # pixel values between [0, 1]
                plt.title('T1ce MR image')
                plt.axis('off')
                plt.subplot(1, 4, 2)  
                plt.imshow(Ybatch[0,:,:,Yvalid.shape[3]//2,0]*0.5+0.5, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                plt.subplot(1, 4, 3)  
                plt.imshow(pred[0,:,:,pred.shape[3]//2,0]*0.5+0.5, cmap='gray')
                plt.title('Prediction')
                plt.axis('off')
                plt.subplot(1, 4, 4) 
                plt.hist(0.5+0.5*pred[0,:,:,pred.shape[3]//2,0].ravel(), bins=256, range=(0.0, 1.0))
                plt.title('Prediction histogram')
                plt.show()
                
                fig.savefig('./Results_IoU/Prediction@Epoch_{}.png'.format(e+1))
                
        return history
        
imShape = Xtrain_patches.shape[1:]
gtShape = Ytrain_patches.shape[1:]
gan = vox2vox(imShape, gtShape, Nclasses, batch_size=3)   
