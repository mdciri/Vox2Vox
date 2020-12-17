import os
import numpy as np
import tensorflow as tf
from models import *
from losses import *
import matplotlib.image as mpim
from sys import stdout


# class weights
class_weights = np.load('class_weights.npy')

# Models
G = Generator()
D = Discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = G(image, training=True)

        disc_real_output = D([image, target], training=True)
        disc_fake_output = D([image, gen_output], training=True)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
        
        gen_loss, dice_loss, disc_loss_gen = generator_loss(target, gen_output, disc_fake_output, class_weights)

    generator_gradients = gen_tape.gradient(gen_loss, G.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, D.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D.trainable_variables))
        
    return gen_loss, dice_loss, disc_loss_gen
        
@tf.function
def test_step(image, target):
    gen_output = G(image, training=False)

    disc_real_output = D([image, target], training=False)
    disc_fake_output = D([image, gen_output], training=False)
    disc_loss = discriminator_loss(disc_real_output, disc_fake_output)

    gen_loss, dice_loss, disc_loss_gen = generator_loss(target, gen_output, disc_fake_output, class_weights)
        
    return gen_loss, dice_loss, disc_loss_gen

def fit(train_gen, valid_gen, epochs):
    
    path = './RESULTS' 
    if os.path.exists(path)==False:
        os.mkdir(path)
        
    Nt = len(train_gen)
    history = {'train': [], 'valid': []}
    prev_loss = np.inf
    
    for e in range(epochs):
        print('Epoch {}/{}'.format(e+1,epochs))
        b = 0
        for Xb, yb in train_gen:
            b += 1
            losses = train_step(Xb, yb)
            stdout.write('\rBatch: {}/{} - loss: {:.4f} - dice_loss: {:.4f} - disc_loss: {:.4f}: {:.4f}'
                         .format(b, Nt, losses[0], losses[1], losses[2]))
            stdout.flush()
        history['train'].append(losses)
        
        for Xb, yb in valid_gen:
            losses_val = test_step(Xb, yb)
        stdout.write('\n               loss_val: {:.4f} - dice_loss_val: {:.4f} - disc_loss_val: {:.4f}'
                     .format(losses_val[0], losses_val[1], losses_val[2]))
        stdout.flush()
        history['valid'].append(losses_val)
        
        # save pred image at epoch e 
        y_pred = G.predict(Xb)
        y_true = np.argmax(yb, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        canvas = np.zeros((128, 128*3))
        idx = np.random.randint(len(Xb))
        
        x = Xb[idx,:,:,64,2] 
        canvas[0:128, 0:128] = (x - np.min(x))/(np.max(x)-np.min(x)+1e-6)
        canvas[0:128, 128:2*128] = y_true[idx,:,:,64]/3
        canvas[0:128, 2*128:3*128] = y_pred[idx,:,:,64]/3
        
        fname = (path + '/pred@epoch_{:03d}.png').format(e+1)
        mpim.imsave(fname, canvas, cmap='gray')
        
        # save models
        print(' ')
        if losses_val[0] < prev_loss:    
            G.save_weights(path + '/Generator.h5') 
            D.save_weights(path + '/Discriminator.h5')
            print("Validation loss decresaed from {:.4f} to {:.4f}. Models' weights are now saved.".format(prev_loss, losses_val[0]))
            prev_loss = losses_val[0]
        else:
            print("Validation loss did not decrese from {:.4f}.".format(prev_loss))
        print(' ')
        
        del Xb, yb, canvas, y_pred, y_true, idx
        
    return history
