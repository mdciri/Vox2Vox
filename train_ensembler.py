import numpy as np
import tensorflow as tf

from models import *
from losses import *

class_weights = np.load('class_weights.npy')

# Optimizers
ensembler_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

imShape = (128, 128, 128, 40)
gtShape = (128, 128, 128, 4)
E = ensembler(imShape, gtShape, class_weights, kernel_size=3)                 

# Load the Vox2Vox models

gen1 = Generator()
gen1.load_weights('./RESULTS/Generator1.h5')
print('Vox2Vox generator 1 loaded.')

gen2 = Generator()
gen2.load_weights('./RESULTS/Generator2.h5')
print('Vox2Vox generator 2 loaded.')

gen3 = Generator()
gen3.load_weights('./RESULTS/Generator3.h5')
print('Vox2Vox generator 3 loaded.')

gen4 = Generator()
gen4.load_weights('./RESULTS/Generator4.h5')
print('Vox2Vox generator 4 loaded.')

gen5 = Generator()
gen5.load_weights('./RESULTS/Generator5.h5')
print('Vox2Vox generator 5 loaded.')

gen6 = Generator()
gen6.load_weights('./RESULTS/Generator6.h5')
print('Vox2Vox generator 6 loaded.')

gen7 = Generator()
gen7.load_weights('./RESULTS/Generator7.h5')
print('Vox2Vox generator 7 loaded.')

gen8 = Generator()
gen8.load_weights('./RESULTS/Generator8.h5')
print('Vox2Vox generator 8 loaded.')

gen9 = Generator()
gen9.load_weights('./RESULTS/Generator9.h5')
print('Vox2Vox generator 9 loaded.')

gen10 = Generator()
gen10.load_weights('./RESULTS/Generator10.h5')
print('Vox2Vox generator 10 loaded.')

# ## Training

def get_scores(X, y):

    SCORES = np.empty((X.shape[0], 128, 128, 128, 40))
    
    y_pred_1 = gen1.predict(X)
    SCORES[:,:,:,:,0:4] = y_pred_1
    del(y_pred_1)
    
    y_pred_2 = gen2.predict(X)
    SCORES[:,:,:,:,4:8] = y_pred_2
    del(y_pred_2)
    
    y_pred_3 = gen3.predict(X)
    SCORES[:,:,:,:,8:12] = y_pred_3
    del(y_pred_3)
    
    y_pred_4 = gen4.predict(X)
    SCORES[:,:,:,:,12:16] = y_pred_4
    del(y_pred_4)
    
    y_pred_5 = gen5.predict(X)
    SCORES[:,:,:,:,16:20] = y_pred_5
    del(y_pred_5)
    
    y_pred_6 = gen6.predict(X)
    SCORES[:,:,:,:,20:24] = y_pred_6
    del(y_pred_6)
    
    y_pred_7 = gen7.predict(X)
    SCORES[:,:,:,:,24:28] = y_pred_7
    del(y_pred_7)
    
    y_pred_8 = gen8.predict(X)
    SCORES[:,:,:,:,28:32] = y_pred_8
    del(y_pred_8)
    
    y_pred_9 = gen9.predict(X)
    SCORES[:,:,:,:,32:36] = y_pred_9
    del(y_pred_9)
    
    y_pred_10 = gen10.predict(X)
    SCORES[:,:,:,:,36:40] = y_pred_10
    
    del(X, y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5, y_pred_6, y_pred_7, y_pred_8, y_pred_9, y_pred_10)
    
    # pre-proc zero-center
    SCORES -= 0.5
    
    return SCORES
    

@tf.function
def train_step(image, target):
    with tf.GradientTape() as ens_tape:
       
        scores = get_scores(image, target)
        ens_output = E(scores, training=True)
        dice_loss =  diceLoss(target, ens_output, class_weights)

    ensembler_gradients = ens_tape.gradient(dice_loss, E.trainable_variables)
    ensembler_optimizer.apply_gradients(zip(ensembler_gradients, E.trainable_variables))
        
    return dice_loss

@tf.function
def test_step(image, target):
       
    scores = get_scores(image, target)
    ens_output = E(scores, training=False)
    dice_loss =  diceLoss(target, ens_output, class_weights)
        
    return dice_loss

def fit(train_gen, valid_gen, epochs):
    
    path = './RESULTS' 
    if os.path.exists(path)==False:
        os.mkdir(path)
        
    Nt = len(train_gen)
    
    prev_loss = np.inf
    epoch_dice_loss = tf.keras.metrics.Mean()
    epoch_dice_loss_val = tf.keras.metrics.Mean()
    
    for e in range(epochs):
        print('Epoch {}/{}'.format(e+1,epochs))
        b = 0
        for Xb, yb in train_gen:
            b += 1
            loss = train_step(Xb, yb)
            epoch_dice_loss.update_state(loss)
            stdout.write('\rBatch: {}/{} - dice_loss: {:.4f}'.format(b, Nt, epoch_dice_loss.result()))
            stdout.flush()
            
        for Xb, yb in valid_gen:
            loss_val = test_step(Xb, yb)
            epoch_dice_loss_val.update_state(loss_val)
        stdout.write('\n               dice_loss_val: {:.4f}'.format(epoch_dice_loss_val.result()))
        stdout.flush()
        
        # save models
        print(' ')
        if epoch_dice_loss_val.result() < prev_loss:    
            E.save_weights(path + '/Ensembler.h5') 
            print("Validation loss decresaed from {:.4f} to {:.4f}. Models' weights are now saved."
                  .format(prev_loss, epoch_dice_loss_val.result()))
            prev_loss = epoch_dice_loss_val.result()
        else:
            print("Validation loss did not decrese from {:.4f}.".format(prev_loss))
        print(' ')
        
        # reset losses state
        epoch_dice_loss.reset_states()
        epoch_dice_loss_val.reset_states()
        
        del Xb, yb
