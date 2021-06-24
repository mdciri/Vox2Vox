# Vox2Vox: 3D-GAN for Brain Tumour Segmentation

Vox2Vox is a 3D-GAN, inpired my the Pix2Pix GAN by Isola, to do brain tumor segmentation.

The paper about the Vox2Vox model is published in the [6th International Workshop, BrainLes 2020, Held in Conjunction with MICCAI 2020](https://link.springer.com/book/10.1007%2F978-3-030-72084-1). You can also find it in [arXiv](https://arxiv.org/abs/2003.13653).

The data used are the ones provided by the [Brain Tumour Segmentation (BraTS) Challenge 2020](https://www.med.upenn.edu/cbica/brats2020/data.html).

The model is created and the training is performed using Tensorflow.

Requirements:
- elasticdeform             0.4.6    
- matplotlib                3.3.2   
- numpy                     1.19.2   
- numpy-base                1.19.2          
- python                    3.8.5  
- scikit-learn              0.23.2 
- scipy                     1.5.2 
- tensorflow                2.2.0   
- tensorflow-addons         0.11.2  
- tensorflow-gpu            2.2.0 


to train the model just type in your terminal the command: 

    python ./main.py
  
You can alos add these parameter in the command in case you would like to change some parameters:

'-g' or '--gpu': GPU position, defaul 0

'-nc' or '--num_classes': number of classes, default 4

'-bs' or '--batch_size': batch size, default 4

'-a' or '--alpha': alpha weight between the generator and the discriminator, default 5

'-ne' or '--num_epochs': number of epochs, default 200
