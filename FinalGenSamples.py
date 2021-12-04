# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
print(torch.version.cuda) #10.1
import time
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

t0 = time.time()
##############################################################################
"""args for models"""

args = {}
args['dim_h'] = 64          # factor controlling size of hidden layers
args['n_channel'] = 3       # number of channels in the input data 

args['n_z'] = 300 #600     # number of dimensions in latent space. 

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.1      # hyper param for weight of discriminator loss
args['lr'] = 0.01        # learning rate for Adam optimizer .000
args['epochs'] = 1 #50         # how many epochs to run for
args['batch_size'] = 32   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from

args['dataset'] = 'bridges' #'fmnist' # specify which dataset to use

##############################################################################

## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),#32,64, 112, 112
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),#32,128,56,56
            nn.BatchNorm2d(self.dim_h * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),#32, 256, 28, 28
            nn.BatchNorm2d(self.dim_h * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 16, 16, 1, bias=False),#32, 512, 1, 1
        
            nn.BatchNorm2d(self.dim_h * 8),
            #nn.ReLU(True))
            nn.LeakyReLU(0.2, inplace=True))
            
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)
        

    def forward(self, x):

        x = self.conv(x)

        x = x.squeeze()
        print('aft squeeze ',x.size())

        x = self.fc(x)

        return x
    
    
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, kernel_size = 6, stride = 4, padding = 1, bias=False),#32, 256, 28, 28
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, kernel_size = 4, stride = 2, padding = 1, bias=False),#32,128,56,56
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 3, kernel_size = 6, stride = 4, padding = 1, bias=False),#32,64, 112, 112
            nn.Sigmoid())
            #nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        print('decodedex ', x.size())
        return x

##############################################################################

def biased_get_class1(c):
    
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    
    return xbeg, ybeg
    #return xclass, yclass


def G_SM1(X, y,n_to_sample,cl):

    
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)
    
    print('Samples: ')

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

#############################################################################
np.printoptions(precision=5,suppress=True)

TRAIN_DATA_DIR = './New_Training_Data_unbalanced'

datagen = ImageDataGenerator()

train_generator = datagen.flow_from_directory(TRAIN_DATA_DIR,
                                              target_size = (224,224),
                                              batch_size= 1489,
                                              classes = ['Crack', 'Environment', 'Joint Defect', 'Loss of Section', 'No Defect Wall', 'Spalling', 'Vegetation'],
                                              class_mode='sparse', 
                                              shuffle=False,
                                              interpolation='nearest')

#Use the next iterator
X_train, y_train = next(train_generator)
X_train = X_train / 255.
X_train = X_train.transpose(0,3,1,2)

encf = []
decf = []


enc ='./bst_enc_new_dataset.pth'
dec ='./bst_dec_new_dataset.pth'
encf.append(enc)
decf.append(dec)


dec_x = X_train
dec_y = y_train


print('train labels ',dec_y.shape) #(44993,) (45500,)

print(collections.Counter(dec_y))

print('train imgs after reshape ',dec_x.shape) #(45000,3,32,32)

#classes = ('Crack', 'Environment', 'Joint Defect', 'Loss of Section', 'No Defect Wall', 'Spalling', 'Vegetation')

#generate some images 
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_enc = encf[0]
path_dec = decf[0]

encoder = Encoder(args)
encoder.load_state_dict(torch.load(path_enc), strict=False)
encoder = encoder.to(device)

decoder = Decoder(args)
decoder.load_state_dict(torch.load(path_dec), strict=False)
decoder = decoder.to(device)

encoder.eval()
decoder.eval()

#imbal = [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80]
imbal = [80, 300, 209, 18, 300, 285, 297]

resx = []
resy = []

for i in range(0,7):
    xclass, yclass = biased_get_class1(i)
    print(xclass.shape) #(500, 3, 32, 32)
    print(yclass[0]) #(500,)
        
    #encode xclass to feature space
    xclass = torch.Tensor(xclass)
    xclass = xclass.to(device)
    xclass = encoder(xclass)
    print(xclass.shape) #torch.Size([500, 600])
        
    xclass = xclass.detach().cpu().numpy()
    n = imbal[1] - imbal[i]
    xsamp, ysamp = G_SM1(xclass,yclass,n,i)
    print(xsamp.shape) #(4500, 600)
    print(len(ysamp)) #4500
    ysamp = np.array(ysamp)
    print(ysamp.shape) #4500   

    """to generate samples for resnet"""   
    xsamp = torch.Tensor(xsamp)
    xsamp = xsamp.to(device)
    #xsamp = xsamp.view(xsamp.size()[0], xsamp.size()[1], 1, 1)
    #print(xsamp.size()) #torch.Size([10, 600, 1, 1])
    ximg = decoder(xsamp)
    #print("shape", ximg.Size())

    #ximg = torch.permute(ximg,(0, 2, 3, 1))
    ximn = ximg.detach().cpu().numpy()

    print(ximn.shape) #(4500, 3, 32, 32)
    #ximn = np.expand_dims(ximn,axis=1)
    print(ximn.shape) #(4500, 3, 32, 32)
    resx.append(ximn)
    resy.append(ysamp)
    #print('resx ',resx.shape)
    #print('resy ',resy.shape)
    #print()

resx1 = np.vstack(resx)
resy1 = np.hstack(resy)

print(resy1.shape) #(34720,)

resx1 = resx1.reshape(resx1.shape[0],-1)
print(resx1.shape) #(34720, 3072)

dec_x1 = dec_x.reshape(dec_x.shape[0],-1)
print('decx1 ',dec_x1.shape)
combx = np.vstack((resx1,dec_x1))
comby = np.hstack((resy1,dec_y))
combx = combx.reshape(combx.shape[0], 3, 224, 224)
combx = combx.transpose(0,2,3,1)
combx = combx * 255.

for i in range(0, 2100):
    image = combx[i].astype('uint8')

    plt.imshow(image)
    plt.axis('off')
    
    if comby[i] == 0:
        plt.imsave('./DeepSmoteGeneratedImages/Crack' + '\\' + str(i) + '.jpg', image)
    elif comby[i] == 1:
        plt.imsave('./DeepSmoteGeneratedImages/Environment' + '\\' + str(i) + '.jpg', image)
        #pyplot.imsave('./IsProcessed/Joint Defect' + '\\' + str(i) + '.jpg', image)
    elif comby[i] == 2:
        plt.imsave('./DeepSmoteGeneratedImages/Joint Defect' + '\\' + str(i) + '.jpg', image)
        #pyplot.imsave('./IsProcessed/Loss of Section' + '\\' + str(i) + '.jpg', image)
    elif comby[i] == 3:
        plt.imsave('./DeepSmoteGeneratedImages/Loss of Section' + '\\' + str(i) + '.jpg', image)
        #pyplot.imsave('./IsProcessed/Spalling' + '\\' + str(i) + '.jpg', image)
    elif comby[i] == 4:
        plt.imsave('./DeepSmoteGeneratedImages/No Defect Wall' + '\\' + str(i) + '.jpg', image)
    elif comby[i] == 5:
        plt.imsave('./DeepSmoteGeneratedImages/Spalling' + '\\' + str(i) + '.jpg', image)
    elif comby[i] == 6:
        plt.imsave('./DeepSmoteGeneratedImages/Vegetation' + '\\' + str(i) + '.jpg', image)
    
    plt.show()

t1 = time.time()
print('final time(min): {:.2f}'.format((t1 - t0)/60))