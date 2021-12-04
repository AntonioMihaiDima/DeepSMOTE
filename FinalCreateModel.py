import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
print(torch.version.cuda) #10.1
import time

from torch.utils.data import TensorDataset

from keras.preprocessing.image import ImageDataGenerator

t3 = time.time()

##############################################################################
"""args for models"""

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 3       # number of channels in the input data 

args['n_z'] = 300 #600     # number of dimensions in latent space. 

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.1      # hyper param for weight of discriminator loss
args['lr'] = 0.01        # learning rate for Adam optimizer .000
args['epochs'] = 4000 #50         # how many epochs to run for
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
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),#32,64, 112, 11
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
            nn.ConvTranspose2d(self.dim_h * 2, 3, kernel_size = 6, stride = 4, padding = 1, bias=False),#32,3, 224, 224
            nn.Sigmoid())
            #nn.Tanh())

    def forward(self, x):

        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)

        return x
    
##############################################################################
"""set models, loss functions"""
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


##############################################################################
"""functions to create SMOTE images"""

def biased_get_class(c):
    
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    
    return xbeg, ybeg
    #return xclass, yclass


def G_SM(X, y,n_to_sample,cl):

    # determining the number of samples to generate
    #n_to_sample = 10 

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

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample
    
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


encoder = Encoder(args)
decoder = Decoder(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
decoder = decoder.to(device)
encoder = encoder.to(device)

#decoder = decoder.to(memory_format=torch.channels_last)
#encoder = encoder.to(memory_format=torch.channels_last)

#print ("encoder ",encoder.Size())

train_on_gpu = torch.cuda.is_available()

#decoder loss function
criterion = nn.MSELoss()
criterion = criterion.to(device)

dec_x = X_train
dec_y = y_train
print('train labels ',dec_y.shape) 
print(collections.Counter(dec_y))

batch_size = 32
num_workers = 0

#torch.Tensor returns float so if want long then use torch.tensor
tensor_x = torch.Tensor(dec_x)
#tensor_x = tensor_x.to(memory_format=torch.contiguous_format)
# ^ Here use tensor_image.permute(1, 2, 0) 
tensor_y = torch.tensor(dec_y,dtype=torch.long)
mnist_bal = TensorDataset(tensor_x,tensor_y) 
train_loader = torch.utils.data.DataLoader(mnist_bal, 
    batch_size=batch_size,shuffle=True,num_workers=num_workers)

best_loss = np.inf

t0 = time.time()
if args['train']:
    enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])
    dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])

    for epoch in range(args['epochs']):
        train_loss = 0.0
        tmse_loss = 0.0
        tdiscr_loss = 0.0
        # train for one epoch -- set nets to train mode
        encoder.train()
        decoder.train()
    
        for images,labs in train_loader:
        
            # zero gradients for each batch
            encoder.zero_grad()
            decoder.zero_grad()
            #print(images)
            images, labs = images.to(device), labs.to(device)
            #images = images.contiguous(memory_format=torch.contiguous_format)
            #print('images ',images.size()) 
            labsn = labs.detach().cpu().numpy()
            #print('labsn ',labsn.shape, labsn)
        
            # run images
            
            z_hat = encoder(images)
        
            x_hat = decoder(z_hat) #decoder outputs tanh
            #print('xhat ', x_hat.size())
            #print(x_hat)
            mse = criterion(x_hat,images)
            #print('mse ',mse)
            
                   
            resx = []
            resy = []
        
            tc = np.random.choice(7,1)
            
            xbeg = dec_x[dec_y == tc]
            #print(xbeg.shape)
            ybeg = dec_y[dec_y == tc] 
            xlen = len(xbeg)
            nsamp = min(xlen, 100)
            ind = np.random.choice(list(range(len(xbeg))),nsamp,replace=False)
            ind = ind.astype(int)
            #print("Index is", xbeg[ind])
            xclass = xbeg[ind]
            #print(xclass.shape)
            yclass = ybeg[ind]
        
            xclen = len(xclass)
            #print('xclen ',xclen)
            xcminus = np.arange(1,xclen)
            #print('minus ',xcminus.shape,xcminus)
            
            xcplus = np.append(xcminus,0)
            #print('xcplus ',xcplus)
            xcnew = (xclass[[xcplus],:])
            #xcnew = np.squeeze(xcnew)
            xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
            #print('xcnew ',xcnew.shape)
        
            xcnew = torch.Tensor(xcnew)
            xcnew = xcnew.to(device)
        
            #encode xclass to feature space
            xclass = torch.Tensor(xclass)
            xclass = xclass.to(device)
            xclass = encoder(xclass)
            #print('xclass ',xclass.shape) 
        
            xclass = xclass.detach().cpu().numpy()
        
            xc_enc = (xclass[[xcplus],:])
            #xc_enc = np.squeeze(xc_enc)
            #print('xc enc ',xc_enc.shape)
        
            xc_enc = torch.Tensor(xc_enc)
            xc_enc = xc_enc.to(device)
            
            ximg = decoder(xc_enc)
            #ximg = ximg.to(memory_format=torch.channels_last)
            #xcnew = ximg.to(memory_format=torch.channels_last)
            
            #print("images size", ximg.size())
            
            #ximn = ximg.detach().cpu().numpy()
            
            #ximn = ximn.reshape(ximn.shape[0],-1)
            
            #x_hat1 = x_hat.detach().cpu().numpy()
            
            #x_hat1 = x_hat1.reshape(x_hat1.shape[0],-1)
            
            #images1 = images.detach().cpu().numpy()
            
            #images1 = images1.reshape(images1.shape[0],-1)
            
            mse2 = criterion(ximg,xcnew)
        
            comb_loss = mse2 + mse
            comb_loss.backward()
        
            enc_optim.step()
            dec_optim.step()
        
            train_loss += comb_loss.item()*images.size(0)
            tmse_loss += mse.item()*images.size(0)
            tdiscr_loss += mse2.item()*images.size(0)
        
             
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        tmse_loss = tmse_loss/len(train_loader)
        tdiscr_loss = tdiscr_loss/len(train_loader)
        print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                train_loss,tmse_loss,tdiscr_loss))
        
    
    
        #store the best encoder and decoder models
        #here, /crs5 is a reference to 5 way cross validation, but is not
        #necessary for illustration purposes
        if train_loss < best_loss:
            print('Saving..')
         
            torch.save(encoder.state_dict(), 'bst_enc_new_dataset.pth')
            torch.save(decoder.state_dict(), 'bst_dec_new_dataset.pth')
            #np.savetxt('./decodedoriginal.txt', images1)
            #print('decoded original saved!')
    
            best_loss = train_loss
    
    
    #in addition, store the final model (may not be the best) for
    #informational purposes
    
    torch.save(encoder.state_dict(), 'f_enc.pth')
    torch.save(decoder.state_dict(), 'f_dec.pth')
    print()
         
t1 = time.time()
print('total time(min): {:.2f}'.format((t1 - t0)/60))             
 
t4 = time.time()
print('final time(min): {:.2f}'.format((t4 - t3)/60))