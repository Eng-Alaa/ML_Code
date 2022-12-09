import torch
import torchvision
import torchvision.datasets as datasets
import PIL
#from numpy import asarray
import numpy as np
from sklearn.decomposition import PCA
from sklearn import decomposition
from matplotlib import pyplot as plt

# read the data
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

examples = enumerate(mnist_trainset)
batch_idx, (example_data, example_targets) = next(examples)

# display the first image in the training data 
train_image_zero, train_target_zero = mnist_trainset[0]
train_image_zero.show()

# select only from the first 500 images, the images belong to the one and eight classes
# arrange each image into the features matrix as a column
# By the end of this loop, we will have features matrix (each observation is a column) [784xn], n is the number of data 
# in Features, and Targets is the label for each data point [1xn] 
counter=0
for i in range(500):
    train_image, train_target = mnist_trainset[i]
    if (train_target==1):
        data_one = np.asarray(train_image)
        Label=1
        if (counter==0):
            Features = data_one.reshape(-1,1)
            Targets= Label
        else:
            Features = np.c_[Features, data_one.reshape(-1,1)]
            Targets= np.c_[Targets, Label]        
        counter+=1  
    elif (train_target==8):
        data_eight = np.asarray(train_image)
        Label=8
        if (counter==0):
            Features = data_eight.reshape(-1,1)
            Targets= Label
        else:
            Features = np.c_[Features, data_eight.reshape(-1,1)]
            Targets= np.c_[Targets, Label]        

# make transpose for Features and Targets (i.e., each observation will be a row (not a column))
Targets=np.transpose(Targets)
Features=np.transpose(Features)

# use PCA to reduce the size of the data to only two components
pca = decomposition.PCA(n_components=2)
pca.fit(Features)
ReducedFeatures = pca.transform(Features)

# shaow the data
idx=(~(Targets==1)).reshape(-1,)
c0s = plt.scatter(ReducedFeatures[~idx,0],ReducedFeatures[~idx,1],s=40.0,c='r',label='Class one')
c1s = plt.scatter(ReducedFeatures[idx,0],ReducedFeatures[idx,1],s=40.0,c='b',label='Class eight')
plt.legend(fontsize=10,loc=1)
plt.show()


