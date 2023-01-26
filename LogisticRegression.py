from __future__ import print_function
import matplotlib,sys
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
# this function to prepare the contour plot
def PlotContour(X, weights):
    # define bounds of the domain
    #Data=np.r_[TrainingData,TestingData]
    Data=X[:,1:] 
    min1, max1 = Data[:, 0].min()-1, Data[:, 0].max()+1
    min2, max2 = Data[:, 1].min()-1, Data[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # make predictions for the grid
    yhat = (predict(np.c_[np.ones(grid.shape[0]).reshape(-1,1),grid], 0.5, weights)).astype(float)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    return xx, yy, zz
""" def predict(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return 1.0 if activation>=0.0 else 0.0
 """
# each matrix row: up to last row = inputs, last row = y (classification)
def accuracy(matrix,weights):
	num_correct = 0.0
	preds       = []
	for i in range(len(matrix)):
		pred   = predict(matrix[i][:-1],weights) # get predicted classification
		preds.append(pred)
		if pred==matrix[i][-1]: num_correct+=1.0 
	#print("Predictions:",preds)
	return num_correct/float(len(matrix))

def Sigmoid(z):
        return 1 / (1 + np.exp(-z))

def Loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
       
def predict_prob(X, weights):
    return Sigmoid(np.dot(X, weights))
    
def predict(X, threshold, weights):
    #return 1.0 if predict_prob(X, weights)>=threshold else 0.0
    return predict_prob(X, weights) >= threshold

nb_epoch		= 200
learning_rate  		= 0.2

		  # Bias 	i1 		i2 		y
matrix = [	[1.00,	0.08,	0.72,	1.0],
			[1.00,	0.10,	1.00,	0.0],
			[1.00,	0.26,	0.58,	1.0],
			[1.00,	0.35,	0.95,	0.0],
			[1.00,	0.45,	0.15,	1.0],
			[1.00,	0.60,	0.30,	1.0],
			[1.00,	0.70,	0.65,	0.0],
			[1.00,	0.92,	0.45,	0.0]]

matrix=np.array(matrix)
X=matrix[:,0:-1]
y=matrix[:,-1]
## Training 
#weights = np.zeros(X.shape[1])  #  inicializing our weights vector filled with zeros
weights= [	 0.20,	1.00,  -1.00		] # initial weights specified in problem

plt.figure(figsize=(8.5, 6), dpi=130)
plt.title("Training data")
plt.plot(np.transpose(X[y==0,1]),np.transpose(X[y==0,2]), 'ro', markersize=20)
plt.plot(np.transpose(X[y==1,1]),np.transpose(X[y==1,2]), 'bo', markersize=20)
plt.show()

for epoch in range(nb_epoch):
    z = np.dot(X, weights)
    h = Sigmoid(z)
    Error=(h - y)
    gradient = np.dot(X.T, Error) / y.size
    weights -= learning_rate * gradient
    print(f'loss: {Loss(h, y)} \t')
    print("\nEpoch %d \nWeights: "%epoch,weights)
    yhat=(predict(X, 0.5, weights)).astype(float)
    accuracy=sum(yhat==y)/len(y)
    print('Accuracy is ' + str(100*accuracy) + '%')
    if (epoch%5)==0:
        xx,yy,zz= PlotContour(X,weights)
        cmap = ListedColormap(['pink','lightgreen','lightblue'])
        bounds = [0, 0.4,0.6, 1.1]
        norm = BoundaryNorm(bounds, cmap.N)
        plt.figure(figsize=(8.5, 6), dpi=130)
        #plt.title("Active learner class predictions after one iteration")
        plt.plot(np.transpose(X[y==0,1]),np.transpose(X[y==0,2]), 'ro', markersize=20)
        plt.plot(np.transpose(X[y==1,1]),np.transpose(X[y==1,2]), 'bo', markersize=20)
        plt.contourf(xx, yy, zz, cmap=cmap)
        plt.plot(np.transpose(X[yhat==0,1]),np.transpose(X[yhat==0,2]), 'rx', markersize=25)
        plt.plot(np.transpose(X[yhat==1,1]),np.transpose(X[yhat==1,2]), 'bx', markersize=25)
        plt.show()

	#plot(matrix,weights,title="Epoch %d"%epoch)
		