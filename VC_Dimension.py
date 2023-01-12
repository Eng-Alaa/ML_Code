
from __future__ import print_function
import matplotlib,sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_classification


d=2
N=200
X, Y = make_classification(n_samples=N,n_features=d, n_redundant=0, n_informative=d, n_clusters_per_class=1)
print(X.shape[1])
if d==2:
    plt.figure(figsize=(8, 8))
    plt.plot(X[:,0],X[:,1], 'ko')
    plt.plot(X[Y==0,0],X[Y==0,1], 'rx')
    plt.plot(X[Y==1,0],X[Y==1,1], 'bx')
    plt.show()

def predict(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return 1.0 if activation>=0.0 else 0.0

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

TrainingData=X[0:int(N/2),:]
TestingData=X[int(N/2):,:]
TrainingLabels=Y[0:int(N/2)] 
TestingLabels=Y[int(N/2):] 
Testing_matrix=np.ones(TestingData.shape[0])
Testing_matrix=np.c_[Testing_matrix,TestingData]
Testing_matrix=np.c_[Testing_matrix,TestingLabels]
if d==2:
    plt.figure(figsize=(8, 8))
    plt.plot(TrainingData[:,0],TrainingData[:,1], 'ro')
    plt.plot(TestingData[:,0],TestingData[:,1], 'bo')
    plt.show()

nb_epoch		= 10
l_rate  		= 0.1
Partitions=TrainingData.shape[0]
FinalTestAcc=np.zeros(Partitions)
FinalTrainingAcc=np.zeros(Partitions)
for i in range(Partitions):
    #print(str(int((i+1)*(1/Partitions)*TrainingData.shape[0])))
    CurrentTrainingData=TrainingData[0:int((i+1)*(1/Partitions)*TrainingData.shape[0]) ,:]
    CurrentTrainingLabels=TrainingLabels[0:int((i+1)*(1/Partitions)*TrainingData.shape[0])]
    Training_matrix=np.ones(CurrentTrainingData.shape[0])
    Training_matrix=np.c_[Training_matrix,CurrentTrainingData]
    Training_matrix=np.c_[Training_matrix,CurrentTrainingLabels]
    #print(Training_matrix.shape[0])
    #weights= [	 0.20,	1.00,  -1.00		] # initial weights specified in problem
    weights=np.random.rand(d+1,1)
    ## Training 
    for epoch in range(nb_epoch):
        cur_acc = accuracy(Training_matrix,weights)
        #print("\nEpoch %d \nWeights: "%epoch,weights)
        #print("Accuracy: ",cur_acc)
                        
        for k in range(len(Training_matrix)):
            prediction = predict(Training_matrix[k][:-1],weights) # get predicted classificaion
            error      = Training_matrix[k][-1]-prediction		 # get error from real classification
            #sys.stdout.write("Training on data at index %d...\n"%(i))
            for j in range(len(weights)): 				 # calculate new weight for each node
                #print(f'Weights[{j}] is {weights[j]:.2f}')
                weights[j] = weights[j]+(l_rate*error*Training_matrix[k][j]) 
                #print(f'Weights[{j}] is {weights[j]:.2f}')
    
    cur_test_acc = accuracy(Testing_matrix,weights)
    cur_training_acc = accuracy(Training_matrix,weights)
    
    FinalTestAcc[i]=cur_test_acc
    FinalTrainingAcc [i]=cur_training_acc

idx=[i for i in range(FinalTestAcc.shape[0])]
plt.plot(idx, FinalTestAcc.reshape(-1,1), color='red')
plt.plot(idx, FinalTrainingAcc.reshape(-1,1), color='blue')
plt.show()


    

    






# 	Bias 	i1 		i2 		y
""" matrix = [	[1.00,	0.08,	0.72,	1.0],
			[1.00,	0.10,	1.00,	0.0],
			[1.00,	0.26,	0.58,	1.0],
			[1.00,	0.35,	0.95,	0.0],
			[1.00,	0.45,	0.15,	1.0],
			[1.00,	0.60,	0.30,	1.0],
			[1.00,	0.70,	0.65,	0.0],
			[1.00,	0.92,	0.45,	0.0]]
 """



