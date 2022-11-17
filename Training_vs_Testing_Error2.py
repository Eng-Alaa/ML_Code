from __future__ import print_function
import matplotlib,sys
from util import plot, GenerateSyntheticFn1, accuracy, UpdateWeights, predict
from matplotlib import pyplot as plt
import numpy as np
from random import seed
#from random import random
import random
import numpy.matlib
seed(1)

nb_epoch		= 50
l_rate  		= 0.1
plot_each_epoch	= False
stop_early 		= True
N=1000

dim=2
LBx=np.zeros((2,1))
UBx=np.zeros((2,1))
for i in range(dim):
    LBx[i]=0
    UBx[i]=5
ImbalanceRatio=0.5
N1=N*ImbalanceRatio
N2=N-N1

Patterns, Labels= GenerateSyntheticFn1(UBx, LBx, N1, N2)
matrix=np.c_[np.ones([Patterns.shape[0],1]), Patterns]
matrix=np.c_[matrix, Labels]
IDXrand=np.random.permutation(matrix.shape[0])
matrix=matrix[IDXrand]
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

weights= [	 0.20,	1.00,  -1.00		] # initial weights specified in problem

TrainingData=matrix[0: np.int16(0.5*matrix.shape[0]),0:3]
TestData=matrix[np.int16(0.5*matrix.shape[0]):,0:3]
TrainingLabels=matrix[0: np.int16(0.5*matrix.shape[0]),3]
TestLabels=matrix[np.int16(0.5*matrix.shape[0]):,3]

plt.figure(figsize=(8.5, 6), dpi=130)
#IDX=Labels==0
#Temp= (IDX==True).sum()
plt.scatter(x=matrix[matrix[:,-1]==0,1], y=matrix[matrix[:,-1]==0,2], color='black', marker='o',s=50, alpha=8/10)
plt.scatter(x=matrix[matrix[:,-1]==1,1], y=matrix[matrix[:,-1]==1,2], color='black', marker='x', s=50, alpha=8/10)
plt.scatter(x=TrainingData[TrainingLabels==0,1], y=TrainingData[TrainingLabels==0,2], color='red', marker='.', s=50, alpha=8/10)
plt.scatter(x=TrainingData[TrainingLabels==1,1], y=TrainingData[TrainingLabels==1,2], color='blue', marker='.', s=50, alpha=8/10)
#plt.scatter(x=TestData[TestLabels==0,1], y=TestData[TestLabels==0,2], color='red', marker='s', s=50, alpha=8/10)
#plt.scatter(x=TestData[TestLabels==1,1], y=TestData[TestLabels==1,2], color='blue', marker='s', s=50, alpha=8/10)
plt.show()

# each matrix row: up to last row = inputs, last row = y (classification)

#Percentages=np.array([0.01,0.05,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,0.5])
Percentages=np.arange(0.05, 0.9, 0.05)
FinalTrainingAcc=np.zeros([Percentages.shape[0]])
FinalTestAcc=np.zeros([Percentages.shape[0]])
for i in range(Percentages.shape[0]):
    Per=Percentages[i]
    Current_TrainingData=TrainingData[0: np.int16(Per*TrainingData.shape[0]),0:3]
    Current_TrainingLabels=TrainingLabels[0: np.int16(Per*TrainingData.shape[0])]
    Matrix=np.c_[Current_TrainingData, Current_TrainingLabels]
    TestMatrix=np.c_[TestData, TestLabels]
    #plt.figure(figsize=(8.5, 6), dpi=130)
    #plt.scatter(x=TrainingData[TrainingLabels==0,1], y=TrainingData[TrainingLabels==0,2], marker="o", color='red', s=50, alpha=8/10)
    #plt.scatter(x=TrainingData[TrainingLabels==1,1], y=TrainingData[TrainingLabels==1,2], marker="o",color='blue', s=50, alpha=8/10)
    #plt.scatter(x=TestData[TestLabels==0,1], y=TestData[TestLabels==0,2], marker="x",color='red', s=50, alpha=8/10)
    #plt.scatter(x=TestData[TestLabels==1,1], y=TestData[TestLabels==1,2], marker="x",color='blue', s=50, alpha=8/10)
    #plt.show()

    for epoch in range(nb_epoch):
        cur_acc = accuracy(Matrix,weights)
        #print("Training accuracy: ",cur_acc)
        test_acc = accuracy(TestMatrix,weights)
        #print("Test accuracy: ",test_acc) 
        weights= UpdateWeights(Matrix,weights,l_rate)
    
    FinalTrainingAcc[i]=cur_acc
    FinalTestAcc[i]=test_acc

plt.figure(figsize=(8.5, 6), dpi=130)
#IDX=Labels==0
#Temp= (IDX==True).sum()
plt.plot(Percentages, FinalTrainingAcc, color='blue', label='Training accuracy')
plt.plot(Percentages, FinalTestAcc, color='red', label='Test accuracy')
plt.xlabel('Percentage of training data')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()



