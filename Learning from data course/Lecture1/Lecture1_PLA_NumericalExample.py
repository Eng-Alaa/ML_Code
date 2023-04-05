# This code is an illustrative numerical example to explain the PLA algorithm using only two data points
""" for v in dir():
    exec('del '+ v)
    del v """
from __future__ import print_function
import matplotlib,sys
from matplotlib import pyplot as plt
import numpy as np
from utilLect1 import predict, plot
	
# each matrix row: up to last row = inputs, last row = y (classification)
		   # Bias 	f1 		f2 		y
matrix = [	[1.00,	1,	1,	1.0],
			[1.00,	2,	-2,	-1]]

# initial weights specified in problem (---> A- model representation)
weights= [	 0,	1.00,  0.5		] 
plot(matrix,weights,title="Epoch")

""" def accuracy(matrix,weights):
	num_correct = 0.0
	preds       = []
 """				
nb_epoch= 10
l_rate= 0.2

## Training 
for epoch in range(0,nb_epoch):
	#print(epoch)
	for i in range(len(matrix)):
		# ---> B-Model evaluation: 
		# ------->	B1- calculating predictions
		# ------->	B2- compare predictions (classifications) with target values	
		wTX = np.array(matrix[i][:-1])*np.array(weights) # get predicted classification
		if (np.sum(wTX))>=0:
			pred=1
		else:
			pred=-1
		print(pred)
		if pred==int(matrix[i][-1]):
			print('The training point ' + str(i+1) + ' is classified correctly')
		else:
			print('The training point ' + str(i+1) + ' is classified Incorrectly')
		# get error from real classification
		if int(matrix[i][-1])==pred:
			error=0
		else:
			error=1
    # ---> C-Model update (optimization)
		if error!=0:
			for j in range(len(weights)): 				 # calculate new weight for each node
				#print(f'Weights[{j}] is {weights[j]:.2f}')
				weights[j] = weights[j]+(l_rate*matrix[i][j]*matrix[i][-1]) 
				#print(f'Weights[{j}] is {weights[j]:.2f}')
	plot(matrix,weights,title="Epoch %d"%epoch)

