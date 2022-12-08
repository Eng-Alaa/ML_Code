import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
  
# Create data set.
n=10
x, y = make_regression(n_samples=n, n_features=1,
                       n_informative=1, noise=10, random_state=10)
# sample data instance.
x_sample = np.array([[-2], [2]])
  
# Plot the generated data set.
plt.scatter(x, y, s=30, marker='o')
plt.scatter(x_sample, [0, 0], s=30, marker='x', color='red')
plt.xlabel("Feature_1 --->")
plt.ylabel("Target_Variable --->")
plt.title('Simple Linear Regression')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()

# Convert  target variable array from 1d to 2d.
y = y.reshape(n, 1)

# Adding x0=1 to each instance
x_new = np.array([np.ones(len(x)), x.flatten()]).T

def predict_Reg(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return activation

# each matrix row: up to last row = inputs, last row = y (classification)
def Our_MSE(matrix,weights,y):
    MSE = 0.0
    preds       = []
    for i in range(len(matrix)):
        pred   = predict_Reg(matrix[i],weights)
        preds.append(pred)
        MSE+= pow((pred-y[i]),2)
    #print("Predictions:",preds)
    return MSE/len(matrix)

nb_epoch		= 100
l_rate  		= 0.1
weights= [	 0.20,	1.00		] # initial weights specified in problem
x_new_sample = np.array([np.ones(len(x_sample)), x_sample.flatten()]).T

## Training 
for epoch in range(nb_epoch):
    print("\nEpoch %d \nWeights: "%epoch,weights)
    Mse = Our_MSE(x_new,weights,y)
    print("MSE: ",Mse)
    print(f'The prediction of sample [{x_sample[0]}] is {predict_Reg(x_new_sample[0],weights)}')
    print(f'The prediction of sample [{x_sample[1]}] is {predict_Reg(x_new_sample[1],weights)}')
		
    for i in range(len(x_new)):
        prediction = predict_Reg(x_new[i],weights) # get predicted regression
        error      = np.array(prediction)-y[i]		 # get error from real regression
        for j in range(len(weights)): 				 # calculate new weight for each node
            #print(f'Weights[{j}] is {weights[j]}')
            weights[j] = weights[j]-(l_rate*error*x_new[i][j]) 
            #print(f'Weights[{j}] is {weights[j]}')
    
    
