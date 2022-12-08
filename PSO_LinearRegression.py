import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_regression

def predict_Reg(inputs,weights):
	activation=0.0
	for i,w in zip(inputs,weights):
		activation += i*w 
	return activation

# each matrix row: up to last row = inputs, last row = y (classification)
def Our_MSE(matrix,Weights,y):
    MSE = 0.0
    preds       = []
    for i in range(len(matrix)):
        pred   = predict_Reg(matrix[i],Weights)
        preds.append(pred)
        MSE+= pow((pred-y[i]),2)
    #print("Predictions:",preds)
    return MSE/len(matrix)

# Create data set.
n=10
x, y = make_regression(n_samples=n, n_features=1,
                       n_informative=1, noise=10, random_state=10)
# sample data instance.
x_sample = np.array([[-2], [2]])

# Convert  target variable array from 1d to 2d.
y = y.reshape(n, 1)
#weights= [	 0.20,	1.00		] # initial weights specified in problem

# Adding x0=1 to each instance
x_new = np.array([np.ones(len(x)), x.flatten()]).T
x_new_sample = np.array([np.ones(len(x_sample)), x_sample.flatten()]).T

def f(x_new,y,w0,w1):
    "Objective function"
    Weights=np.c_[w0,w1]
    fitness_value=np.zeros((len(w0),1))
    for i in range(len(w0)):
        fitness_value[i]=Our_MSE(x_new,Weights[i,:],y)
    return fitness_value

#print(f(x_new,y,np.array([4.55]),np.array([81.14])))
#w0=np.array([0.2])
    #return (x-3.14)*2 + (y-2.72)*2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)
    
# Compute and plot the function in 3D within [0,5]x[0,5]
""" w0, w1 = np.array(np.meshgrid(np.linspace(0,5,100), np.linspace(0,5,100)))
w0=w0.reshape(-1,1)
w1=w1.reshape(-1,1)
w1.shape
weights=np.c_[w0,w1]
weights.shape """

# Find the global minimum
#x_min = x.ravel()[z.argmin()]
#y_min = y.ravel()[z.argmin()]
 
# Hyper-parameter of the algorithm
c1 = c2 = 2
w = 1
 
# Create particles
n_particles = 20
iterations=1000
np.random.seed(100)
W = np.random.rand(2, n_particles) *1
V = np.random.randn(2, n_particles) * 0.1
 
# Initialize data
pbest = W # position
pbest_obj = f(x_new, y, W[0], W[1]) # fitness/objective value
gbest = pbest[:, pbest_obj.argmin()] # position
gbest_obj = pbest_obj.min() # fitness/objective value
 
def update(x_new,y,W,V,pbest, pbest_obj,gbest,gbest_obj):
    "Function to do one iteration of particle swarm optimization"
    #global V, W, pbest, pbest_obj, gbest, gbest_obj
    # Update params
    r1, r2 = np.random.rand(2)
    V = w * V + c1*r1*(pbest - W) + c2*r2*(gbest.reshape(-1,1)-W)
    W = W + V
    obj = f(x_new,y, W[0], W[1])
    pbest[:, (pbest_obj.reshape(-1,) >= obj.reshape(-1,))] = W[:, (pbest_obj.reshape(-1,) >= obj.reshape(-1,))]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()
    print(gbest_obj)
    return W,V,pbest, pbest_obj,gbest,gbest_obj

for i in range(iterations):
    W,V,pbest, pbest_obj,gbest,gbest_obj =update(x_new,y,W,V,pbest, pbest_obj,gbest,gbest_obj)

print(f"prediction of ({x_new_sample[0]}=", predict_Reg(x_new_sample[0],gbest))
print(f"prediction of ({x_new_sample[1]}=", predict_Reg(x_new_sample[1],gbest))
print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
