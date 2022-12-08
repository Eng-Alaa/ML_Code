import numpy as np
import matplotlib.pyplot as plt

X=np.array([1, 2, 3, 4])
y=np.array([1, 2, 3, 4])
#y=np.array([1, -1, 3.5, 5])

def Cost_function(weight,X,y):
    sum=0
    for i in range(len(X)):
        sum+=np.power((weight*X[i]-y[i]),2)
    return sum/(2*len(X))

cost_value=np.array([])
Weights=np.array([])
for w1 in np.arange(-0.5, 2.5, 0.01):
    print(Cost_function(w1,X,y))
    cost_value=np.r_[cost_value,np.array(Cost_function(w1,X,y))]
    Weights=np.r_[Weights, w1]
plt.plot(Weights.reshape(1,-1), cost_value.reshape(1,-1), 'bo')
plt.show()

