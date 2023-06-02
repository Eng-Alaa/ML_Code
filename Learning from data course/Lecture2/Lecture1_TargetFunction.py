# This code is to show how the relation of the number of samples and the knowledge 
# that we can get about the target function
import numpy as np
import matplotlib.pyplot as plt
from utilLect1 import predict, plot, BlackBox_TF2

#n_samples = 8
n_samples = input("Enter number of training points:")
print("Number of training points is: " + n_samples)
n_samples=int(n_samples)
np.random.seed(0)
x = 10 ** np.linspace(0, 1, n_samples)
y = BlackBox_TF2(x)

fig = plt.figure(figsize=(9, 3.5))
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.85, wspace=0.05)
plt.scatter(x, y, marker='x', c='k', s=50)
plt.title('Training points')
plt.show()

