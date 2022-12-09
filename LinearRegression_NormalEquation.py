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
  
# Using Normal Equation.
theta_best_values = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y)
  
# Display best values obtained.
print(theta_best_values)

# Adding x0=1 to each instance.
x_sample_new = np.array([np.ones(len(x_sample)), x_sample.flatten()]).T
  
# predict the values for given data instance.
predict_value=x_sample_new.dot(theta_best_values)
print(predict_value)

# Plot the output.
plt.scatter(x,y,s=30,marker='o')
plt.plot(x_sample,predict_value,c='red')
plt.plot()
plt.xlabel("Feature_1 --->")
plt.ylabel("Target_Variable --->")
plt.title('Simple Linear Regression')
plt.show()

# Verification.
# Here, we compare our results from normal equation regression model and builtin regression model in Scikit learn
from sklearn.linear_model import LinearRegression
lr=LinearRegression()    # Object.
lr.fit(x,y)              # fit method.
  
# Print obtained theta values.
print("Best value of weights:",lr.intercept_,lr.coef_,sep='\n')
  
#predict.
print("predicted value:",lr.predict(x_sample),sep='\n')