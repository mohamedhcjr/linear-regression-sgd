import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Load Dataset
# ---------------------------------
data = pd.read_csv('student_scores.csv')
print(data.head())
# رسم النقاط
#Assign feature column to x and label column to y
x = data['Hours'].values
y = data['Scores'].values

print(x)
print(y)
print(x.shape)
print(y.shape)
#Visualization

plt.figure(figsize=(8,6))
plt.title('Data distribution')
plt.scatter(x, y)
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
# Hyper-parameteres
L_rate = 0.001
iterations = 100

# Initialization
theta_1= 0
theta_0= 0

# The number of samples in the dataset
n = x.shape[0]

# An empty list to store the error in each iteration
losses = []
for i in range(iterations):
    h_x = theta_0 + theta_1*x
    
    # Keeping track of the error decrease
    mse = (1/n) * np.sum((h_x - y)**2)
    losses.append(mse)
    
    # Derivatives
    d_theta0 = (2/n) * np.sum(h_x-y)
    d_theta1 = (2/n) * np.sum(x * (h_x-y))
    # Values update
    theta_1 = theta_1 - L_rate * d_theta1
    theta_0 = theta_0 - L_rate* d_theta0

print("theta_0= ", theta_0)
print("theta_1= ", theta_1)
print("MSE= ", mse)
# predict the score for a new student
new_x= 9

Prediction_Model = theta_0 + theta_1*new_x
print ('Score:', Prediction_Model)
#---------------------
x_line = np.linspace(0,10,100)
y_line = theta_0 + theta_1*x_line
plt.figure(figsize=(8,6))
plt.title('Data distribution')
plt.plot(x_line, y_line, c='r')
plt.scatter(x, y, s=10)
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
# ------------
plt.title('Loss values')
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('iteration')
print('Initial loss\t:', losses[0])
print('Final loss\t:', losses[-1])