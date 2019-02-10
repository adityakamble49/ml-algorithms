# # Linear Regression using Gradient Descent

# ## Import Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ## Import Data set

train_data = pd.read_csv('apartment_data_train.csv')
train_data.head()

train_data.info()

train_data.describe()

test_data = pd.read_csv('apartment_data_test.csv')
test_data.head()

# ## Feature Separation and Normalization

X_train = train_data.iloc[:, :-1].values
X_test = test_data.iloc[:, :].values
y_train = train_data.iloc[:, -1:].values
m = y_train.shape[0]

train_data_mean = train_data.mean(0)
train_data_std = train_data.std(0)

X_train_mean = train_data_mean[0:2].values
X_train_std = train_data_std[0:2].values
X_train_norm = (X_train - X_train_mean) / X_train_std

X_test_norm = (X_test - X_train_mean) / X_train_std

# ## Add Intercept Term to Features

train_ones = np.ones((X_train_norm.shape[0], 1))
test_ones = np.ones((X_test_norm.shape[0], 1))
X_train_norm = np.column_stack((train_ones, X_train_norm))
X_test_norm = np.column_stack((test_ones, X_test_norm))


# ## Perform Gradient Descent

# ### Compute Cost

def compute_cost(X, y, theta):
    hx = np.matmul(X, theta)
    error_values = hx - y
    squared_error = np.square(error_values)
    cost = np.sum(squared_error, 0)
    return cost


# ### Gradient Descent

def gradient_descent(X, y, theta, alpha, iterations):
    iteration_array = np.array([itr for itr in range(iterations)])
    cost_history = []
    for iteration in range(iterations):
        hx = np.matmul(X, theta)
        error_value = hx - y
        error_value_multi = np.matmul(error_value.T, X)
        delta = np.multiply(error_value_multi.T, (alpha / m))
        theta = theta - delta
        cost_history.append(compute_cost(X, y, theta))
    return [theta, np.column_stack((iteration_array, np.asarray(cost_history)))]


# ### Use Gradient Descent

alpha = 0.01
num_iterations = 400
theta = np.zeros((X_train_norm.shape[1], 1))
theta, cost_history_result = gradient_descent(X_train_norm, y_train, theta, alpha, num_iterations)
result = np.matmul(X_test_norm, theta)
print(result)

# ### Cost Function

plt.plot(cost_history_result)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
