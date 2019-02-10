# # Logistic Regression using Gradient Descent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ## Import Data Set
from sklearn.preprocessing import PolynomialFeatures

train_data = pd.read_csv('micro_data_train.csv')

train_data.head()

train_data.info()

train_data.describe()

test_data = pd.read_csv('micro_data_test.csv')
test_data.head()

train_data_0 = train_data[train_data.accepted == 0]
train_data_1 = train_data[train_data.accepted == 1]
plt.scatter(train_data_1.iloc[:, 0], train_data_1.iloc[:, 1], marker='+', color='black')
plt.scatter(train_data_0.iloc[:, 0], train_data_0.iloc[:, 1], marker='o', color='y')
plt.xlabel('Micro Test 1')
plt.ylabel('Micro Test 2')
plt.legend(labels=['Accepted', 'Rejected'])
# plt.show()

# ## Feature Separation and Normalization

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
m = y_train.shape[0]

X_test = test_data.iloc[:, :].values

poly = PolynomialFeatures(6)
X_train_mapped = poly.fit_transform(X_train[:, 0:2])
X_test_mapped = poly.fit_transform(X_test[:, -1:])

cost_history = []
gradient_history = []


# ## Gradient Descent

def sigmoid(input_var):
    sigmoid_result = 1 / (1 + np.exp(-input_var))
    return sigmoid_result


def compute_cost_reg(theta, X, y):
    z = np.dot(X, theta)
    hx = sigmoid(z)
    neg_0_cost = (-1 * (1 - y)) * np.log((1 - hx))
    pos_1_cost = (-1 * y) * np.log(hx)
    cost_normal = (pos_1_cost + neg_0_cost) / m
    cost_reg = (lambda_value / (2 * m)) * np.sum(np.power(theta[1:], 2))
    cost_normal_sum = np.sum(cost_normal)
    cost = cost_normal_sum + cost_reg
    cost_history.append(cost)
    return cost.flatten()


def gradient_reg(theta, X, y):
    z = np.dot(X, theta)
    hx = sigmoid(z)
    error_value = hx - y
    error_value_final = np.matmul(error_value.T, X)
    delta = error_value_final / m
    delta_reg = (lambda_value / m) * theta.reshape(-1, 1).T
    delta_reg[0] = 0
    delta_total = delta + delta_reg
    grad = delta_total.flatten()
    gradient_history.append(gradient_history)
    return grad.flatten()


lambda_value = 1
theta_value = np.zeros(X_train_mapped.shape[1])
cost = compute_cost_reg(theta_value, X_train_mapped, y_train)
grad = gradient_reg(theta_value, X_train_mapped, y_train)

print("For lambda = 1 and theta = zeros")
print("Cost: " + str(cost))
print("Grad (First 5): " + str(grad[:5]))

lambda_value = 10
theta_value = np.ones(X_train_mapped.shape[1])
cost = compute_cost_reg(theta_value, X_train_mapped, y_train)
grad = gradient_reg(theta_value, X_train_mapped, y_train)

print("For lambda = 10 and theta = ones")
print("Cost: " + str(cost))
print("Grad (First 5): " + str(grad[:5]))

# Rest history
cost_history.clear()
gradient_history.clear()

# lambda_value = 1
# theta_value = np.zeros(X_train_mapped.shape[1])
# result = optimize.minimize(fun=compute_cost_reg, x0=theta_value, args=(X_train_mapped, y_train), method='TNC',
#                            jac=gradient_reg)
# print(result)
