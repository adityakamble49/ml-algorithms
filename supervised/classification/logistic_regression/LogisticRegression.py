import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Data Set

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

# Feature Separation and Normalization

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1:].values
m = y_train.shape[0]

X_test = test_data.iloc[:, :].values


def map_features(x1, x2):
    degree = 6
    mapped_result = np.ones((x1.shape[0], sum(range(degree + 2))))
    current_col = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            mapped_result[:, current_col] = np.power(x1, i - j)
            np.power(x2, j)
            current_col += 1
    return mapped_result


X_train_mapped = map_features(X_train[:, 0], X_train[:, 1])
X_test_mapped = map_features(X_test[:, 0], X_test[:, 1])


# Gradient Descent

def sigmoid(input_var):
    sigmoid_result = 1 / (1 + np.exp(-input_var))
    return sigmoid_result


def cost_grad_reg(X, y, theta, lambda_val):
    z = np.matmul(X, theta)
    hx = sigmoid(z)
    neg_0_cost = (-1 * (1 - y)) * np.log((1 - hx))
    pos_1_cost = (-1 * y) * np.log(hx)
    cost_normal = (pos_1_cost + neg_0_cost) / m
    cost_reg = (lambda_val / (2 * m)) * np.sum(np.power(theta[1, :], 2))
    cost_normal_sum = np.sum(cost_normal)
    cost = cost_normal_sum + cost_reg

    error_value = hx - y
    error_value_final = np.matmul(error_value.T, X)
    delta = error_value_final / m
    delta_reg = (lambda_val / m) * theta.T
    delta_reg[0] = 0
    delta_total = delta + delta_reg
    return [cost, delta_total]


alpha = 0.1
lambda_value = 1
theta_value = np.zeros((X_train_mapped.shape[1], 1))
[cost, delta_total_result] = cost_grad_reg(X_train_mapped, y_train, theta_value, lambda_value)
print(cost)
print(delta_total_result)
