

################################### References ########################################
# 'Hands-On Machine Learning with SciKit Learn & TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems' First Edition; Geron, Aurelien; O'Reilly, 2017
# 'The Hundred-Page Machine Learning Book', Andriy Burkov


############################ How things work under the hood ###########################
import warnings
warnings.simplefilter(action = 'ignore', category = UserWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
import numpy as np
import numpy.random as rnd
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline

# To make this file stable across all runs
np.random.seed(42)


####################################################################################
################################ The Normal Equation ###############################
####################################################################################

# Computes the inverse of X^T * X, an n* n matrix, with computational complexity of O(n^2.4) to O(n^3)

# Theta_hat = (X^T * X)^-1 * X^T * y

# Generate linear-looking data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best  # 3.4, 3.38

# Make predictions using theta
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
y_predict  # 3.4, 10.17

# Plot this model's predictions
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()

# Same thing using Scikit-Learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_  # Same results: 3.4, 3.38
lin_reg.predict(X_new)  # Same results: 3.4, 10.17


######################################################################################
################################## Gradient Descent ##################################
######################################################################################

# Gradient Descent tweaks parameters iteratively to minimize a cost function (MSE)
# Gradient is the generalization of derivative for functions that take several inputs (or one input in the form of a vecctor)
# A gradient of a funtion is a vector of partial derivatives
# Finding partial derivatives of a function is the same process as finding the derivative while focusing on one of the function's inputs at a time, and considering all other inputs as constant values (the derivative of a constant equals 0)
# Training a model means finding a combination of parameters that minimizes a cost function
# Ensuring features have a similar scale allows this process to reach global minimums more quickly

# Gradient vector of the cost function - Batch Gradient Descent is very slow on large training sets but faster than the Normal Equation on a LInear model with hundreds of thousands of features
2/m * X^T * (X * theta - y)  # Calculations over the full training set, X, at each gradient step

# Batch Gradient Descent - Gradient Decsent Step = theta_next_step = theta - gradient_vector_MSE

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title("eta = {}".format(eta), fontsize=16)

eta = 0.1  # learning rate
n_iterations = 1000
m = 100

np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

plt.figure(figsize=(10,4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.2)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)
plt.show()

theta  # Same results: 3.4, 3.38 - grid search can help find a good learning rate (use a limited number of iterations to elimate models that take too long to converge)


# Stochastic Gradient Descent - based on one random instance oposed to the whole training set

theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

theta  # Similar result: 3.44, 3.40 - a fairly good solution comparing 50 epochs to 1000 iterations from batch above

# Same thing using Scikit-Learn SGDRegressor - default optimizes the squared error cost function
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())
'''
SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
            eta=0.1, fit_intercept=True,l1_ratio=0.15,
            learning_rate='invscaling', loss='squared_loss', max_iter=50,
            n_iter=None, n_iter_no_change=5, penalty=None, power_t=0.25,
            random_state=42, shuffle=True, tol=-np.infty, validation_fraction=0.1,
            verbose=0, warm_start=False)
'''
sgd_reg.intercept_, sgd_reg.coef_  # Similar result: 3.44, 3.44

# Mini-Batch Gradient Descent
theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

theta

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(7,4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-Batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.show()

# Polynomial Regression - PolynomialFeatures(degree=d) transforms array of n features to an array of ((n+d)! / (d!n!) features
np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis=([-3, 3, 0, 10])
plt.show()


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]  # 0.3274
X_poly[0]  # 0.3274, 0.1072

# Now this has the original fetaure of X plus the square of this feature; fit LinearRegression model to extended training data
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_  # The model estimates y_hat = 0.494x**2 + 1.03x + 1.99 for x-sub-1, which is very close to the original function

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("y", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis=([-3, 3, 0, 10])
plt.show()


for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)

plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

plt.plot(X, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis=([-3, 3, 0, 10])
plt.show()


# Plotting learning curves
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

# Let's look at the learning curves for a 10th degree polynomial model on the same data
polynomial_regression = Pipeline((
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
))

plot_learning_curves(polynomial_regression, X, y)


#############################################################################################
########################## Regularization Techniques ########################################
#############################################################################################

# Ridge Regression - Cholesky : variant of theta_hat = (X^T * X + alphaA)^-1 * X^T * y using a matrix factorization technique
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])  # 5.178

# Ridge Regression - SGD : Penalty hyperparameter sets type of regularization term : l2 = adds regularization term to the cost function, half the square of the l-sub-2 norm of the weight vector
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])  # 5.154

# Lasso Regression - Least Absolute Shrinkage and Selection Operator Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])  # 5.140

# Elastic Net Regression - l1_ratio corresponds to the mix ratio r
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])  # 5.1378

# Early Stopping - A different way to regularize iterative learning algorithms : stop as soon as the validation erro reaches a minimum
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005) # Warm start continues training where it left off instead of starting from scratch
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y-train)  
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)

