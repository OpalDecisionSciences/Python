
################################### References ########################################
# 'Hands-On Machine Learning with SciKit Learn & TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems' First Edition; Geron, Aurelien; O'Reilly, 2017

import warnings
warnings.simplefilter(action = 'ignore', category = UserWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline

##################################################################################################
### Build a Classifier to Detect the Iris-Virginica Type Based Only on the Petal-Width Feature ###
##################################################################################################

iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 3:]  # Petal Width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

# Train a model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Estimated probability for flowers with petal widths varying from 0 to 3 cm
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
