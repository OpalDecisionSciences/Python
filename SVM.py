
# Reference: 'Hands-On Machine Learning with SciKit Learn & TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems' First Edition; Geron, Aurelien; O'Reilly, 2017

# SVM - A 'most-popular' model, powerful and versatile for linear/non-linear classification, regression, outlier detection, and well-suited for classification of complex but small-medium sized data sets 

import os
import numpy as np
import pandas as pd

import time

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import axes
import matplotlib.pyplot as plt
mpl.rc('axes',labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from scipy.stats import reciprocal, uniform

from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split

np.random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = CURRENT_DIR + "\\"

#################### Save Figures ###################

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


############################################################
############### Large Margin Classification ################
############################################################

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  
y = iris["target"] 

setosa_or_versicolor = (y==0) | (y==1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)


# SVC(C=inf, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto_deprecated', 
#     kernel='linear', max_iter=-1, probability=False, random_state=None,
#     shrinking=True, tol=0.0001, verbose=False)



#############################################################
#################### Good vs Bad Models #####################
#############################################################

# Left graph exhibits decision boundaries for three possible linear classifiers
### The model represented by the dotted line does not properly separate classes
### The other two models in the left plot work well on this training set, but
### the decision boundaries come so close to the instances, or support vectors, 
### that the model will likely not perform well on new data - overfitting
#
# In contrast, the solid line in the right plot represents the decision boundary  
### of a SVM classifier that effectively separates the two classes while staying 
### as far away from the closest training instances as possible
# 
# SVM classifiers should fit the widest possible space between the classes, 
# or 'street' as represented by the dotted lines
# a.k.a. large margin classification

x0 = np.linspace(0, 5.5, 200)
pred_1 = 5*x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision bundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), constrained_layout=True, sharey=True)

plt.sca(axes[0])
plt.plot(x0, pred_1, "g--", linewidth=2)
plt.plot(x0, pred_2, "m-", linewidth=2)
plt.plot(x0, pred_3, "r-", linewidth=2)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis=([0, 5.5, 0, 2])

plt.sca(axes[1])
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
plt.xlabel("Petal length", fontsize=14)
plt.axis=([0, 5.5, 0, 2])
plt.ylim([-0.1, 2.0])
save_fig("large_margin_classification_plot")
plt.show()


#######################################################
############ Sensitivity to Feature Scales ############
#######################################################

Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float)
ys = np.array([0, 0, 1, 1])
svm_clf = SVC(kernel="linear", C=100)
svm_clf.fit(Xs, ys)

plt.figure(figsize=(9,2.7))
plt.subplot(121)
plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")
plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")
plot_svc_decision_boundary(svm_clf, 0, 6)
plt.xlabel("$x_0$", fontsize=20)
plt.ylabel("$x_1$    ", fontsize=20, rotation=0)
plt.title("Unscaled", fontsize=16)
plt.axis=([0, 6, 0, 90])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xs)
svm_clf.fit(X_scaled, ys)

plt.subplot(122)
plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")
plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 0][ys==0], "ms")
plot_svc_decision_boundary(svm_clf, -2, 2)
plt.xlabel("$x_0$", fontsize=20)
plt.ylabel("$x_1$  ", fontsize=20, rotation=0)
plt.title("Scaled", fontsize=16)
plt.axis=([-2, 2, -2, 2])

save_fig("sensitivity_to_feature_scales_plot")


############################################################
################## Sensitivity to Outliers #################
############################################################

####### Hard Margin vs Soft Margin Classification #########

# Hard margin classification is sensitive to outliers and 
# only works if the data is linearly separable
# Hard margin classification strictly imposes that all 
# instances be 'off the street' and off to the side
### The left graph shows this as impossible
### The right graph shows a decision boundary that will 
### likely not generalize well

##### The objective is to find a balance between keeping 
##### the street as wide as possible and limiting the margin
##### violations, or the number of instances that end up in 
##### the middle of the street or on the wrong side of the 
##### street; soft margin classification, a more flexible model

# This balance is controlled by the C hyperparameter in 
# Scikit-Learn's SVM classes, where smaller C values lead 
# to a wider street with more margin violations

X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
y_outliers = np.array([0, 0])
Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
Xo2 = np.concatenate([X, X_outliers[:1]], axis=0)
yo2 = np.concatenate([y, y_outliers[:1]], axis=0)

svm_clf2 = SVC(kernel="linear", C=10**9)
svm_clf2.fit(Xo2, yo2)

fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

plt.sca(axes[0])
plt.plot(Xo1[:, 0][yo1==1], Xo1[:, 1][yo1==1], "bs")
plt.plot(Xo1[:, 0][yo1==0], Xo1[:, 1][yo1==0], "yo")
plt.text(0.3, 1.0, "Impossible!", fontsize=24, color="red")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("petal width", fontsize=14)
plt.annotate("Outlier", 
            xy=(X_outliers[0][0], X_outliers[0][1]),
            xytext=(2.5, 1.7),
            ha="center",
            arrowprops=dict(facecolor='black', shrink=0.1),
            fontsize=16,
            )
plt.axis=([0, 5.5, 0, 2])

plt.sca(axes[1])
plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "bs")
plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "yo")
plot_svc_decision_boundary(svm_clf2, 0, 5.5)
plt.xlabel("Petal length", fontsize=14)
plt.annotate("Outlier",
            xy=(X_outliers[1][0], X_outliers[1][1]),
            xytext=(3.2, 0.08),
            ha="center",
            arrowprops=dict(facecolor='black', shrink=0.1),
            fontsize=16,
            )
plt.axis=([0, 5.5, 0, 2])
plt.ylim([-0.1, 2.0])
save_fig("sensitivity_to_outliers_plot")
plt.show()


#########################################################
########## Large Margin vs Margin Violations ############
#########################################################

# Two soft margin SVM classifiers on a non-linearly separable 
# dataset

# Left plot shows a high C value exhibiting less margin 
# violations with a smaller margin, or street
# Right plot has a low C value, wider margin, with many instances
# in the street; however, in this case, most of the margin violations
# are on the correct side of the decision boundary, this model 
# leads to fewer prediction errors, and therefore will likely 
# generalize better than the model on the left

##### If your SVM model is overfitting you can try regularizing it 
##### by reducing C value

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris virginica

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])
svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])

# Generate graphs comparing different regularization settings
scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])
scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)
###### Received Convergence Warning: Increase the number of iterations

# Convert to Unscaled Parameters
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])

w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_

svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])

svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

# Find support vectors - Linear SVC does not do this automatically
t = y * 2 - 1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()

svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]

fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

plt.sca(axes[0])
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris virginica")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris versicolor")
plot_svc_decision_boundary(svm_clf1, 4, 5.9)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Ptal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
plt.axis=([4, 5.9, 0.8, 2.8])
plt.ylim([0.75, 2.75])
plt.xlim([4.0, 6.0])

plt.sca(axes[1])
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 5.99)
plt.xlabel("Petal length", fontsize=14)
plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
plt.axis=([4, 5.9, 0.8, 2.8])
plt.ylim([0.75, 2.75])
plt.xlim([4.0, 6.0])

save_fig("regularization_plot")


########################################################################
######################### Non-Linear Classification ####################
########################################################################

# Adding more features to non-linear datasets is one approach to handling 
# non-linear datasets, such as Polynomial features, and can result in
# linearly separable datasets in some cases

### Shows a simple one feature dataset that is not linearly separable
### Adding a second feature, X_2 = (X_1)^2, results in a 2-dimensional 
### dataset that is perfectly linearly separable
 
X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
X2D = np.c_[X1D, X1D**2]
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(10, 3))

plt.subplot(121)
plt.grid(True, which='both')
plt.axhline(y=0, color="k")
plt.plot(X1D[:, 0][y==0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][y==1], np.zeros(5), "g^")
plt.gca().get_yaxis().set_ticks([])
plt.xlabel(r"$x_1$", fontsize=20)
plt.axis=([-4.5, 4.5, -0.2, 0.2])

plt.subplot(122)
plt.grid(True, which='both')
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.plot(X2D[:, 0][y==0], X2D[:, 1][y==0], "bs")
plt.plot(X2D[:, 0][y==1], X2D[:, 1][y==1], "g^")
plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
plt.axis=([-4.5, 4.5, -1, 17])

plt.subplots_adjust(right=1)

save_fig("higher_dimensions_plot", tight_layout=False)
plt.show()


# Side Project for Later: 
# Mathematically find the focus and latus rectum of the parabola on the right; for the higher dimension plot
# Detail additions of focal chords vs latus rectum for fun, added character, and calculus implementation

# Continuing from above
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$'", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()