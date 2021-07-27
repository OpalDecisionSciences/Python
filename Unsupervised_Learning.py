
# Reference: 'Hands-On Machine Learning with SciKit Learn & TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems' First Edition; Geron, Aurelien; O'Reilly, 2017

# Import packages
import os
import sys
import numpy as np
import pandas as pd
import sklearn

from scipy import stats
import urllib.request
from timeit import timeit

from matplotlib import interactive
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl
mpl.__file__  # Find file location where mpl is installed
mpl.get_configdir()  # mpl configuration and cache directory locations
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from sklearn.datasets import load_iris, make_blobs, fetch_openml
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, silhouette_samples

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = CURRENT_DIR + "\\"

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


#############################################################################
################################ Clustering #################################
#############################################################################

data = load_iris()
X = data.data
y = data.target
data.target_names

plt.figure(figsize=(9, 3.5))

plt.subplot(121)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(122)
plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
plt.xlabel("Petal length", fontsize=14)
plt.tick_params(labelleft=False)

save_fig("classification_vs_clustering_plot")
plt.show()

# Separate clusters using all 4 features: petal length, petal width, sepal length, sepal width
y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)

# Map most common class for each cluster using scipy.stats.mode()
mapping = {}
for class_id in np.unique(y):
    mode, _ = stats.mode(y_pred[y==class_id])
    mapping[mode[0]] = class_id

mapping

y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

plt.plot(X[y_pred==0, 2], X[y_pred==0, 3], "yo", label="Cluster 1")
plt.plot(X[y_pred==1, 2], X[y_pred==1, 3], "bs", label="Cluster 2")
plt.plot(X[y_pred==2, 2], X[y_pred==2, 3], "g^", label="Cluster 3")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.show()

np.sum(y_pred==y)

np.sum(y_pred==y) / len(y_pred)

#############################################################################
################################### K-Means #################################
#############################################################################

blob_centers = np.array(
    [[0.2, 2.3],
    [-1.5, 2.3],
    [-2.8, 1.8],
    [-2.8, 2.8],
    [-2.8, 1.3]]
)
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_plot")
plt.show()

# Train a k-means clusterer on this dataset 
# Hard clustering finds each blob's center and assigns each instance to the closest blob
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

# Each instance was asssigned to one of the 5 clusters
y_pred

# Each instance preserves the labels (index of the cluster) it gets assigned to (object.labels_)
y_pred is kmeans.labels_

# Estimated centroids
kmeans.cluster_centers_

# Predict labels of new instances
X_new = np.array([[0,2], [3,2], [-3,3], [-3,2.5]])
kmeans.predict(X_new)

##############################################################
###################### Voronoi Diagram #######################
##############################################################

# Plot the model's decision boundaries
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=35, linewidths=8, color=circle_color, zorder=11, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=2, linewidths=12, color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True, show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                        np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
    
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
save_fig("voronoi_plot")
plt.show()


############################################################################
##################### Hard Clustering vs Soft Clustering ###################
############################################################################

# Soft Clustering measures the distance of each instance to all centroids (5 in this case)
# This is what the transofrm() method does: Euclidian distance

kmeans.transform(X_new)

# Check Euclidia distance between each instance and each centroid

np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2)


############################################################################################
#######################################K-Means Algorithm ###################################
############################################################################################

# Fast and simple clustering algorithm
# Initialize k centroids randomly and repeat until convergence
# Assign each instance to the closest centroid
# Update the centroids to be the mean of the instances that are assigned to them
# Using hyperparamters that set the original K-Means Algorithm; see how the centroids move around

kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
                    algorithm="full", max_iter=1, random_state=0)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
                    algorithm="full", max_iter=2, random_state=0)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
                    algorithm="full", max_iter=3, random_state=0)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

# Plot this
plt.figure(figsize=(10,8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)", fontsize=14)

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Label the instances", fontsize=14)

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

save_fig("kmeans_algorithm_plot")
plt.show()


##############################################################
##################### K-Means Variability ####################
##############################################################

# In the original K-Means Algorithm the centroids initialize ranfomly and the 
# algorithm gradually improves the centroids with each iteration
# This is a problem because running this multiple times or with different random 
# seeds will likely converge to very different solutions; as shown below

def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10,3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)
    
    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)
    
kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,
                        algorithm="full", random_state=2)
kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,
                        algorithm="full", random_state=5)

plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
                        "Solution 1", "Solution 2 (with a different random init)")

save_fig("kmeans_variability_plot")
plt.show()


#######################################################
###################### Inertia ########################
#######################################################

# Evaluating model performanceby measuring the distance between each instance and its centroid

kmeans.inertia_  # 211.59853725816828

# Verify that inertia is the sum of the squared distances between 
# each training instance and its closest centroid

X_dist = kmeans.transform(X)
np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_]**2)  # 211.59853725816862

# score() method returns the negative inertia
kmeans.score(X)  # -211.59853725816836

# Multiple Inializations to solve the variability issue
# Select the model that minimizes the inertia
kmeans_rnd_init1.inertia_  # 219.84385402233195
kmeans_rnd_init2.inertia_  # 236.95563196978736

# Default n_init=10
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10,
                            algorithm="full", random_state=2)
kmeans_rnd_10_inits.fit(X)
plt.figure(figsize=(8,4))
plot_decision_boundaries(kmeans_rnd_10_inits, X)
plt.show()  # We end up with the original model, the optimal K-Means solution (at least in terms of inertia)


#################################################################
######################### K-Means++ #############################
#################################################################

# K-Means++ is the default initialization: init="k-means++"

KMeans()

good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
kmeans.fit(X)
kmeans.inertia_

#############################################################################
############################# Acclerated K-Means ############################
#############################################################################

# Accelerated K-Means with pythagorean theorem and triangle inequalities such that AC <= AB + BC
# Elkan does not support sparse data, for sparse data use the regular k-means algorithm "full"

# Time it
%timeit -n 50 KMeans(algorithm="elkan", random_state=42).fit(X)
%timeit -n 50 kMeans(algorithm="full", random_state=42).fit(X)


########################################################################
########################## Mini-Batch K-Means ##########################
########################################################################

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)

minibatch_kmeans.inertia_


# fetch_openML() returns a Pandas DataFrame by default, use as_frame=False to avoid this
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    mnist["data"], mnist["target"], random_state=42
)

# Write it to a memmap
filename = "my_mnist_data"
X_mm = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mm[:] = X_train

# Large data sets cannot take advanctage of memmap, so write a function to load the next batch
def load_next_batch(batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]

# Train the model by feeding it one batch at a time, 
# implement multiple initializations, 
# and keep model with the lowest inertia

np.random.seed(42)

k = 5
n_init = 10
n_iterations = 100
batch_size = 100
init_size = 500
evaluate_on_last_n_iters = 10

best_kmeans = None

for init in range(n_init):
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
    X_init = load_next_batch(init_size)
    minibatch_kmeans.partial_fit(X_init)

    minibatch_kmeans.sum_inertia_ = 0
    for iteration in range(n_iterations):
        X_batch = load_next_batch(batch_size)
        minibatch_kmeans.partial_fit(X_batch)
        if iteration >= n_iterations - evaluate_on_last_n_iters:
            minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_
    
    if (best_kmeans is None or
        minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):
        best_kmeans = minibatch_kmeans

best_kmeans.score(X)

# Time it
%timeit KMeans(n_clusters=5, random_state=42).fit(X)
%timeit MiniBatchKMeans(n_clusters=5, random_state=42)

# Minibatch is much faster than regular kmeans
# Minibatch performance is often lower (having a higher inertia)
# Plot inertia ratio and training time ratio between minibatch K-Means and regular K-Means

times = np.empty((100, 2))
inertias = np.empty((100, 200))
for k in range(1, 101):
    kmeans_ = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    print("\r{}/{}".format(k, 100), end="")
    times[k-1, 0] = timeit("kmeans_.fit(X)", number=10, globals=globals())
    times[k-1, 1] = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
    inertias[k-1, 0] = kmeans_.inertia_
    inertias[k-1, 1] = minibatch_kmeans.inertia_

plt.figure(figsize=(10,4))

plt.subplot(121)
plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Inertia", fontsize=14)
plt.legend(fontsize=14)
plt.axis=([1, 100, 0, 100])
plt.ylim([0, 100])
plt.xlim([0, 100])

plt.subplot(122)
plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), times[:, 1], "b.-", label="mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Training time (seconds)", fontsize=14)
plt.axis=([1, 100, 0, 6])
plt.ylim([0, 6])
plt.xlim([1, 100])

save_fig("minibatch_kmeans_vs_kmeans")
plt.show()


########################################################################
################### Finding Optimal Number of CLusters #################
########################################################################

kmeans_k3 = KMeans(n_clusters=3, random_state=42)
kmeans_k8 = KMeans(n_clusters=8, random_state=42)

plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")
save_fig("bad_n_clusters_plot")
plt.show()

kmeans_k3.inertia_
kmeans_k8.inertia_

# The more clusters, the closer each instance will be to its closest centroid, 
# therefore the inertia will keep getting lower as we increase k
# Plot the inertia as a function of k and analyze the results

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
            for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
            xy=(4, inertias[3]),
            xytext=(0.55, 0.55),
            textcoords='figure fraction',
            fontsize=16,
            arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.axis=([1, 8.5, 0, 1300])

# elboaw at k=4 means less clusters is bad and more clusters will not help much, 
# so k=4 is a pretty good choice, except for two clusters being grouped into one single cluster

plot_decision_boundaries(kmeans_per_k[4-1], X)
plt.show()


# silhouette coefficient = (b - a) / max(a, b); between -1 & 1, where -1 means potential assignment 
# to the wrong cluster, 0 means on a cluster boundary, and 1 means the instance is well inside it's own cluster
# a = mean intra-cluster distance (the mean distance to other instances in the same cluster)
# b = mean nearest-cluster distance (mean distance to instances of the next closest cluster, 
# defined as the one that minimizes b not including the istance's own cluster)

# Plot the silhouette score as a fnuction of k
silhouette_score(X, kmeans.labels_)

silhouette_scores = [silhouette_score(X, model.labels_)
                    for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis=([1.8, 8.5, 0.55, 0.7])
save_fig("silhouette_score_vs_k_plot")
plt.show()

plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs, facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding
    
    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")
    
    if k in (5, 6): 
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)
    
    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

save_fig("silhouette_analysis_plot")
plt.show()
########## Above plot shows k=5 as best option because all clusters are roughly the same size and 
######### they all cross the dashed line, which represents the mean silhouette score


###################### Limits of K-Means ####################










