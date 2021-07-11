

# References: 'Hands-On Machine Learning with SciKit Learn & TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems' First Edition; Geron, Aurelien; O'Reilly, 2017


import warnings
warnings.simplefilter(action = 'ignore', category = UserWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

from sklearn.preprocessing import  StandardScaler
 
from sklearn.base import BaseEstimator
from sklearn.base import clone 

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

from sklearn.multiclass import OneVsOneClassifier

from sklearn.neighbors import KNeighborsClassifier

BASE_PATH = '\\\\lgamsfs04\\\\fpanda\\bi\\'
DATA_DIR = BASE_PATH + 'data\\'
INPUT_DATA = 'data.csv'

# Fetch the data set
mnist = fetch_openml('mnist_784')
mnist

# Look at array shapes
X, y = mnist["data"], mnist["target"]

X.shape # (70,000 images,each image has 784 features - because each image is 28 X 28 pixels, each feature represents one pixel's intensity from 0 (white) to 255 (black))
y.shape # (70,000 labels - dependent variable)

# Look at one digit from the dataset
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()

# Look at the label of some_digit
y[36000]

# Split into train/test sets (mnist is already setup - first 60,000 records are training set, last 10,000 are test set)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Using fetch_openml returns labels as strings; cast to int 8
'''
X_train = X_train.astype(np.int8)
X_test = X_test.astype(np.int8)
y_test = y_test.astype(np.int8)
'''
y_train = y_train.astype(np.int8)


# Shuffle training set to ensure cross-validation folds aren't missing some digits, and also, to prevent the sensitivity of some algorithms having ordered training instances (too many similar digits in a row can inhibit performance)
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Create a binary classifier for "is 9" and "not 9" - target vectors

y_train_9 = (y_train == 9)
y_test_9 = (y_test == 9)
np.unique(y_train_9)

# SGDClassifier handles very large data sets efficiently; trains instances independently, good for online learning
sgd_clf = SGDClassifier(random_state=42) # set random state parameter for reproducible results
sgd_clf.fit(X_train, y_train_9)

# detect images of #9
sgd_clf.predict([some_digit])

# Implementing cross validation - Compare custom cross validation to out-of-the-box from sklearn 

# For added control over the cross validation process, create your own cross validation
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_9):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_9[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_9[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

cross_val_score(sgd_clf, X_train, y_train_9, cv=3, scoring="accuracy")

# Create a dumb classifier that classifies everything to our "not 9" class
class Never9Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_9_clf = Never9Classifier()
cross_val_score(never_9_clf, X_train, y_train_9, cv=3, scoring="accuracy")

'''
90% accuracy because about 10% of the images are 9s, so always guessing an image is not a 9 yields 90% accuracy
This demonstrates why accuracy is not a preferred performance measure for classifiers, and also for skewed data sets
Confusion matrix is a better alternative for evaluating the performance of classifiers'''

# Using cross_val_predict yields a clean prediction for each record from a model that hasn't seen the data during the training process
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3)

# Confusion matrix - each row represents an actual class and each column represents a predicted class
# upper row = negative class (not 9s); upper left = negative class correctly classified (true negatives); upper right = negative class incorrectly classified as 9s (false positives)
# second row = positive class (is 9); bottom left = wrongly classified as non-9s (false negatives); bottom right = correctly classified as 9s (true positives)
confusion_matrix(y_train_9, y_train_pred)

# Accuracy of the positive predicitions - precision of the classifier 
# precision = true positive / (true poitive + false positive)
# Precision is often used with recall
# Recall = sensitivity = true positive rate
# recall = true positive / (true positive + false negative)
precision_score(y_train_9, y_train_pred)  # 4752 / 4752 + 1986 = 70.5%
recall_score(y_train_9, y_train_pred)  # 4752 / 4752 + 1197 = 79.87%

# F1 score is the harmonic mean of precision and recall; this value will only be high if both precision and recall are high
# F1 = 2 X [(precision X recall) / (precision + recall)]
f1_score(y_train_9, y_train_pred)  # 74.9%

# precision/recall tradeoff and decision threshold
# scikit-learn does not let you set the decision threshold but it allows you access to the decision scores that it uses to make predictions
# we called SGDClassifier's predict funx on line 100
# Now we call it's 'decision_function()' method for a score on each instance, then we make predictions using this score and any threshold we want
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0 # mimicking threshold of SGDClassifier, which uses a threshold of 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

# raising threshold typically decreases recall
threshold = 200000
y_some_digit_pred =(y_scores > threshold)
y_some_digit_pred

# determine which threshold to use; obtain scores on all records
y_scores = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3, method="decision_function")

# Compute precision and recall for all thresholds
precisions, recalls, thresholds = precision_recall_curve(y_train_9, y_scores)

# Plot precision and recall as functions of the threshold value
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

# Precision/recall curve - when positive class is rare, or when false positive are most important than false negatives, otherwise use ROC curve
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

plt.plot(precisions[:-1], recalls[:-1], "g-")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper left")
plt.ylim([0, 1])
plt.show()

# ROC Curve = receiver operating characteristic - when false negatives are more important than false positives
# roc = true positive rate vs false positive rate = sensitivity (recall) vs 1 - specificity (True Negative Rate)
# calculate TPR, FPR for varying thresholds 
fpr, tpr, thresholds = roc_curve(y_train_9, y_scores)

# Plot fpr vs tpr - this looks good because there are few positives compared to negatives
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

roc_auc_score(y_train_9, y_scores)

''' Fix Me!
# Create a Random Forest Classifier to compare SGD Classifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_9, cv=3, method="predict_proba")

# Use positive class's probability as the score, because roc requires scores not probabilities
y_scores_forest = y_probas_forest[:1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_9, y_scores_forest) 

# Plot ROC curve compared to original
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_9, y_scores_forest)
'''

######## Create a multi-class classifier #########
######## Create a multinomial (multiclass) classifier #########
# Random Forest and Naive Bayes can handle multiple classes directly; linear and svm are strictly binary
# For multi-class classification Scikit-Learn automatically runs OvA, except for SVM classifiers which it runs OvO
# OvA strategy: one-versus-all - train 10 binary classifiers, one for each digit, classify an image and select the class whose classifier outputs the highest decision score
# OvO: one-versus-one - train one binary classifier for every pair of digits, 0's and 1's, 0's and 2's, for N classes, there are N X (N-1)/2 classifiers (45 for MNIST); disadvantage is training 45 models, advanctage is training each model only on the two classes that it must distinguish

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

# to see that scikit-learn actually trained 10 binary classifiers, call the decision_function() method, retunring 10 scores, 1 per class
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores

np.argmax(some_digit_scores) # model guesses wrong - it's supposed to be a 9

# when a classifier is trained the target classes are stored in 'classes_' attribute ordered by value
sgd_clf.classes_

# Force sklearn to use OvA or OvO - use OneVsOneClassifier or OneVsRestClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])

len(ovo_clf.estimators_) # 45 estimators

# train RandomForestClassifier - sklearn did not have to run OvA or OvO because rfc can directly classify instances into multiple classes
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])

# call predict_proba() to get list of probabilitiesthat rfc assigned to each instance for each class - you can see 9 is pretty confident at 0.86 
forest_clf.predict_proba([some_digit]) # 86% for a 9, 9% for a 4, 3% for a 5

#Evaluate
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy") # 87.8%, 89.2%, 86.8%

# Improve - scaling inputs increases accuracy
scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_trained_scaled, y_train, cv=3, scoring="accuracy") # 90.5%, 89.8%, 90.1%



# Imagine you already have a great model and you want to proceed with that
############################ Error Analysis #########################
y_train_pred = cross_val_predict(sgd_clf, X_trained_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# look at an image representation
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# look at plot errors - divide each value by number images in class - slightly darker could mean there are fewer images in this class/data set or that the classifier does not perform well on that class
row_sums = conf_mx.sum(axis = 1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# fill diagonal with zeros to keep only errors, plot result - rows represent classes, columns represent predicted classes
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# To Improve: we need to improve predicted class 8; we could gather more training data
# you can write an algorithm to help you count the number of closed loops and use this for feature engineering to aid the classifier
# you could also preprocess the images using Scikit-Image, Pillow, or OpenCV to make some patterns stand out more than others (i.e. closed loops)

# Analyzing individual errors - as confusion matrix

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    imges_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(8, 8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()


# It is important to note that we udes a simple SGDClassifier - a linear model that assigns a weight per class 
# to each pixel - for new images the model sums the value of the weighted pixel intensities to get a score for 
# each class - this can explain a model's confusion in similar digits that only differ by a few pixels
# Models like this can benefit from preprocessing images, minimizing shifts/rotations, ensuring images 
# are centered nd upright, to avoid similar digit confusion



######## Multilabel Classification  Systems - Outputs mutlipe binary labels ###########
# Pretend you're doing facial recognition and you want to label all the faces your model recognizes in one image
# Before facial recognition, lets output two labels for each digit image
y_train_large = (y_train >= 7)  # large digit greater than 7
y_train_odd = (y_train % 2 == 1)  # odd digit
y_multilabel = np.c_[y_train_large, y_train_odd]  #array containing two target labels for each image

# KNeighborsClassifier() supports multilabel classification; not all classifiers do 
# Train it using multiple targets array
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])  # True, True : because 9 is greater than 7, and 9 is odd

# Evaluating multilabel classifiers
# all other binary metrics discussed above may apply
# measure individual F1 scores and avergae across all labels, assuming all labels are equally important

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1_score(y_train, y_train_knn_predict, average="macro")

# To give a weight equal to the support of each digit's class; i.e. the number of instances with that target label
f1_score(y_train, y_train_knn_predict, average="weighted")


############## Multioutput Classification = Multioutput multiclass classification ################

# Build a system removing noise from images : imput a noisy digit image, output a clean digit image
# The classifiers output is multilabel(one label per pixel) and each label can have multiple values (pixel intensities rane from 0 to 255)
# You can also have systems that output multiple labels per instance including both class labels and value labels
# multioutput systems are not limited to classification tasks; perhaps predicting pixel intensity is better suited to regression

# Take MNIST images, add noise to pixel intensities using Numpy's randint()
# Target images will be the original images

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0 ,100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# Not supposed to peak at test set, but good to check work in this scenario - ensure setup is correct
X_test_mod[some_digit]
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)

