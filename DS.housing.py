# Reference: 'Hands-On Machine Learning with SciKit Learn & TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems' First Edition; Geron, Aurelien; O'Reilly, 2017
import warnings
warnings.simplefilter(action = 'ignore', category = UserWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
from six.moves import urllib

import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix

import hashlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction import DictVectorizer


BASE_PATH = '\\\\lgamsfs04\\\\fpanda\\bi\\'
DATA_DIR = BASE_PATH + 'data\\'
INPUT_DATA = 'data.csv'

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_input_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


#################### Get a general understanding of the data #####################
### EDA ###

fetch_housing_data()
data = load_input_data()

# View top 5 rowa
data.head()

# Data description
data.info()

# Categorical columns category counts
data['ocean_proximity'].value_counts()

# Numerical attributes - count, mean, standard deviation, min/max, perentiles
data.describe()


# Histograms for each numerical attribute
#plt.style.use('classic') 
#plt.show()

# import matplotlib.pyplot as plt
data.hist(bins = 50, figsize = (20, 15))



####################### Differing Methods for Random Sampling #####################

# Create test set
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    # using the above method will generate a different test set when you run the program again; one solution is, you can save the test set on the first run and load it in subsequent runs
    # another solution is np.random.seed(19) before calling permutation, which will always generate the same shuffled indices
    # Both of these solutions will break when fetching an updated data set; important to maintain consistency in test set instances, making sure the model has not seen any records that have been used in the training set, for illiminating bias 

train_set, test_set = split_train_test(data, 0.2)
print(len(train_set), "train + ", len(test_set), "test")

# Another method to ensure test set remains consistent across multiple runs, even if you refresh the data set; this test set will contain 20% of new instances but will not contain any instance that was previously used in the training set. c
# Create a unique and immutable hash of each instance's identifier, keep only the last byte of the hash, and place instance in test set if this value is lower or equal to 51 (~ 20% of 256)

# import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

# If using row index as unique identifier, make sure new data gets appended to the end of the data set, and no rows ever get deleted; if this is not possible use another more stable features to build a unique identifier
data_with_id = data.reset_index()
train_set, test_set = split_train_test_by_id(data_with_id, 0.2, "index")

# Using longitude and latitude are stable features, however, level of detail becomes increasingly important, depending on whether this longitude/latitudes are referencing districts, where many records will have the same values, placing them in the same test or train data set, and adding bias; whereas a level of detail pertianing to a specific address would not.
data_with_id["id"] = data["longitude"] * 1000 + data["latitude"]
train_set, test_set = split_train_test_by_id(data_with_id, 0.2, "id")

# Random state parameter and multiple data sets with same number rows split on same indices, such as separate data frames for labels

# from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 19)

############### Stratified Sampling ################

# Have  sufficient number of instances in each stratum so that the the estimate of the stratum's importance is not biased; should not have too many stratum
# For a continuous numerical attribute, create a new categorical attribute by dividing by 1.5 to limit the number of strata (or categories) and rounding up (to create discrete categories) and then merging all the categories greater than 5 into category 5, then do stratified sampling on the categories
data["income_cat"] = np.ceil(data["median_income"] / 1.5)
data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace = True)

# from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 19)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# Check categorical proportions 
data["income_cat"].value_counts() / len(data)
# Using the line of code above, check proportions comparing test sets (purely random sampling vs. stratified sampling) against the full data set (above) to see that stratified is likely identical or really close to full data set, whereas random sampling will likely be skewed

# Remove new_discrete_categorical attribute for original data
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)



#################### EDA: Exploratory Data Analysis to Gain Insights ####################

# Do not explore the test set! 
# Only explore the training set
# If training set is large take a sample for ease and speed of manipulations
# Make a copy of the training set to keep its integrity

data = strat_train_set.copy()

# For geographical information
data.plot(kind = "scatter", x = "longitude", y = "latitude")

# Same code with alpha to visualize high density data points (areas)
data.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)

data.plot(kind = "scatter", x="longitude", y="latitude", alpha=0.4, 
    s=data["population"]/100, label="population", figsize=(10,7), 
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
# Additionl visual parameters: s = radius of each circle on scatterplot, c = color (cmap = predefined color map) called "jet" (ranges from blue = low values, to red = high values)

#data.plot(kind="scatter", x = "longitude", y = "latitude", alpha = 0.4, s = data["numerical vallue attribute"]/100, label = data["numerical value attribute"], figsize = (10, 7), c = "numerical value attribute", cmap = plt.get_cmap("jet"), colorbar = True,)

#plt.legend()
# If the graph exhibits a close relationship between the attributes used, it will probably be useful to use a clustering algorithm to detect the main clusters and add new features that measure geographical proximity to the cluster centers; moutains, desert, ocean, lakes, highways (small scale = shopping centers, to large scale = climates)

# calculate the standard correlation coefficient, pearson's r, between every pair of attributes; only measures linea correlations
corr_matrix = data.corr()
corr_matrix

# Look at how much each attribute correlates with your target variable; 1 = strong positive correlation, -1 = strong negative correlation, 0 = no linear correlation (this does not mean that the attributes' axes are independent; there may be a non-linear pattern)
corr_matrix["median_house_value"].sort_values(ascending = False)

# Plot every numerical attribute against every numerical attribute; if there are 11 numerical attributes you will get 11^2 scatterplots, so nest to focus on most important ones; pandas does not plot each variable against itself, and instead plots a histogram of each attribute in the top left to bottom right diagonal

# from pandas.tools.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(data[attributes], figsize = (12, 8))

# Zoom in on the most promising attrubites
data.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)


# Based on your data, try to experiment with attribute combinations, in an effort to achieve new data points with higher correlation values than the original data fields; before preparing the data for ML algorithm, and after analyzing your first output to gain even more insights, you can circle back to this exploration step and repeat
data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"]/data["households"]

corr_matrix = data.corr()
corr_matrix
corr_matrix["median_house_value"].sort_values(ascending = False)

############### Prepare Data for ML Algorithms ##################

# Separate predictors and labels
data = strat_train_set.drop("median_house_value", axis = 1)
data_labels = strat_train_set["median_house_value"].copy()

# Data Cleaning

# Missing values
data.dropna(subset=["total_bedrooms"]) # Remove corresponding values
data.drop("total_bedrooms", axis = 1) # Remove the attribute
median = data["total_bedrooms"].median() # Calculate the median (or mean, or mode)
data["total_bedrooms"].fillna(median, inplace = True) # Replace null values with median value (mean or mode)
# If you use the median value to replace an attribute's nulls, save the value for later use in replacing refreshed data set or replacing missing values in the test set for evaluating the system

# from sklearn.preprocessing import Imputer

imputer = SimpleImputer(strategy = "median") # Only for numerical values; imputer is an estimator, strategy is a hyperparameter, data set = parameter, fit() performs the estimation
data_numerical_values_set = data.drop("ocean_proximity", axis = 1) # Copy data without text attributes
imputer.fit(data_numerical_values_set) # fit inmputer instance to training data numerical values
# Results stored in imputer.statistics_
# data_numerical_values_set.median().data_numerical_values_set

imputer.statistics_
data_numerical_values_set.median().values
# use imputer to transform data set nulls with calculated values, returns numpy array
X = imputer.transform(data_numerical_values_set)

# put np array into pd dataframe
data_tr = pd.DataFrame(X, columns = data_numerical_values_set.columns)

# Scikit learn estimators estimate parameters based on data set, predictors predict, score() measures quality of predictions
# Transformers --> fit_transform() runs faster than fit() and transform(); optimized, faster

# Text and categorical attributes

# from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data_cat_values_set = data["ocean_proximity"]
data_cat_encoded = encoder.fit_transform(data_cat_values_set)
data_cat_encoded # This yields an array and assumes nearby values are more similar than distant values (0 and 1 are more similar opposed to 1 and 4, which may not be the case)

print(encoder.classes_) # look at categorical mapping

# Alternate option: one hot encoding --> to create one binary attribute per category (0 = cold, 1 = hot) 
# Convert integer categorical values into one-hoy vectors
# data_cat_values_set is 1-dimensional array, fit_transform() expects a 2-dimensional array
# Reshape data_cat_values_set

# from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
data_cat_1hot = encoder.fit_transform(data_cat_encoded.reshape(-1, 1)) # NumPy reshape() allows 1-dimension to be -1, which means unspecified, where the value is inferred from the length of the array and the remaining dimensions
data_cat_1hot # Outputs a SciPy sparse matrix instead of np array; which only stores the location of the non-zero elements, saving tons of memory

# You can call the toarray() method if you want a dense NumPy array
data_cat_1hot.toarray()

# Apply both transformations from text categories to integer categories, then from integer categories to one-hot vectors using LabelBinarizer

# from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer() 
# You can pass "sparse_output = True" to LabelBinarizer constructor for a sparse matrix as output opposed to a NumPy array, otherwise, note that the output is a dense NumPy array
data_cat_1hot = encoder.fit_transform(data_cat_values_set)
data_cat_1hot

# Custom cleanup operations and combining specific attributes requires a custom transformer; this needs to work seamlessly with cikit-Learn functionality (such as pipelines) which reliy on ducktyping not inheritance
# Create a class and implement 3 methods: fit(), transform(), and fit_transform() - add TransformerMixin as a base class for the third (and avoid *args and **kargs in constructor) - add BaseEstimator as a base class for two extra methods (get_params() and set_params()) for automatic hyperparameter tuning

# from sklearn.base import BaseEstimator, TransformerMixin

# For housing data set: page 65 
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
 
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # No *args or **Kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:,household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
data_extra_attribs = attr_adder.transform(data.values)

# Feature Scaling
# Not required to scale target values, but ml algorithms don't perform well with input numerical attributes having very different scales
# Two common methods for getting all attributes to have the same scale: min-max (normalization) and standarization

# min-max (normalization) shifts/rescales values from 0-1: [(x-Value(i) - min value) / (max - min)] 
# Scikit-Learn has MinMaxScaler transformer with hyperparameter feature_range that allows you to change range from 0-1 to something else 

# standardization does not bound values to 0-1 range, which is not preferred by neural networks (NNs expect 0-1 range), but this method is much less affected by outliers
# transformer is called StandardScaler
# standardization [(x-Value(i) - mean value)/variance]: resulting distribution has a unit variance 

# Be sure to fit the scalers to the training data only; not the full data set

########## Transformation Pipeline ############

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

data_num_tr = num_pipeline.fit_transform(data_numerical_values_set)

# Create pipeline to take Pandas DataFRame, pick only numerical attributes and automate preprocessing steps from above
# repeat this for categorical variables 
# page 67

# Scikit-learn has changed 
# For fix: https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb
# https://github.com/ageron/handson-ml/issues/388

# from sklearn.base import BaseEstimator, TransformerMixin
'''
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values


class catDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return self.encoder.transform(X)
'''


# Look into ColumnTransformer class (Pull Request #3886) for easy attribute-specific transformations 
# pip3 install sklearn-pandas to get DataFrameMapper class 9similar objective0

num_attribs = list(data_numerical_values_set)
cat_attribs = data["ocean_proximity"]

###Testing line 363 of code
c = DataFrameSelector(cat_attribs, sparse_output = True)

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer()),
])

# Join these pipelines

# from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

data_prepared = full_pipeline.fit_transform(data)
data_prepared


################ Select and Train a Model ###################

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(data_prepared, data_labels)

# Try out linear regression on a few instances of training set
some_data = data.iloc[:5]
some_labels = data_labels.iloc[:5]

# Compare predictions to labels for accuracy
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# RMSE (root mean squared error) : Evaluate model on training set
# from sklearn.metrics import mean_squared_error

data_predictions_lin_r = lin_reg.predict(data_prepared)
lin_mse = mean_squared_error(data_labels, data_predictions_lin_r)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# Build another model #

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, data_labels)

# Evaluate model on training set
data_predictions_dec_tr_r = tree_reg.predict(data_prepared)
tree_mse = mean_squared_error(data_labels, data_predictions_dec_tr_r)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# Evaluate Decision Tree Regression 
# train_test_split, run smaller training sets and reevaluate against validation set


##### K-fold Cross-Validation ######
# random split dataset into 10 distinct subsets (folds), trains decision tree reg model (10 times), and evaluates model (10 times) using a different fold for evaluation and training on the other 9 each time
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, data_prepared, data_labels, scoring = "neg_mean_squared_error", cv = 10)

# sklearn cross val expects utility fnx (greater is better) opposed to a cost fnx (lower is better), hence scoring fnx is opposite mse (negative value; '-score' is computed before sqrt)
tree_rmse_scores = np.sqrt(-scores)

# display results
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# Compute same for lin_reg
lin_scores = cross_val_score(lin_reg, data_prepared, data_labels, scoring = "neg_mean_squared_error", cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


### Build another Model ####

# Random Forest Regressor
# Random Forests train many Decision Trees on random subsets of the features, then averages out their predictions
# Ensemble Learning: building a model on top of (many) other models

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit()(data_prepared, data_labels)
forest_predictions = forest_reg.predict(data_prepared)
forest_mse = mean_squared_error(data_labels, forest_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

forest_scores = cross_val_score(forest_reg, data_prepared, data_labels, scoring="neg_mean_squared_errror", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


'''
#you can save models, trained parameters, hyperparameters,cross-validation scores, and actual predictions
# --> for the purposes of comparing model types and scores, types of errors like overfitting/underfitting, etc
# Python's pickle module does this but Scikit-Learn's is omre efficient at serializing large NumPy arrays

from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl")

# To load later
my_model_loaded = joblib.load("my_model.pkl")
'''

# Scitkit-Learn searches for the best set of hyperparameter value combinations  via cross-validation
# apply to random forest regressor

# from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
    {'bootstrap':[False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
# param_grid evaluates all 3 X 4 = 12 combinations of n_estimators and max_features hyperparameter values in the first dict, then tries all 2 X 3 = 6 hyperparameter values in the second dict, but this time with hyperparameter bootstrap set to false
# all in all this grid_search explores 12 + 6 = 18 combinations of random forest regressor hyperparameter values, and trains each model 5 times (we're using 5-fold cross validation)
# total 18 X 5 = 90 rounds of training

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(data_prepared, data_labels)

grid_search.best_params_ 
# These results are greatest combination of parameters evaluated in 90 training sessions, and the best results are the largest number parameters evaluated, so we attempt larger values/combinations to see if we can improve

# Try in consecutive powers of ten if you don't know what the hyperparameters should be
param_grid = [
    {'n_estimators':[30, 45, 60], 'max_features':[8, 10, 12, 14]},
    {'bootstrap':[False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = 'neg_mean_squared_error')
grid_search.fit(data_prepared, data_labels)
'''
fill in hyperparameter results from above here
'''

# We can also get the best estimator
grid_search.best_estimator_
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features=8, max_leaf_nodes=None, min_impurity_split=1e-07,
            min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1, 
            oob_score=False, random_state=19, verbose=0, warm_start=False)
# Grid_search is initialized with 'refit=True' as default setting, meaning that once it finds the best estimator using cross-validation, it retrains on the whole training set (and feeding it more data will likely improve performance)

# Evaluation scores - page 74
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Grid_search is great for fewer combinations, RndomizedSearchCV is better when the hyperparameters search space/combination is large
# RandomizedSearchCV operates similarly to gridSearch but selects random values for hyperparameters during each iteration 

# Ensemble methods are great to fine tune your models as well, especially if te best models you are blending make different types of errors
# Analyze best models and their erros to proceed

# RandomForestRegressor can indicate the relative importance of each attribute for making accurate predictions; compare this to the corr_matrix you ran in data preprocessing section for before and after retrospection

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# Display the above results next to their attribute names
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
# look at the system erros and try to understand where these errors are stemming from, then make adjustments; remove less usefull or uninformative features, clean up outliers etc

# evaluate final model on test set
# get predictors and lables from test set, run full_pipeline to transform the data - call transform() not fit_transform()

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# Present solution, what you learned, what worked and what did not work, assumptions made, and system limitations
# document everything, in a clear and concise presentation, with data visualizations and easy-to-remember statements


''' 
Implement a support vector regreesor using different kernels, comparing linear (varying values for C hyperparameter) and rbf (varying values for C and gamma hyperparameters)
Do a compare between GridSearchCV and RandomizedSearchCV
Add transformer in preparation pipeline to select only the most important attributes
Create a single pipeline that does the full data preparation plus the final prediction
Automatically explore some preparation options using GridSearchCV
'''