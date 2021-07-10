
################################# References ############################

# GITHUB REPO : https://github.com/jem1031/pandas-pipelines-custom-transformers
# Julie Michelman, PyData Seattle 2017 

# Data set: 
# https://data.seattle.gov/Permitting/Special-Events-Permits/dm95-f8w5


############################################################
########################## Begin ###########################
############################################################

# Import Packages
import warnings
warnings.simplefilter(action = 'ignore', category = UserWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
import numpy as np
import pandas as pd

import re

from pandas.plotting import scatter_matrix

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler, PolynomialFeatures, FunctionTransformer
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone 

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.multiclass import OneVsOneClassifier

from sklearn.neighbors import KNeighborsClassifier

# Specify Directories
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.chdir('C:\\_data\\')
CWD = os.getcwd()
INPUT_DATA = CWD + '\\Special_Events_Permits.csv'

# Import data set
df = pd.read_csv(INPUT_DATA)

# Rename columns : camelCase is to lowerCaseFirstWordCapitalizeAllOtherWords
df = df.rename(columns={'Application Date':'applicationDate', 'Permit Status':'permitStatus', 'Permit Type':'permitType', 'Event Category':'eventCategory', 'Event Sub-Category':'eventSubCategory', 'Name of Event':'nameOfEvent', 'Year-Month-App#':'yearMonthApp', 'Event Start Date':'eventStartDate', 'Event End Date':'eventEndDate', 'Event Location - Park':'eventLocationPark', 'Event Location - Neighborhood':'eventLocationNeighborhood', 'Council District':'councilDistrict', 'Precinct':'precinct', 'Organization':'organization', 'Attendance':'attendance'})
df.columns

# Filter to 2016 
df.applicationDate = pd.to_datetime(df.applicationDate)
df = df[df.applicationDate.dt.year == 2016]

# Split into training and test data
df_train, df_test = train_test_split(df, random_state = 19)


#################################################################################
######################### Exploratory Data Analysis #############################
#################################################################################

# EDA
appDateNullCt = df_train.applicationDate.isnull().sum()
print(appDateNullCt)  # 0 Null Values for applicationDate

pStatusCnt = df_train.permitStatus.value_counts(dropna=False)
print(pStatusCnt)  # Complete vs. Not Complete 

pTypeCnt = df_train.permitType.value_counts(dropna=False)
print(pTypeCnt)  # Mostly Special Events

eCatCnt = df_train.eventCategory.value_counts(dropna=False)
print(eCatCnt)  # 79 Null values

typeCatCnt = df_train.groupby([df_train.permitType, df_train.eventCategory.isnull()]).size()
print(typeCatCnt)  # Event Category Not Null if Special Event Permit Type; other permit types have null values for event category

subCatCnt = df_train.eventSubCategory.value_counts(dropna=False)
print(subCatCnt)  # 347 Nulls

eCatSubCatCnt = df_train.groupby([df_train.eventCategory, df_train.eventSubCategory.isnull()]).size()
print(eCatSubCatCnt)  # Not Null if Athletic Special Event (permitType = Special Event, and, eventCategory = Athletic)

eventNameNullCnt = df_train.nameOfEvent.isnull().sum()
print(eventNameNullCnt)  # 0 Null Values for nameOfEvent

eventNameCnt = df_train.nameOfEvent.value_counts(dropna=False)
print(len(eventNameCnt))  # Mostly unique values

startDateNullCnt = df_train.eventStartDate.isnull().sum()
print(startDateNullCnt)  # No Null Values

endDateNullCnt = df_train.eventEndDate.isnull().sum()
print(endDateNullCnt)  # No Null Values

multiDayCnt = (df_train.eventStartDate != df_train.eventEndDate).sum()
print(multiDayCnt)  # 44 Multiday Events, 11% of data set are MultiDay Events
percentMultiDayEvents = round((multiDayCnt/len(df_train) * 100), 2)

parkCnt = df_train.eventLocationPark.value_counts(dropna=False)
print(parkCnt)  # ~90% Null Values, possible new values in test set, over 400+ parks in Seattle

neighborhoodNullCnt = df_train.eventLocationNeighborhood.isnull().sum()
print(neighborhoodNullCnt)  # 0 Null Values
neighborhoodCnt = df_train.eventLocationNeighborhood.value_counts(dropna=False)
print(neighborhoodCnt)  # Possible new values in test set
print(len(neighborhoodCnt))  # 32 - how many neighborhoods are there in Seattle?

districtCnt = df_train.councilDistrict.value_counts(dropna=False)
print(districtCnt)  # No Null Values, combinations separated by semi-colon, possible new combinations in test data set

precinctCnt = df_train.precinct.value_counts(dropna=False)
print(precinctCnt)  # No Null Values, combinations separated by semi-colon, possible new combinations in test set

organizationNullCnt = df_train.organization.isnull().sum()
print(organizationNullCnt)
organizationCnt = df_train.organization.value_counts(dropna=False)
print(organizationCnt)  # 250 different organizations, No Null Values, possible new values in test set

attendanceNullCnt = df_train.attendance.isnull().sum()
print(attendanceNullCnt)  # Few Missing Values; 4 Null Values 
attendanceStats = df_train.attendance.describe()
print(attendanceStats)  # A lot of variation
df_train.attendance.plot(kind="hist", title="Histogram of Attendance")  # Better on log scale
np.log(df_train.attendance).plot(kind="hist", title="Histogram of Log Attendance")  # Much better


###########################################################################
############################### Pre-Processing ############################
###########################################################################

# Start Fresh for Fun; use a different method for renaming/standardizing field names
preprocessDF = pd.read_csv(INPUT_DATA)

# Standardize Data : snake_case = column names to lower_case_with_underscores
def StandardizeNames(cname):
    cname = re.sub(r'[-\.]', ' ', cname)
    cname = cname.strip().lower()
    cname = re.sub(r'\s+', '_', cname)
    return cname

preprocessDF.columns = preprocessDF.columns.map(StandardizeNames)
preprocessDF.columns

preprocessDF['filter_start_date'] = pd.to_datetime(preprocessDF.event_start_date) 
df = preprocessDF[np.logical_and(preprocessDF.filter_start_date >= '2016-01-01',
                                 preprocessDF.filter_start_date <= '2016-12-31')]
df =df.drop('filter_start_date', axis=1)


###############################################################################
###################### Build a Model & Predict an Outcome #####################
###############################################################################

# Define the Outcome : Binary Classification, Complete or Not Complete
y_train = np.where(df_train.permitStatus == "Complete", 1, 0)

# Attendance is the only numeric field in the data set, set null values to 0
X_train = df_train[["attendance"]].fillna(value=0)


# Create model object
model = LogisticRegression()

# Fit model and predict on training data
# Predict() produces a class (0 or 1), predict_proba produces probability of 0 and probability of 1
# We are interested in the probability of the event happening so we take the second column
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
p_pred_train = model.predict_proba(X_train)[:, 1]

# Predict on the test data; evaluate model performance 
# Create a baseline model for comparison purposes, really simplistic, out-of-the-box
# Predict the mean for everything and use this, no parameters or anything fancy
p_baseline = [y_train.mean()]*len(y_test)
p_pred_test = model.predict_proba(X_test)[:, 1]

# Measure performance on test set 
auc_base = roc_auc_score(y_test, p_baseline)
auc_test = roc_auc_score(y_test, p_pred_test)

Transformers are for data preparation
Estimators have the predict method and are for modeling

Both have:
fit() methods find parameters from training data
transform() method applies to training or test sets

StandardScaler is a transformer
fit() finds the mean, standard deviation of each feature
transform() subtracts the mean, then divides by the standard deviation

LogisticRegression is an estimator
fit() finds coefficients in logistic regression formula
predict() plugs data into the formula to get the predicted class

Transformation steps:

From sklearn.preprocessing import (Imputer, PolynomialFeatures, StandardScaler)

Imputer() imputes missing values

Say we want to throw in some interactions, square the variables, by default the polynomial features class will do the second degree polynomials

Multiple transformers can get a bit clunky:

Instead of writing this:

X_train_imp = imputer.fit_transform(X_train_raw)
X_train_quad = quadratic.fit_transform(X_train_imp)
X_train = standardizer.fit_transform(X_train_quad)

X_test_imp = imputer.transform(X_test_raw)
X_test_quad = quadratic.transform(X-test_imp)
X-test = standardizer.transform(X_test_quad)

Put steps together in a pipeline

Pipelines work as a meta transformer where they take your whole list of transformers and apply them in a row, in order

Pipeline takes a list, in this case tuples of transformers, their alias and the actual transformer object, and then applies them in order

From skleaarn.pipeline import Pipeline

pipeline = Pipeline([
(‘imputer’, Imputer()),
(‘quadratic’, PolynomialFeatures()),
(‘standardizer’, StandardScaler())
])

Just write this to apply instead:

X_train = pipeline.fit_transform(X_train_raw)
X_test = pipeline.transform(X_test_raw)

http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html


Sometimes we want transformers to be applied in parallel; try more than one transformation and put the steps together with a FeatureUnion

From sklearn.pipeline import FeatureUnion

feature_union = FeatureUnion([
(‘fill_avg’, Imputer(strategy=‘mean’)),
(‘fill_mid’, Imputer(strategy=‘median’)),
(‘fill_freq’, Imputer(strategy=‘most_frequent’))
])

If we cannot decide which imputation strategy to use - let the machine learning model decide for us

Feature Union will put the results of these three tranformers next to each other and we end up with a wider data set

Again, they take a list of tuples with their aliases and transformer objects

Write this:

X_train = feature_union.fit_transform(X_train_raw)
X_test = feature_union.transform(X_test_raw)



…..back to the data set from above….

Looking at a data column with mostly n/a values

df.train.event_location_park.value_counts(dropna=False).head()
 - yields the count of NaN values 364 out of about 400
 - yields the count of all other values as well (parks in this case)
 - value_counts() is from pandas
 - this particular case and data set is from Seattle, and the above results in 4 parks with smaller counts per each park, because most of the data set is null values for this park attribute, and Seattle has over 400 parks, so we are sure the machine learning model is going to see some parks in our test set that we haven’t seen before, again because there are only 4 different parks in the training data set, and we need o make sure that our transformer knows how to handle that

Another interesting column is attendance, with few missing values (3), and right-skewed, binary (0 or 1) having a very high influence on the model (the predictions could get pulled towards it, in a bad way), and putting this into log scale turns this heavily skewed variable into a normal Gaussian distribution variable that will be much better feature for our use in the model

df_train.attendance.isnull().sum()
x = df_train.attendance
x.plot(kind=‘hist’,
title=‘Histogram of Attendance’)
np.log(x).plot(kind=‘hist’,
title=‘Histogram of Log Attendance’)

Use a FunctionTransformer
 - turns any function into a transformer
 - works well for stateless transformations
 - generic transformer that, if you give it a function, it will apply a function to the whole data set

From sklearn.preprocessing import FunctionTransformer
Logger = FunctionTransformer(np.log1p) 
******np.log1p adds 1 to x-values and then logs it, so 0’s won’t result in negative infinity values********
X_log = logger.transform(X)

All you have to do is create a function transformer object, pass in that function and then call transform with it

Or you can create your own transformer!!!

Create a class that will inherit from TransformerMixin; the TransformerMixin will take the fit and transform methods and automatically create the fit_transform method (convenience method)

The fit method doesn’t need to do anything because there are no parameters that the transformer needs to know from the training data

The transform method just calls the function on the data set and return it

From sklearn.base import TransformerMixin
class Log1pTransformer(TransformerMixin):

def fit(self, X, y=None):
return self

def transform(self, X):
Xlog = np.log1p(X)
return Xlog

*** This lays out the structure for creating your own transformer ******

Going back to the park column discussed above
 - the goal is to turn the string column into a set of dummy variables
 - this ‘park_name’ column gets dummy encoded with one-hot-encoder
 - Another custom transformer

The DictVectorizer creates dummy variables from string variables but does not have the input/output that we want


Create a class and create a field to hold that internal transformer; the dummy transformer

From sklearn.feature_extraction import DictVectorizer 
Class DummyTransformer(TransformerMixin):

def __init__(self):
self.dv = None

Take x data - a bunch of string columns - turning each row of the data set into a record, a mapping from the column name to the column value - this is the format the DictVectorizer wants - then we fit it and then return the transformer object
 - Convert each row to map: column name —> value
Then fit a DictVectorizer

def fit(self, X, y=None):
Xdict = X.to_dict(‘records’)
self.dv = DictVectorizer(sparse=False)
self.dv.fit(Xdict)
return self

Turn each row of data set into dictionary records, call the transform on the internal transformer, grab the feature names that it produces (in this case each event location will equal the park name, creating the column header as park name and value in cell boolean 0 or 1), pull the index from the original data set, throw those back on and turn this back into a data frame 

def transform(self, X):
Xdict = X.to_dict(‘records’)
Xt = self.dv.transform(Xdict)
cols = self.dv.get_feature_names()
Xdum = pd.DataFrame(Xt, index=X.index,
columns=cols)
return Xdum

Pandas accepts DataFrame with different data types, allows null values, and most importantly has labeled rows and columns; Scitikt-Learn expects all numeric features, usually can’t handle nulls, and casts to unlabeled numpy arrays.

Pipelines are pandas DataFrame-friendly if their component Transformers are as well; pipelines readily accept and return pandas dataframes. 
 - create custom transformers that do too

Writing custom transformers that return dataframes (with labeled columns):
 - subset columns, wrap existing transformers to reapply those labels, or some more complex logic that is not already implemented (in which case writing your own will suffice)

Custom transformer to subset columns:

class ColumnExtractor(TransformerMixin):
Def __init__(self, cols):
self.cols = cols

def fit(self, X, y=None):
return self
def transform(self, X):
Xcols = X[self.cols]
return Xcols 

Standard scaler returns numpy array, so we wrap it with a custom transformer, and create a field for that standard scaler to live in - fit method we fit fit and save it - then we transform it with an internal standard scaler and transform with the index and column names back on

class DFStandardScaler(TransformerMixin):

def __init__(self):
self.ss = None

def fit(self, X, y=None):
self.ss = StandardScaler().fit(X)
return self

def transform(self, X):
Xss = self.ss.transform(X)
Xscaled = pd.DataFrame(Xss, index=X.index, 
columns=X.columns)
return Xscaled

Putting it all together: Feature Engineering

CAT_FEATS = [
‘permit_type’, ‘event_category’,
‘event_sub_category’, 
‘event_location_park’,
‘event_location_neighborhood’]
NUM_FEATS = [‘attendance’]

Feature Engineering Pipeline
 - One hot encoding for categorical features
 - LOOK UP CODE FOR DATAFRAME FRIENDLY FEATURE UNION IN GITHUB REPO! (DFFeatureUnion and other transformers - website below)
 - Log attendance then standardize everything

pipeline = Pipeline([
(‘features’, DFFeatureUnion([
(‘categoricals, Pipeline([
(‘extract’, ColumnExtractor(CAT_FEATS)),
(‘dummy’, DummyTransformer())
])),
(‘numerics’, Pipeline([
(‘extract’, ColumnExtractor(NUM_FEATS)),
(‘zero_fill’, ZeroFillTransformer()),
(‘log’, Log1pTransformer())
]))

])),
(‘scale’, DFStandardScaler())
])


Now that the transformation process is wrapped up in the pipeline above, we can apply the pipeline to our training/test sets

X_train = pipeline.fit_transform(df_train)
X_test = pipeline,transform(df_test)

Fit Model:

Model = LogisticRegression()
Model.fit(X_train, y_train)

Predict on test data:

p_pred_test = model.predict_proba(X_test)[:, 1]

Measure performance on test data:

auc_test = roc_auc_score(y_test, p_pred_test)



# Feature Engineer
df['eventDuration'] = endDate - startDate



########################### Collection of Transformers ######################
# From Github repo

Class DFFunctionTransformer(TransformerMixin):
# FunctionTransformer but for pandas DataFrames
def __init__(self, *args, **Kwargs):
self.ft = FunctionTransformer(*args, **Kwargs)

def fit(self, X, y=None):
#stateless transformer
return self

def transform(self, X):
Xt = self.ft.transform(X)
Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns)
return Xt


Class DFFeatureUnion(TransformerMixin):
# FeatureUnion but for pandas DataFrames

def __init__(self, transformer_list):
self.transformer_list = transformer_list

def fit(self, X, y=None):
for (name, t) in self.transformer_list:
t.fit(X, y)
return self

def transform(self, X):
# assumes X is a DataFrame
Xts = [t.transform(X) for _, t in self.transformer_list]
Xunion = reduce(lambda X1, X2: pd,merge(X1, X2, left_index=True, right_index=True), Xts)
return Xunion


Class DFImputer(TransformerMixin):
# Imputer but for pandas DataFrames

def __init__(self, strategy=‘mean’):
self.strategy = strategy
self.imp = None
self.statistics_ = None

def fit(self, X, y=None):
self.imp = Imputer(strategy=self.strategy)
self.imp.fit(X)
self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
return self

def transform(self, X):
#assumes X is a DataFrame
Ximp = self.imp.transform(X)
Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
return Xfilled


classDFStandardScaler(TransformerMixin):
# StandardScaler but for pandas DataFrames

def __init__(self):
self.ss = None
self.mean_ = None
self.scale_ = None

def fit(self, X, y=None):
self.ss = StandardScaler()
self.ss.fit(X)
self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
return self

def transform(self, X):
#assumes X is a DataFrame
Xss = self.ss.transform(X)
Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
return Xscaled


Class DFRobustScaler(TransformerMixin):
# RobustScaler but for pandas DataFrame

def __init__(self):
self.rs = None
self.center_ = None
self.scale_ = None

def fit(self, X, y=None):
self.rs - RobustScaler()
self.rs.fit(X)
self.center_ = pd.Series(self.rs.center_, index=X.columns)
self.scale_ = pd.Series(self.rs.scale_, index=X.columns)

def transform(self, X):
# assumes X is a DataFrame
Xrs = self.rs.transform(X)
Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
return Xscaled


Class ColumnExtractor(TransformerMixin):

def __init__(self, cols):
self.cols = cols

def fit(self, X, y=None):
# Stateless transformer
return self

def transform(self, X):
# Assumes X is a DataFrame
Xcols = X[self.cols]
return Xcols


class ZeroFillTransformer(TransformerMixin):

def fit(self, X, y=None):
# stateless transformer
return self

def transform(self, X):
# assumes X is a DataFrame
Xz = X.fillna(value=0)
return Xz


class Log1PTransformer(TransformerMixin):
def fit(self, X, y=None):
#stateless tranformer
return self

def transform(self, X):
# assumes X is a DataFrame
Xlog = np.log1p(X)
return Xlog


class DatFormatter(TransformerMixin):
def fit(self, X, y=None):
# stateless transformer
return self

def transform(self, X):
# assumes X is a DataFrame
Xdate = X.apply(pd.to_datetime)
return Xdate 


class DateDiffer(TransformerMixin):
def fit(self, X, y=None):
# stateless transformer
return self

def transform(self, X):
# assumes X is a DataFrame
beg_cols = X.colums[:-1]
end_cols = X.columns[1:]
Xbeg = X[beg_cols].as_matrix()
Xend = X[end_cols].as_matrix()
Xd = (Xend - Xbeg) / np.timedelta64(1, ‘D’)
diff_cols = [‘—>’.join(pair) for pair in zip(beg_cols, end_cols)]
Xdiff = pd.DataFrame(Xd, index=X.index, columns=diff_cols)
return Xdiff 


class DummyTransformer(TransformerMixin):
def __init__(self:
self.dv = None

def fit(self, X, y=None):
#assumes all columns of X are strings
Xdict = X.to_dict(‘records’)
self.dv = DictVectorizer(sparse=False)
self.dv.fit(Xdict)
return self

def transform(self, X):
# assumes X is a DataFrame
Xdict = X.to_dict(‘records’)
Xt = self.dv.transform(Xdict)
cols = self.dv.get_feature_names()
Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
# drop column indicating NaNs
nan_cols = [c for c in cols if ‘=‘ not in c]
Xdum = Xdum.drop(nan_cols, axis=1)
return Xdum

class MultiEncoder(TransformerMixin):
# Multiple-column MultiLabelBinarizer for pandas DataFrames

def __init__(self, sep=‘, ‘):
self.sep = sep
self.mlbs = None

def _col_transform(self, x, mlb):
cols = [‘’.join([x.name, ‘=‘, c]) for c in mlb.classes_]
xmlb = mlb.transform(x)  
xdf = pd.DataFrame(xmlb, index=x.index, columns=cols)
return xdf

def fit(self, X, y=None):
Xsplit = X.applymap(lambda x: x.split(self.sep))
self.mlbs = [MultiLabelBinarizer().fit(Xsplit[c]) for c in X.columns]
return self

def transfomr(self, X):
# assumes X is a DataFrame
Xsplit = X.applymap(lambda x: x.split(self.sep))
Xmlbs = [self._col_transform(Xsplit[c], self.mlbs[i]) 
for i, c in enumerate(X.columns)]
Xunion = reduce(lambda X1, X2, left_index=True, right_index=True), Xmlbs)
return Xunion


class StringTransformer(TransformerMixin):
def fit(self, X, y=None):
# stateless transformer
return self

def transform(self, X):
# assumes X is a DatFrame
Xstr = X.applymap(str)
return Xstr


class ClipTransformer(TransformerMixin):
def __init__(self, a_min, a_max):
self.a_min = a_min
self.a_max = a_max

def fit(self, X, y=None):
# stateless transformer
return self

def transform(self, X):
# assumes X is a DataFrame
Xclip = np.clip(X, self.a_min, self.a_max)
 
return Xclip


class AddConstantTransformer(TransformerMxin):
def __init__(self, c=1):
self.c = c

def fit(self, X, y=None):
 
# stateless transformer
return self

def transform(self, X):
# assumes X is a DataFrame
Xc = X + self.c
return Xc


