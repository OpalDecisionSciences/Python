
################################# References ############################
# GITHUB REPO : https://github.com/jem1031/pandas-pipelines-custom-transformers
# Julie Michelman, PyData Seattle 2017 

From sklearn.linear_model import LogisticRegression
Model = LogisticRegression()

 - Fit model and predict on training data
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
p_pred_train = model.predict_proba(X_train)[:, 1]

Predict function produces a class zero or one, predict_proba produces a probability of zero and a probability of one (just take the second column, probability of one)

•	Predict on the test data, evaluate model performance 
Predict average for everything as base - a dumb model comparison as the most basic model we could build, no parameters or anything fancy, taking the mean 
p_baseline = [y_train.mean()]*len(y_test)
p_pred_test = model.predict_proba(X_test)[:, 1]

•	Measure performance on test set 
From sklearn.metrics import roc_auc_score
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



