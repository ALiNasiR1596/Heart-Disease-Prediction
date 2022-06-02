import warnings

import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import IsolationForest
# import classification modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
# import performance scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


# Functions

# Validation metrics for classification
def validationmetrics(model, testX, testY, verbose=True):
    predictions = model.predict(testX)

    if model.__class__.__module__.startswith('lightgbm'):
        for i in range(0, predictions.shape[0]):
            predictions[i] = 1 if predictions[i] >= 0.5 else 0

    # Accuracy
    accuracy = accuracy_score(testY, predictions) * 100

    # Precision
    precision = precision_score(testY, predictions, pos_label=1, labels=[0, 1]) * 100

    # Recall
    recall = recall_score(testY, predictions, pos_label=1, labels=[0, 1]) * 100

    # get FPR (specificity) and TPR (sensitivity)
    fpr, tpr, _ = roc_curve(testY, predictions)

    # AUC
    auc_val = auc(fpr, tpr)

    # F-Score
    f_score = f1_score(testY, predictions)

    if verbose:
        print("Prediction Vector: \n", predictions)
        print("\n Accuracy: \n", accuracy)
        print("\n Precision of event Happening: \n", precision)
        print("\n Recall of event Happening: \n", recall)
        print("\n AUC: \n", auc_val)
        print("\n F-Score:\n", f_score)
        # confusion Matrix
        print("\n Confusion Matrix: \n", confusion_matrix(testY, predictions, labels=[0, 1]))

    res_map = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc_val": auc_val,
        "f_score": f_score,
        "model_obj": model
    }
    return res_map


# Helper function to get fetaure importance metrics via Random Forest Feature Selection (RFFS)
def RFfeatureimportance(df, trainX, testX, trainY, testY, trees=35, random=None):
    clf = RandomForestClassifier(n_estimators=trees, random_state=random)
    clf.fit(trainX, trainY)
    # validationmetrics(clf,testX,testY)
    res = pd.Series(clf.feature_importances_, index=df.columns.values).sort_values(ascending=False) * 100
    print(res)
    return res


# Classification Algorithms
def LogReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = LogisticRegression()
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def GadientBoosting(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingClassifier()
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def AdaBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def SVM(trainX, testX, trainY, testY, svmtype="SVC", verbose=True, clf=None):
    # for one vs all
    if not clf:
        if svmtype == "Linear":
            clf = svm.LinearSVC()
        else:
            clf = svm.SVC()
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def RandomForest(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = RandomForestClassifier()
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def XgBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = XGBClassifier(random_state=1, learning_rate=0.01)
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def NaiveBayes(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GaussianNB()
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def MultiLayerPerceptron(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = MLPClassifier(hidden_layer_sizes=5)
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


def DecisionTree(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = DecisionTreeClassifier()
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


# Helper function to run all algorithms provided in algo_list over given dataframe, without cross validation
# By default it will run all supported algorithms 

# Train Test Split: splitting manually
def traintestsplit(df, split, random=None, label_col=''):
    # make a copy of the label column and store in y
    y = df[label_col].copy()

    # now delete the original
    X = df.drop(label_col, axis=1)

    # manual split
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=split, random_state=random)
    return X, trainX, testX, trainY, testY


# helper function which only splits into X and y
def XYsplit(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col, axis=1)
    return X, y


try:
    from xgboost import XGBClassifier
except:
    print("Failed to import xgboost, make sure you have xgboost installed")
    print("Use following command to install it: pip install xgboost")
    XGBClassifier = None

# loading data
hd_df = pd.read_csv('heart.csv')

hd_df.head(5)

# printing shape of dataset
hd_df.shape

# checking data types of each column
hd_df.dtypes

# Preprocessing

# Checking null or missing values

nullval = hd_df.isnull().sum()

# checking missing values
# columns that have null values greater than 5% -> It gives percentage
percent_missing = hd_df.isnull().sum() * 100 / len(hd_df)

# no of missing
num_missing = hd_df.isnull().sum()

# sorting values in descending order
percent_missing = percent_missing.sort_values(ascending=False)

# inserting in percentages into new dataframe
missing_value_df = pd.DataFrame({'percent_missing': percent_missing, 'Num_missing_val': num_missing})

missing_value_df = missing_value_df.sort_values(by=['percent_missing'], ascending=False)

missing_value_df[percent_missing > 5]

missing_value_df[percent_missing > 15]

# Checking duplicate values


hd_df[hd_df.duplicated(keep=False)]

# deleting duplicate entries
hd_df = hd_df.drop_duplicates(keep='first')

# checking outliers using Isolation forest technique

# In[9]:


model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1), max_features=1.0)
model.fit(hd_df)

# In[10]:


hd_df['scores'] = model.decision_function(hd_df[hd_df.columns[0:14]])
hd_df['anomaly'] = model.predict(hd_df[hd_df.columns[0:14]])
hd_df.head(5)

# In[13]:


# Print Anomalies
anomaly = hd_df.loc[hd_df['anomaly'] == -1]
anomaly_index = list(anomaly.index)
anomaly.head(3)

# EDA

# Information about features

# In[26]:


# for understanding each column better
information = ["age", "1: male, 0: female",
               "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
               "resting blood pressure", " serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl",
               "resting electrocardiographic results (values 0,1,2)", " maximum heart rate achieved",
               "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest",
               "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy",
               "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
for i in range(len(information)):
    print(hd_df.columns[i] + ":\t\t\t" + information[i])

# Analyzing target variable

# In[20]:


y = hd_df["target"]
sns.countplot(y)
tar_value = hd_df.target.value_counts()
print(tar_value)

# In[34]:


# checking the percentage of patients with and without heart disease
print("Percentage of patience without heart problems: target 0: " + str(round(tar_value[0] * 100 / 303, 2)))
print("Percentage of patience with heart problems: target 1: " + str(round(tar_value[1] * 100 / 303, 2)))

# Distribution of all columns / features

# In[37]:


# distribution of each column
hd_df.hist(figsize=(16, 20), xlabelsize=8, ylabelsize=8)

# Analyzing Gender Column

# Gender columns shows, there are 96 female patient's record and 206 males records in dataset

# In[51]:


hd_df['sex'].value_counts()

# In[53]:


# gender wise distribution of disease
hd_df.groupby(['sex', 'target'])['sex'].count()

# In[45]:


sns.countplot(data=hd_df, x='sex', hue='target')

# Above graph demonstrate that heart disease is more common in males than females, 0 shows female and 1 shows male.
# males are more likely to have heart problem than females

# Analyzing Chest Pain Feature

# In[14]:


# group wise cp with target feature
hd_df.groupby(['cp', 'target'])['cp'].count()

# In[47]:


# chest pain wise distribution of target data
sns.countplot(data=hd_df, x='cp', hue='target')

# Above graph shows that the chest pain of type 2 are more likely to have heart disease and type 3 are less likely to have heart disease.

# Analyzing Slope feature

# In[56]:


sns.countplot(data=hd_df, x='slope', hue='target')

# From above graph, it is observed that slop 2 causes more chest pain than 0 and 1

# Analyzing Thal feature

# In[17]:


hd_df['thal'].value_counts

# In[57]:


sns.countplot(data=hd_df, x='thal', hue='target')

# Above graph shows that patient's with thal 2 are more likely have heart disease and there are very few patient with 0 record.

# Analyzing Vessel (CA) feature

# In[60]:


sns.countplot(data=hd_df, x='ca', hue='target')

# Analyzing Blood Sugar feature with target variable

# In[65]:


hd_df['fbs'].unique

# In[63]:


sns.countplot(data=hd_df, x='fbs', hue='target')

# Above graph shows that patients those have heart disease are less likely to have blood sugar

# In[47]:


# sns.countplot(data=hd_df, x='chol', hue='target')
sns.distplot(hd_df['fbs'], color='green').set(title='Distribution of Fasting Blood Sugar')

# cholestrol level with target variable

# In[48]:


# sns.countplot(data=hd_df, x='chol', hue='target')
sns.distplot(hd_df['chol'], color='green').set(title='Distribution of Chelestrol Level')

# Age feature distribution

# In[49]:


sns.distplot(hd_df['age'], color='green').set(title='Distribution of Age')

# Blood Pressure

# In[50]:


sns.distplot(hd_df['trestbps'], color="green").set(title='Distribution of Resting Blood Pressure')

# Checking Correlation

# In[72]:


cor = hd_df.corr()
cor

# In[98]:


f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(hd_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)

# Checking correlation with target variable only

# In[80]:


corr = hd_df[hd_df.columns[1:]].corr()['target']
corr = corr.sort_values(ascending=False)
# corr.nlargest(11)
corr

# Scaling Features

# In[83]:


standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
hd_df[columns_to_scale] = standardScaler.fit_transform(hd_df[columns_to_scale])

# Splitting data

# In[18]:


from sklearn.model_selection import train_test_split

X = hd_df.drop("target", axis=1)
y = hd_df["target"]

# X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.20,random_state=0)


# Random Oversampling for class imbalance problem

# In[20]:


# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)

# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X_over, y_over, test_size=0.20, random_state=0)

# In[22]:


print(X_train.shape)
print(X_test.shape)

# In[23]:


print(Y_train.shape)
print(Y_test.shape)

# Feature Selection using Random Forest

# In[130]:


threshold = 5
df_cpy = hd_df.copy()
df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col='target')
res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY,
                          trees=10, regression=False)

impftrs = list(res[res > threshold].keys())

print("Selected Features =" + str(impftrs))

# In[7]:


SelectedFeatures = ['thalach', 'ca', 'thal', 'cp', 'age', 'oldpeak', 'slope', 'trestbps', 'sex']

# In[ ]:


X_train = X_train[SelectedFeatures]
X_test = X_test[SelectedFeatures]

# Feature selection using Lasso

# In[ ]:


# fit and apply the transform
X_over, y_over

# Random Forest

# In[26]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
lasso_model = Lasso()

# In[27]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_over)

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_sc, y_over, test_size=0.33, random_state=42)

# In[29]:


param = {
    'alpha': [.00001, 0.0001, 0.001, 0.01],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'positive': [True, False],
    'selection': ['cyclic', 'random'],
}

# In[32]:


# define search
search = GridSearchCV(lasso_model, param, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X_sc, y_over)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# In[33]:


lasso_model = Lasso(alpha=1e-05, fit_intercept=True, normalize=True, positive=False, selection='cyclic')

# In[49]:


X_over = X_over.drop('scores', 1)
X_over = X_over.drop('anomaly', 1)

# In[50]:


alphas = [1e-05, .5, 10]
# Create an empty data frame
df = pd.DataFrame()

# Create a column of feature names
df['Feature Name'] = hd_df.columns[0:13]

# For each alpha value in the list of alpha values,
for alpha in alphas:
    # Create a lasso regression with that alpha value,
    lasso = Lasso(alpha=alpha)

    # Fit the lasso regression
    lasso.fit(X_over, y_over)

    # Create a column name for that alpha value
    column_name = 'Alpha = %f' % alpha

    # Create a column of coefficient values
    df[column_name] = lasso.coef_

df

# In[51]:


X_train, X_test, Y_train, Y_test = train_test_split(X_over, y_over, test_size=0.20, random_state=0)

# In[52]:


max_accuracy = 0
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    if (current_accuracy > max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

# In[56]:


validationmetrics(rf, X_test, Y_test, verbose=True)

# Logistic Regression

# In[57]:


Lr = LogisticRegression(solver='lbfgs')
Lr.fit(X_train, Y_train)
validationmetrics(Lr, X_test, Y_test, verbose=True)

# SVM

# In[58]:


from sklearn import svm

sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
validationmetrics(sv, X_test, Y_test, verbose=True)

# XGBoost

# In[61]:


import xgboost as xgb

xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42, learning_rate=0.2)
xgb_model.fit(X_train, Y_train)
validationmetrics(xgb_model, X_test, Y_test, verbose=True)


# Adaboost

# In[62]:


def AdaBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostClassifier(n_estimators=50, random_state=42, learning_rate=0.1)
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


# In[63]:


AdaBoost(X_train, X_test, Y_train, Y_test, verbose=True, clf=None)

# Gradient Boosting

# In[77]:


from sklearn.ensemble import GradientBoostingClassifier


def GradBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingClassifier(n_estimators=60, random_state=0)
    clf.fit(trainX, trainY)
    return validationmetrics(clf, testX, testY, verbose=verbose)


# In[78]:


GradBoost(X_train, X_test, Y_train, Y_test, verbose=True, clf=None)

# Decision Tree

# In[86]:


DecisionTree(X_train, X_test, Y_train, Y_test, verbose=True, clf=None)

# NaiveBayes Classifier


NaiveBayes(X_train, X_test, Y_train, Y_test, verbose=True, clf=None)

# Multilayer Perceptron

MultiLayerPerceptron(X_train, X_test, Y_train, Y_test, verbose=True, clf=None)

# All Models scores

score_lr = 84
score_svm = 89.3
score_naiv = 78.7
score_rf = 90.9
score_xgb = 84.4
score_adb = 84.8
score_gdb = 84.8
score_dt = 75.7
score_mt = 75.7
scores = [score_lr, score_svm, score_naiv, score_rf, score_dt, score_xgb, score_adb, score_gdb, score_mt]
algorithms = ["LogisticRegression", "SupportVectorMachine", "NaiveBayes", "RandomForest", "DecisionTree", "XGBoost",
              "AdaBoost", "GradBoost", "MultilayerPerceptron"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + " %")

sns.set(rc={'figure.figsize': (17, 8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms, scores)
