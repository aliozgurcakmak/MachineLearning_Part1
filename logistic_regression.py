######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# Business Problem:

# Can you develop a machine learning model that predicts whether
# individuals are diabetic based on given features?

# The dataset is part of a large database held by the National Institute of Diabetes and Digestive and Kidney Diseases in the USA.
# It was used for a diabetes study conducted on Pima Indian women aged 21 and above, living in Phoenix, the 5th largest city
# in the state of Arizona, USA. The dataset contains 768 observations and 8 numerical independent variables.
# The target variable is labeled as "outcome," where 1 indicates a positive diabetes test result,
# and 0 indicates a negative result.

# Variables
# Pregnancies: Number of pregnancies
# Glucose: Glucose level.
# BloodPressure: Blood pressure.
# SkinThickness: Skin thickness.
# Insulin: Insulin level.
# BMI: Body Mass Index.
# DiabetesPedigreeFunction: A function that calculates the likelihood of diabetes based on family history.
# Age: Age (years)
# Outcome: Whether the individual has diabetes. Positive (1) or Negative (0)



# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#region Explorotary Data Analysis

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape

# Target Analysis
df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

# Feature Analysis
df.describe().T

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_cols(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

for col in df.columns:
    plot_numerical_cols(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_cols(df, col)

# Target vs. Features

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)

#endregion

#region Data Pre-Processing

df.shape
df.isnull().sum()
df.describe().T

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

#endregion

#region Model&Prediction

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)
y_pred[:10]
y[:10]

#endregion

#region Model Evaluation

def plot_confusion_matrix(y,y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()

plot_confusion_matrix(y,y_pred)

print(classification_report(y, y_pred))

"""
Accuracy: 0.78
Precision: 0.74
Recall: 0.58
F1-score: 0.65
"""

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

#0.8394104477611941
#endregion

#region Model Validation: Holdout

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))


"""
Accuracy: 0.78
Precision: 0.74
Recall: 0.58
F1-score: 0.65
"""

"""
Accuracy: 0.77
Precision: 0.79
Recall: 0.53
F1-Score: 0.63
"""
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(
    estimator=log_model,
    X=X_test,
    y=y_test
)
plt.show()




#endregion

#region Model Validation: 10-Fold Cross Validation

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results["test_accuracy"]
cv_results["test_accuracy"].mean()

cv_results["test_precision"]
cv_results["test_precision"].mean()

cv_results["test_recall"]
cv_results["test_recall"].mean()

cv_results["test_f1"]
cv_results["test_f1"].mean()
#endregion

#region Prediction for A New Obversation

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)
#endregion