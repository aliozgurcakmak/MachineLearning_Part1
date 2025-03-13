"""
Develop a machine learning model to predict customer churn.

Data Fields:
- CustomerId: Customer ID
- Gender: Gender
- SeniorCitizen: Whether the customer is a senior (1, 0)
- Partner: Whether the customer has a partner (Yes, No)
- Dependents: Whether the customer has dependents (Yes, No)
- tenure: Number of months the customer has stayed with the company
- PhoneService: Whether the customer has phone service (Yes, No)
- MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
- InternetService: Customer's internet service provider (DSL, Fiber optic, No)
- OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
- OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
- DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
- TechSupport: Whether the customer has tech support (Yes, No, No internet service)
- StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
- StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
- Contract: Contract term (Month-to-month, One year, Two year)
- PaperlessBilling: Whether the customer has paperless billing (Yes, No)
- PaymentMethod: Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- MonthlyCharges: Monthly charges
- TotalCharges: Total charges
- Churn: Whether the customer churned (Yes or No)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", None)

# Load data
df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()

print(df.head())
print(df.shape)
df.info()
print(df.isnull().sum())

# Identify categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtypes == "object"]
print(len(cat_cols))
num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]
print(len(num_cols))

# Data type correction for TotalCharges
df.info()
print(df[df["TotalCharges"] == " "])
print(df["TotalCharges"].isnull().sum())
for col in df.columns:
    if df[df[col] == " "][col].any():
        print(f"{col} column has blank values")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]
cat_cols = [col for col in df.columns if df[col].dtypes == "object"]
df.info()

# Numerical summary and histograms
def num_summary(df, num_cols, plot=False):
    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(df[num_cols].describe(percentiles=q).T)
    if plot:
        for col in num_cols:
            df[col].hist()
            plt.title(col)
            plt.show(block=True)

num_summary(df, num_cols, plot=True)

# Categorical summary and bar plots
def cat_summary(df, col, plot=False):
    print(pd.DataFrame({col: df[col].value_counts(),
                        "ratio": (df[col].value_counts() / len(df)) * 100}))
    if plot:
        df[col].value_counts().plot(kind="bar")
        plt.title(col)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

cat_cols = [col for col in cat_cols if col != "customerID"]
print(df.head())

df.info()

# Create a binary version of Churn for group analysis
binary_cols = [col for col in df.columns if df[col].nunique() == 2]
not_binary_num = [col for col in num_cols if col not in binary_cols]
not_binary_cat = [col for col in cat_cols if col not in binary_cols]

df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})
print(df.head())

for col in not_binary_cat:
    print(df.groupby(col)["Churn_Binary"].mean())
for col in binary_cols:
    print(df.groupby(col)["Churn_Binary"].mean())
df = df.drop("Churn_Binary", axis=1)

for col in not_binary_num:
    print(df.groupby("Churn").agg({col: "mean"}))

# Outlier visualization for 'tenure'
for col in num_cols:
    plt.boxplot(df["tenure"])
    plt.title(col)
    plt.show(block=True)

print(df.isnull().sum())

# Fill missing TotalCharges with median
num_summary(df, num_cols)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print(df.isnull().sum())

# Local Outlier Factor analysis
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[["tenure", "MonthlyCharges", "TotalCharges"]])
df_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()
th = np.sort(df_scores)[14]
print(th)
print(df[["tenure", "MonthlyCharges", "TotalCharges"]][df_scores < th])
num_summary(df, ["tenure", "MonthlyCharges", "TotalCharges"])
print(df[df["TotalCharges"] < df["MonthlyCharges"]])

# Feature Engineering
df.loc[(df["PaymentMethod"] == "Electronic check") & (df["Contract"] == "Month-to-month") &
       (df["StreamingMovies"] != "No internet service") & (df["StreamingTV"] != "No internet service") &
       (df["TechSupport"] == "No") & (df["DeviceProtection"] != "No internet service") &
       (df["OnlineBackup"] != "No internet service") & (df["OnlineSecurity"] == "No") &
       (df["InternetService"] == "Fiber optic"), "New_High_Churn"] = 1
df["New_High_Churn"] = df["New_High_Churn"].fillna(0)

df.loc[(df["SeniorCitizen"] == 1) & (df["Partner"] == "No") & (df["PaperlessBilling"] == "Yes"), "New_High_Churn_2"] = 1
df["New_High_Churn_2"] = df["New_High_Churn_2"].fillna(0)
print(df["New_High_Churn_2"].value_counts())

threshold = 11
print(df.loc[df["tenure"].between(17 - threshold, 17 + threshold)]["Churn"].value_counts())
print(df.loc[df["tenure"].between(17 - threshold, 17 + threshold)]["Churn"].value_counts() / len(df) * 100)
df.loc[df["tenure"].between(17 - threshold, 17 + threshold), "NEW_Not_Churn"] = 1
df["NEW_Not_Churn"] = df["NEW_Not_Churn"].fillna(0)
print(df.head())

# Encoding
def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

dff = pd.get_dummies(df, columns=not_binary_cat, drop_first=True)
print(dff.head())

# Scaling numerical features
mms = MinMaxScaler()
dff[not_binary_num] = mms.fit_transform(dff[not_binary_num])

# Prepare data for modeling
dff = dff.drop("customerID", axis=1)
X = dff.drop(["Churn"], axis=1)
y = dff["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
print(X_train.shape)
print(X_test.shape)

# CatBoost model
catboost_model = CatBoostClassifier(verbose=False, random_state=17).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
print(classification_report(y_pred, y_test))
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")

# Random Forest model
rf_model = RandomForestClassifier(random_state=17).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(classification_report(y_pred, y_test))

# LGBM model
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)
y_lgbm_pred = lgbm_model.predict(X_train)
print(classification_report(y_pred, y_test))

# Hyperparameter Optimization for Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [3, 6, 9, 12, 15],
    'min_samples_split': [2, 4, 8, 16, 32],
    'min_samples_leaf': [1, 3, 5, 7, 10],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(dt_model, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 10-fold Cross Validation with Decision Tree
X = dff.drop(["Churn"], axis=1)
y = dff["Churn"]
clf = DecisionTreeClassifier(random_state=42)
cv_results = cross_validate(clf, X, y, cv=10, scoring=['accuracy', 'f1_weighted', 'recall', 'precision'])
print("Average Accuracy:", cv_results['test_accuracy'].mean())
print("Average F1 Score:", cv_results['test_f1_weighted'].mean())
print("Average Recall:", cv_results['test_recall'].mean())
print("Average Precision:", cv_results['test_precision'].mean())

print(classification_report(y_pred, y_test))
