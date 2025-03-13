# Machine Learning Algorithms & Case Studies 🚀

Welcome to the repository for the **6th Module - Part 1, Week 8 of the Miuul Data Scientist Bootcamp**. This repository includes three machine learning projects focusing on predicting customer churn, logistic regression, linear regression, and k-nearest neighbors (KNN) implementations. These case studies cover comprehensive exploratory data analysis, feature engineering, preprocessing steps, model building, validation techniques, and hyperparameter tuning.

---

## 📂 Contents

### 1️⃣ Telco Customer Churn Prediction

**Dataset:** Telco Customer Churn

### Project Overview 🎯
This case study aims to predict customer churn (whether a customer leaves the company) using various machine learning techniques.

### Project Steps 🛠️
- Data loading and exploration
- Handling missing and outlier values
- Feature engineering (creation of new features)
- Encoding categorical variables
- Scaling numerical variables (MinMaxScaler)
- Model building and validation

### Models Applied ⚙️
- **CatBoost Classifier**
- **Random Forest Classifier**
- **LightGBM Classifier**
- **Decision Tree Classifier** (with hyperparameter tuning)

### Performance Metrics 📈
- Accuracy
- Precision
- Recall
- F1-score

### Results 🎖️
- **CatBoost** achieved an accuracy of ~0.79 and F1-score around 0.56.
- **Decision Tree** hyperparameter optimization significantly improved performance.

---

## Logistic Regression 📊

### Project Overview 🎯
This script demonstrates logistic regression modeling with detailed data preprocessing, model building, validation, and hyperparameter optimization.

### Techniques Covered 🧩
- Data loading and cleaning
- Exploratory data analysis (EDA)
- Feature engineering
- Label encoding and One-hot encoding
- Standardization using RobustScaler

### Model Validation Strategies ✅
- Holdout (train-test split)
- 10-fold Cross-validation

### Key Metrics Evaluated 📐
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score

### Best Results 🥇
- Logistic Regression achieved a consistent accuracy of around 0.80 with an ROC-AUC of 0.85 using cross-validation.

---

## Linear Regression 📈

### Project Overview 🎯
This project demonstrates the implementation of linear regression, including data preprocessing, feature engineering, handling assumptions, and evaluation metrics.

### Steps Included 🔧
- Data preparation and cleaning
- Detecting and handling outliers
- Feature transformation and engineering
- Model training and validation
- Hyperparameter tuning (if applicable)

### Metrics Evaluated 📏
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (coefficient of determination)

### Highlights ⭐
- Model assumptions clearly checked (linearity, normality, multicollinearity)
- Effective feature engineering and selection techniques demonstrated

---

## K-Nearest Neighbors (KNN) 🌐

### Project Overview 🎯
This script explores the KNN algorithm for classification problems, including preprocessing, feature scaling, model validation, and hyperparameter optimization.

### Techniques Covered 🛠️
- Data standardization using MinMaxScaler
- Handling categorical features
- Distance calculation
- Hyperparameter optimization using GridSearchCV

### Metrics Evaluated 📏
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Best Parameters ⚡
- Optimal number of neighbors identified: 34
- Achieved accuracy around 0.80 and ROC-AUC of approximately 0.83.

---

## Libraries Used 📚
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn` (classification, regression, preprocessing)
- `catboost`, `lightgbm` (advanced modeling)
- `missingno` (missing data analysis)

---

## Final Thoughts 💡
These projects greatly enhanced my understanding of various machine learning algorithms and their practical applications. Real-world data sets and rigorous model evaluation methods provided insights into creating robust, reliable, and accurate predictive models.

Feel free to explore, modify, and use this repository as a reference for your own projects!

Happy Coding! 🚀😊

