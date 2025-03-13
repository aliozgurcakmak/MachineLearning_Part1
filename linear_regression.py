##########################################
# Sales Prediction with Linear Regression
##########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# region Simple Linear Regression with OLS Using Scikit Learn

df = pd.read_csv("datasets/advertising.csv")

X = df[['TV']]
y = df['sales']

######################################################
# Model
######################################################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV

# Intercept = (b - bias)
reg_intercept = reg_model.intercept_
print("Intercept :", reg_intercept)

# Coefficient = (W1)
reg_coef = reg_model.coef_[0]
print("Coefficient :", reg_coef)


########################
# Predict
########################


reg_intercept + (reg_coef * 150)
reg_intercept + (reg_coef * 500)

df.describe().T

# Visualization of Model
g = sns.regplot(x=X, y=y, scatter_kws={"color": "b", "s": 9},
                ci=False, color="r")
g.set_xlabel("TV Affords")
g.set_ylabel("Sales Number")
g.set_title(f"Model Equalization: Sales {round(reg_intercept, 2)} + TV*{round(reg_coef, 2)}")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()


# Prediction Success

## Mean Squared Error
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
#10.512652915656757
y.mean()
#14.0225
y.std()
#5.217456565710477

## Root Mean Squared Error
np.sqrt(mean_squared_error(y, y_pred))
#3.2423221486546887

## Mean Absoloute Error
mean_absolute_error(y, y_pred)
#2.549806038927486

## R-SQUARE
reg_model.score(X, y)
#endregion

#region Multiple Linear Regression

df = pd.read_csv("datasets/advertising.csv")
X = df.drop(['sales'], axis=1)
y = df['sales']

#####################
# Model
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.shape
y_train.shape

X_test.shape
y_test.shape

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

#reg_model = LinearRegression().fit(X_train, y_train)

# Intercept ( b - bias )
reg_intercept = reg_model.intercept_
print("Intercept :", reg_intercept)

# Coefficients ( w - weights )
reg_coef = reg_model.coef_
print("Coefficients :", reg_coef)

# Prediction

# What is the predicted value of sales according to followings observation values?

# TV: 30
# Radio : 10
# Newspaper : 40

reg_intercept + (reg_coef[0]*30 + reg_coef[1]*10 + reg_coef[2]*40)
new_data = [[30], [10], [40]]
new_data = pd.DataFrame(new_data).T
reg_model.predict(new_data)

# Prediction Success

## TRAIN RMSE
y_pred = reg_model.predict(X_train)
mean_squared_error(y_train, y_pred)
np.sqrt(mean_squared_error(y_train, y_pred))
#1.73

## TRAIN RSQUARE
reg_model.score(X_train, y_train)
#0.8959372632325174

## TEST RMSE
y_pred = reg_model.predict(X_test)
mean_squared_error(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))
#1.41

## TEST RSQUARE
reg_model.score(X_test, y_test)
#0.8927605914615384

## 10 storeys CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X, y,
                                 cv=10,
                                 scoring='neg_mean_squared_error')))
np.median(np.sqrt(-cross_val_score(reg_model,
                                   X, y,
                                   cv=10,
                                   scoring='neg_mean_squared_error')))

#endregion

#region Simple Linear Regression with Gradient Descent from Scratch

# Cost Function
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)