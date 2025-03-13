import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#region Mission One
data = {
    "Real Value": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "Model Probability Predict": [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]
}
df = pd.DataFrame(data)

y = df["Real Value"]
y_prob = df["Model Probability Predict"]


def plot_confusion_matrix(y, y_prob, threshold=0.5):

    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", linewidths=1, linecolor='black')
    plt.xlabel("Tahmin Edilen Değer (y_pred)")
    plt.ylabel("Gerçek Değer (y_true)")
    plt.title(f"Confusion Matrix\nAccuracy Score: {acc} (Threshold: {threshold})", size=12)
    plt.show()

plot_confusion_matrix(y, y_prob, threshold=0.5)

y_pred = [1 if prob>= 0.5 else 0 for prob in y_prob]
print(classification_report(y, y_pred))
#endregion

#region Mission Two

conf_matrix = np.array([[5, 5], [90, 900]])

y_true = [1] * 5 + [1] * 5 + [0] * 90 + [0] * 900  #
y_pred = [1] * 5 + [0] * 5 + [1] * 90 + [0] * 900

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor='black')
    plt.xlabel("Tahmin Edilen Değer (y_pred)")
    plt.ylabel("Gerçek Değer (y_true)")
    plt.title(f"Confusion Matrix\nAccuracy Score: {round(accuracy, 2)}", size=12)
    plt.show()

plot_confusion_matrix(conf_matrix)
classification_report(y_true, y_pred)

"""
📌 What’s Wrong with the Model?  
It correctly identifies most non-fraud transactions.But it fails to catch fraud cases properly!  
Only 33% of real fraud cases are detected.
It falsely flags too many transactions as fraud (Precision = 0.05). 
"""

"""
🛠️ How Can We Fix It?  
Gather more fraud datato make the dataset fairer.  
Adjust the decision threshold so it catches more fraud cases.  
Use better features (for example: transaction amount, location, time patterns).  
"""

