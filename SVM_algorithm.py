#!/usr/bin/env python
# coding: utf-8

# Aim: To perform and analysis of SVM Algorithm

# Name : Shruti Anil Dhote  
# Roll no : 71  
# Sec: C  
# Subject : ET2  
# Date : 18/01/2025  

import pandas as pd
import numpy as np
import os

# Set working directory
os.chdir('C:\\Users\\SURUTI DHOTE\\Desktop')

# Load dataset
data = pd.read_csv("heart.csv")

# Data preview
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())
print("Shape:", data.shape)
print("Size:", data.size)
print("Dimensions:", data.ndim)

# Data Cleaning: Missing Values
print("Missing values in each column:\n", data.isna().sum())

# Removing Duplicates
if data.duplicated().any():
    print("Duplicates found. Removing...")
    data = data.drop_duplicates()
else:
    print("No duplicates found.")

# Splitting features and target
x = data.drop("target", axis=1)
y = data["target"]

# Splitting into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# SVM Classifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

svm_model = svm.SVC()
svm_model.fit(x_train, y_train)

# Prediction
y_pred2 = svm_model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred2))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred2)
labels = np.unique(y_test)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(4, 3))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens', linewidths=1, linecolor='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
