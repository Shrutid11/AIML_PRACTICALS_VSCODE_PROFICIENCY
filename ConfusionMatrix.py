#!/usr/bin/env python
# coding: utf-8

# # To Perform and Data Analysis With Confusion Matrix

# In[1]:


# Name : Shruti Anil Dhote
# Roll no : 71
# Sec: C
# Subject : ET2
# Date : 22/03/2025


# In[3]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


iris = load_iris()
X = iris.data
y = iris.target


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42)


# In[6]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[7]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# In[8]:


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assuming `cm` is your confusion matrix and `iris.target_names` contains the labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Iris Classification")
plt.show()


# In[9]:


print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

