#!/usr/bin/env python
# coding: utf-8

# # Aim: To perform and analysis of K-Nearest Neighbors (KNN) Algorithm

# In[2]:


# Name : Shruti Anil Dhote
# Roll no : 71
# Sec: C
# Subject : ET2
# Date : 18/01/2025


# In[4]:


import pandas as pd
import numpy as np


# In[5]:


import os


# In[6]:


os.getcwd()


# In[7]:


os.chdir('C:\\Users\\SURUTI DHOTE\\Desktop')


# In[8]:


data=pd.read_csv("heart.csv")


# In[9]:


data.head()


# In[10]:


data.tail()


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


data.shape


# In[14]:


data.size


# In[15]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment

# In[16]:


# check Missing Value by record
data.isna()


# In[17]:


data.isna().any()


# In[18]:


data.isna().sum()


# # Removing duplicates

# In[19]:


data_dup =data.duplicated().any()


# In[20]:


data_dup


# In[21]:


data=data.drop_duplicates()


# In[22]:


data_dup =data.duplicated().any()


# In[23]:


data_dup


# # Splitting of DataSet into train and Test
# 

# In[25]:


x=data.drop("target", axis=1)
y=data["target"]


# In[33]:


# Splitting the data into training and testing datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[34]:


x_train


# In[35]:


x_test


# In[36]:


y_train


# In[37]:


y_test


# # KNN Classifier

# In[39]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[40]:


knn=KNeighborsClassifier()


# In[41]:


knn.fit(x_train, y_train)


# In[42]:


y_pred=knn.predict(x_test)


# In[43]:


accuracy = accuracy_score(y_test, y_pred)


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

score = []
for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred3 = knn.predict(x_test)
    score.append(accuracy_score(y_test, y_pred3))


# In[46]:


score


# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred3)
labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

