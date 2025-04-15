#!/usr/bin/env python
# coding: utf-8

# # Aim: To perform and analysis of Decision Trees Algorithm
# 

# In[1]:


# Name : Shruti Anil Dhote
# Roll no : 71
# Sec: C
# Subject : ET2
# Date : 25/01/2025


# # Importing the Libraries

# In[2]:


import pandas as pd
import numpy as np


# # Data acquisitionuing Pandas

# In[3]:


import os


# In[4]:


os.getcwd()


# In[5]:


os.chdir('C:\\Users\\SURUTI DHOTE\\Desktop')


# In[6]:


data=pd.read_csv("heart.csv")


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.info()


# In[10]:


data.describe()


# In[11]:


data.shape


# In[12]:


data.size


# In[13]:


data.ndim


# # Data preprocessing _ data cleaning _ missing value treatment

# In[14]:


# check Missing Value by record
data.isna()


# In[15]:


data.isna().any()


# In[16]:


data.isna().sum()


# # Removing duplicates
# 

# In[17]:


data_dup =data.duplicated().any()


# In[18]:


data_dup


# In[19]:


data=data.drop_duplicates()


# In[20]:


data_dup =data.duplicated().any()


# In[21]:


data_dup


# # Splitting of DataSet into train and Test

# In[22]:


x=data.drop("target", axis=1)
y=data["target"]


# In[25]:


# Splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[26]:


x_train


# In[27]:


x_test


# In[28]:


y_train


# In[29]:


y_test


# # Decision Trees Algorithm

# In[30]:


from sklearn.tree import DecisionTreeClassifier


# In[31]:


dt=DecisionTreeClassifier()


# In[32]:


dt.fit(x_train, y_train)


# In[33]:


y_pred5=dt.predict(x_test)


# In[36]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred5)


# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred5)
labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




