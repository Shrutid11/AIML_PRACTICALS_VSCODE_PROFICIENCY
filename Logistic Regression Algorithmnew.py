#!/usr/bin/env python
# coding: utf-8

# # To perform and analysis of Logistic Regression Algorithm

# # Importing the Libraries

# In[1]:


import pandas as pd 
import numpy as np


# # Data acquisitionuing Pandas 

# In[2]:


import os


# In[3]:


os.getcwd()


# In[5]:


os.chdir("C:\\Users\\SURUTI DHOTE\\DESKTOP")


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


# In[23]:


#splitting the data into training and testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)


# In[24]:


x_train


# In[25]:


x_test


# In[26]:


y_train


# In[27]:


y_test


# In[28]:


data.head()


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[46]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Ensure correct capitalization
scaler = StandardScaler()

# Fit and transform the training data
x_train_scaled = scaler.fit_transform(x_train)

# Transform the test data
x_test_scaled = scaler.transform(x_test)

# Train Logistic Regression
log = LogisticRegression(max_iter=1000)  # Increase iterations
log.fit(x_train_scaled, y_train)

# Predict using the correctly scaled data
y_pred1 = log.predict(x_test_scaled)  

print(y_pred1)  # Check predictions


# In[36]:


y_pred1=log.predict(x_test)


# In[33]:


from sklearn.metrics import accuracy_score 


# In[34]:


accuracy_score (y_test,y_pred1)


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[36]:


cm = confusion_matrix(y_test, y_pred1)


# In[37]:


labels = np.unique(y_test)  # Get unique class labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)


# In[36]:


# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

