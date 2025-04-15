#!/usr/bin/env python
# coding: utf-8

# # To perform and Data analysis with Co-relation Matrix

# In[1]:


# Name : Shruti Anil Dhote
# Roll no : 71
# Sec: C
# Subject : ET2
# Date : 15/03/2025


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import os


# In[4]:


os.getcwd()


# In[32]:


os.chdir(r'C:\Users\SURUTI DHOTE\Downloads\archive')


# In[33]:


data=pd.read_csv("student_scores.csv")


# In[34]:


print("First 5 rows of the dataset:")
print(data.head())


# In[35]:


correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns

# If not already computed
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap - Student Scores')
plt.show()

