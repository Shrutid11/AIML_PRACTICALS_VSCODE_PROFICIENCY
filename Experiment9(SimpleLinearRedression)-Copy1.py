
 # Aim : To perform and Analysis of Simple Linear Regression.

# Name : Shruti Anil Dhote
# Roll no : 71
# Sec: C
# Subject : ET2
# Date : 04/01/2025



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np





import os


# In[3]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


os.getcwd()


# In[4]:


os.chdir("C:\\Users\\SURUTI DHOTE\\Desktop\\")


# In[5]:


df=pd.read_csv("salary_data.csv")


# In[6]:


df.head()


# In[7]:


df.head(10)


# In[8]:


df.info()


# In[9]:


df.tail()


# In[10]:


df.describe()


# In[11]:


df.shape


# In[12]:


df.size


# In[13]:


df.ndim


# In[14]:


df.isnull()


# In[15]:


df.isnull()


# In[16]:


df.isnull().sum()


# In[17]:


df.head()


# In[18]:


df.columns


# In[19]:


df.loc[4,"Salary"]


# In[20]:


df.head(15)


# In[21]:


df.loc[2,"YearsExperience"]


# In[22]:


df.loc[12]


# In[23]:


df.loc[4]


# In[24]:


a=(1,2,3,4,5,6,7,8,9,10)


# In[25]:


a[1:4]


# In[26]:


df.loc[0:3,'YearsExperience':"Salary"]


# In[27]:


df.iloc[1,0]


# In[28]:


df.head()


# In[29]:


df.loc[1,"Salary"]


# In[30]:


#Assigning values in X & Y
x=df.iloc[:, :-1].values
y=df.iloc[:, :-1].values


# In[31]:


a[:2]


# In[32]:


a[2:]


# In[33]:


a[1:6:2]


# In[34]:


print(x)


# In[35]:


print(y)


# In[36]:


#splitting testdata into x_tarin,y_train'
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=34)


# In[37]:


print(x_train)


# In[38]:


print(x_test)


# In[39]:


print(y_train)


# In[40]:


print(y_test)


# In[41]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[43]:


#Assigning coefficient (slope) to m
m = lr.coef_


# In[44]:


print("Coefficient :",a)


# In[45]:


acc = lr.score(x_test, y_test) * 100
print(f"Accuracy: {acc:.2f}%")


# In[ ]:




