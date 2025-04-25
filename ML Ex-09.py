#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("Salary.csv")


# In[2]:


data.head()


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()


# In[6]:


x=data[["Position","Level"]]
x.head()


# In[7]:


y=data[["Salary"]]


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[14]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print('NAME: A.LAHARI')
print('REG.No : 212223230111')


# In[10]:


from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse


# In[11]:


r2=metrics.r2_score(y_test,y_pred)
r2


# In[12]:


dt.predict([[5,6]])

