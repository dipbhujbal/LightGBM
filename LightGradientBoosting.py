#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[4]:


main_data=pd.read_csv(r'C:\Users\lenovo\Desktop\creditcard.csv')#loading credit card dataset


# In[5]:


main_data.describe()


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X=main_data.iloc[:,:-1] #selection feature set


# In[8]:


Y=main_data.iloc[:,-1:] #outcome variable set (class lable)


# In[9]:


X.head(3)


# In[10]:


Y.head(3)


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0) #splitting into train test dataset


# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)


# In[13]:


x_test=sc.transform(x_test)


# In[40]:


import lightgbm as lgb
#convert dataset into lgb format
data_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, data_train, 100)


# In[41]:


y_pred=clf.predict(x_test)


# In[42]:


print(y_pred)


# In[43]:


#convert into binary values
for i in range(len(y_pred)):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0


# In[44]:


print(y_pred)


# In[45]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mat= confusion_matrix(y_test, y_pred)#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)


# In[46]:


print("accuracy : %d%%"%(accuracy*100))


# In[ ]:





# In[ ]:




