#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes

# In[1]:


import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


mp=pd.read_csv(r"C:\Users\krish\Downloads\trainRF.csv")


# In[4]:


mp


# In[5]:


mp.isnull().sum()


# In[7]:


from sklearn.model_selection import train_test_split
mp_train,mp_test=train_test_split(mp,test_size=.2)


# In[8]:


mp_train_x=mp_train.iloc[::,:-1]
mp_train_y=mp_train.iloc[::,-1]


# In[9]:


mp_test_x=mp_test.iloc[::,:-1]
mp_test_y=mp_test.iloc[::,-1]


# In[11]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()


# In[14]:


nb.fit(mp_train_x,mp_train_y)


# In[15]:


pred=nb.predict(mp_train_x)


# In[16]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


# In[17]:


tab=confusion_matrix(mp_train_y,pred)


# In[18]:


tab


# In[19]:


accuracy_score(mp_train_y,pred)


# In[20]:


precision_score(mp_train_y,pred,average='macro')


# In[23]:


recall_score(mp_train_y,pred,average='macro')


# In[24]:


f1_score(mp_train_y,pred,average='macro')


# In[ ]:




