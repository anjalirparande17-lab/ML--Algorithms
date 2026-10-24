#!/usr/bin/env python
# coding: utf-8

# # Knn -K nearest neighbors

# In[1]:


import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


ctg=pd.read_csv(r"C:\Users\krish\Downloads\CTG.csv")


# In[4]:


ctg.isnull().sum()[ctg.isnull().sum()>0]


# In[5]:


from sklearn.model_selection import train_test_split
ctg_train,ctg_test=train_test_split(ctg,test_size=.2)


# In[8]:


ctg.NSP.value_counts()


# In[9]:


#over sampling
df2=ctg_train[ctg_train.NSP==2]
df3=ctg_train[ctg_train.NSP==3]
ctg_train=pd.concat([ctg_train,df3,df3,df3,df3,df2])


# In[11]:


ctg_train_x=ctg_train.iloc[:,0:-1]
ctg_train_y=ctg_train.iloc[:,-1]

ctg_test_x=ctg_test.iloc[:,0:-1]
ctg_test_y=ctg_test.iloc[:,-1]


# In[12]:


ctg_test_x


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=37)


# In[14]:


knn.fit(ctg_train_x,ctg_train_y)


# In[15]:


pred_knn=knn.predict(ctg_test_x)


# In[16]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[17]:


tab_knn=confusion_matrix(ctg_test_y,pred_knn)


# In[18]:


tab_knn


# In[19]:


#tab_knn.diagonal().sum()/tab_knn.sum()
accuracy_score(ctg_test_y,pred_knn)


# In[20]:


print(classification_report(ctg_test_y,pred_knn))


# In[21]:


l1=[]
for k in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(ctg_train_x,ctg_train_y)
    pred_knn=knn.predict(ctg_test_x)
    tab_knn=confusion_matrix(ctg_test_y,pred_knn)
    acc=accuracy_score(ctg_test_y,pred_knn)
    l1.append(acc)


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


plt.figure(figsize=(15,6))
plt.plot(list(range(1, 51)), l1[:50], marker='*')
plt.grid()


# In[24]:


plt.figure(figsize=(15,6))
plt.plot(list(range(1,100)), l1, marker='*')
plt.grid()


# In[ ]:




