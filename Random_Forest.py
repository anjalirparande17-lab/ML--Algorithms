#!/usr/bin/env python
# coding: utf-8

# # Random Forest

# In[1]:


import pandas as pd


# In[8]:


import warnings
warnings.filterwarnings("ignore")


# In[9]:


at=pd.read_csv(r"C:\Users\krish\Downloads\Attrition.csv")


# In[10]:


at


# In[11]:


at.isnull().sum()[at.isnull().sum()>0]


# In[12]:


at.info()


# In[13]:


at.Attrition.value_counts()


# In[14]:


at.Attrition.replace({'No':0,'Yes':1},inplace=True)


# In[15]:


at.select_dtypes(include='object').columns


# In[16]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[17]:


at[at.select_dtypes(include='object').columns]=at[at.select_dtypes(include='object').columns].apply(le.fit_transform)


# In[18]:


at.head()


# In[19]:


from sklearn.model_selection import train_test_split
at_train,at_test=train_test_split(at,test_size=.2)


# In[20]:


# over sampling
df1=at_train[at_train.Attrition==1]
at_train=pd.concat([at_train,df1])
at_train.shape


# In[21]:


at_train_x=at_train.iloc[::,at.columns!='Attrition']
at_train_y=at_train.Attrition


# In[22]:


at_test_x=at_test.iloc[::,at.columns!='Attrition']
at_test_y=at_test.Attrition


# In[23]:


at_test_y


# In[24]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier( max_depth=6,n_estimators=2,criterion="entropy",)


# In[25]:


rfc.fit(at_train_x,at_train_y)


# In[26]:


#pred_train=rfc.predict(at_train_x)
pred_test=rfc.predict(at_test_x)
pred_test


# In[27]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score


# In[28]:


mat=confusion_matrix(at_test_y,pred_test)


# In[29]:


mat


# In[30]:


accuracy_score(at_test_y,pred_test)


# In[31]:


recall_score(at_test_y,pred_test)


# In[32]:


precision_score(at_test_y,pred_test)


# In[33]:


f1_score(at_test_y,pred_test)


# In[34]:


pred_prob_test=rfc.predict_proba(at_test_x)


# In[35]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[36]:


roc_auc_score(at_test_y,pred_prob_test[:,1])


# In[37]:


fpr,tpr,thr=roc_curve(at_test_y,pred_prob_test[:,1])


# In[38]:


import matplotlib.pyplot as plt


# In[39]:


plt.plot(fpr,tpr,marker='*',color='green')
plt.xlabel(fpr)
plt.ylabel(tpr)
plt.grid()
plt.title('arouc curve')


# In[40]:


at.Attrition.value_counts()


# In[ ]:




