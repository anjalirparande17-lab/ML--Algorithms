#!/usr/bin/env python
# coding: utf-8

# # decision tree

# In[1]:


import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


cr=pd.read_csv(r"C:\Users\krish\Downloads\CreditRisk59.csv")


# In[6]:


cr


# In[7]:


cr.isnull().sum()


# In[8]:


cr.Gender.fillna('Male', inplace = True)
cr. Married. fillna('Yes', inplace = True)
cr. Dependents. fillna(0, inplace = True)
cr. LoanAmount.fillna(cr.LoanAmount.mean(), inplace = True)
cr. Loan_Amount_Term.fillna(cr.Loan_Amount_Term.mean(), inplace = True)
cr. Credit_History. fillna(0, inplace = True)
cr.Self_Employed.fillna('No', inplace= True)


# In[9]:


cr.select_dtypes(include='object').columns


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[11]:


cr[cr.select_dtypes(include='object').columns] = cr[cr.select_dtypes(include='object').columns].apply(le.fit_transform)


# In[12]:


cr1=cr # just for back up
cr=cr.drop(['Loan_ID'],axis=1)


# In[13]:


cr


# In[14]:


from sklearn.model_selection import train_test_split
cr_train,cr_test=train_test_split(cr,test_size=.2)


# In[16]:


cr.Loan_Status.value_counts()


# In[17]:


# over sampling
df1=cr_train[cr_train.Loan_Status==0]
cr_train=pd.concat([cr_train,df1,df1,df1,df1])
cr_train.shape


# In[18]:


cr_train_x=cr_train.iloc[:,0:-1]
cr_train_y=cr_train.iloc[:,-1]

cr_test_x=cr_test.iloc[:,0:-1]
cr_test_y=cr_test.iloc[:,-1]


# In[19]:


cr_test_y


# In[20]:


from sklearn.tree import  DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[21]:


dt.fit(cr_train_x,cr_train_y)


# In[22]:


pred_train=dt.predict(cr_train_x)
pred_test=dt.predict(cr_test_x)


# In[23]:


from sklearn.metrics import confusion_matrix
mat_test=confusion_matrix(cr_test_y,pred_test)
mat_test


# In[24]:


119/(119+33)*100


# In[25]:


from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score


# In[26]:


accuracy_score(cr_test_y,pred_test)


# In[27]:


recall_score(cr_test_y,pred_test)


# In[28]:


precision_score(cr_test_y,pred_test)


# In[29]:


f1_score(cr_test_y,pred_test)


# In[31]:


#fpr
21/(21+24)*100


# In[32]:


pred_prob_test=dt.predict_proba(cr_test_x)
len(pred_prob_test)


# In[33]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[34]:


roc_auc_score(cr_test_y,pred_prob_test[:,1])


# In[35]:


fpr,tpr,thre=roc_curve(cr_test_y,pred_prob_test[:,1])


# In[36]:


import matplotlib.pyplot as plt


# In[37]:


plt.plot(fpr,tpr,marker='*' , color='green')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc plot on fpr and tpr')
plt.grid()


# In[ ]:




