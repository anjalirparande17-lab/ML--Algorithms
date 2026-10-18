#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np


# In[36]:


import warnings 
warnings.filterwarnings("ignore")


# In[37]:


dl=pd.read_csv(r"C:\Users\krish\Downloads\diabetesLogistic.csv")


# In[38]:


dl


# In[39]:


dl.isnull().sum()


# In[40]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[41]:


dl[dl.select_dtypes(include='object').columns]=dl[dl.select_dtypes(include='object').columns].apply(le.fit_transform)


# In[42]:


dl


# In[43]:


from sklearn.model_selection import train_test_split
dl_train,dl_test=train_test_split(dl,test_size=.2)


# In[44]:


dl_train.Outcome.value_counts()


# In[45]:


dl_train_x = dl_train.drop('Outcome', axis=1)
dl_train_y = dl_train['Outcome']

dl_test_x = dl_test.drop('Outcome', axis=1)
dl_test_y = dl_test['Outcome']


# In[46]:


dl_test_x


# In[47]:


from sklearn.linear_model import LogisticRegression 


# In[48]:


logreg=LogisticRegression()
logreg.fit(dl_train_x,dl_train_y)


# In[49]:


pred_train=logreg.predict(dl_train_x)
pred_test=logreg.predict(dl_test_x)


# In[50]:


from sklearn.metrics import confusion_matrix


# In[51]:


mat_test=confusion_matrix(dl_test_y,pred_test)
mat_test


# In[52]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[53]:


21/(21+30)*100


# In[54]:


accuracy_score(dl_test_y,pred_test)


# In[55]:


recall_score(dl_test_y,pred_test)


# In[56]:


precision_score(dl_test_y,pred_test)


# In[57]:


f1_score(dl_test_y,pred_test)


# In[58]:


#recall /TPR
mat_test.diagonal().sum()/ mat_test.sum()


# In[59]:


mat_test1=pd.DataFrame(mat_test)
mat_test1.columns=['Predicted 0','predicted 1 ']


# In[60]:


mat_test1


# In[61]:


pred_prob_test=logreg.predict_proba(dl_test_x)
len(pred_prob_test)


# In[62]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[63]:


roc_auc_score(dl_test_y,pred_prob_test[:,1])


# In[64]:


fpr,tpr,thre  =roc_curve(dl_test_y,pred_prob_test[:,1])


# In[65]:


import matplotlib.pyplot as plt


# In[66]:


plt.plot(fpr,tpr,marker='*' , color='green')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc plot on fpr and tpr')
plt.grid()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




