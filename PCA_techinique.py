#!/usr/bin/env python
# coding: utf-8

# # PCA _Principal Component Analysis

# In[4]:


import pandas as pd


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[7]:


pp=pd.read_csv(r"C:\Users\krish\Downloads\Property_Price_Train.csv")


# In[8]:


pp


# In[9]:


pp.isnull().sum()


# In[11]:


pp=pp.drop(['Id','Fireplace_Quality','Pool_Quality','Fence_Quality','Miscellaneous_Feature','Lane_Type'],axis=1)


# In[12]:


pp.Lot_Extent.fillna(pp.Lot_Extent.mean(),inplace=True)
#pp.Lane_Type.fillna('Grvl',inplace=True)
pp.Brick_Veneer_Type.fillna('None',inplace=True)
pp.Brick_Veneer_Area.fillna(pp.Brick_Veneer_Area.mean(),inplace=True)
pp.Basement_Height.fillna('TA',inplace=True)
pp.Basement_Condition.fillna('TA',inplace=True)
pp.Exposure_Level.fillna('NO',inplace=True)
pp.BsmtFinType1.fillna('Unf',inplace=True)
pp.BsmtFinType2.fillna('Unf',inplace=True)
pp.Electrical_System.fillna('SBrkr',inplace=True)
#pp.Fireplace_Quality.fillna('Gd',inplace=True)
pp.Garage.fillna('Attchd',inplace=True)
pp.Garage_Built_Year.fillna(pp.Garage_Built_Year.mean(),inplace=True)
pp.Garage_Finish_Year.fillna('Unf',inplace=True)
pp.Garage_Quality.fillna('TA',inplace=True)
pp.Garage_Condition.fillna('TA',inplace=True)
#pp.Pool_Quality.fillna('Gd',inplace=True)
#pp.Fence_Quality.fillna('MnPrv',inplace=True)
#pp.Miscellaneous_Feature.fillna('Shed ',inplace=True)


# In[13]:


pp.isnull().sum()


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


le=LabelEncoder()
pp[pp.select_dtypes(include='object').columns]=pp[pp.select_dtypes(include='object').columns].apply(le.fit_transform)


# In[16]:


pp.head()


# In[17]:


pp.columns


# In[18]:


pp1=pp


# In[19]:


pp=pp.drop(['Sale_Price'],axis=1)


# In[20]:


pp.shape


# In[21]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[23]:


scaled_pp=scaler.fit_transform(pp)
scaled_pp


# In[26]:


from sklearn import decomposition
from sklearn.decomposition import PCA


# In[27]:


pca=PCA(n_components=55) # create an instace


# In[28]:


x_pca1=pca.fit_transform(scaled_pp)
# PCA has been performed


# In[30]:


pca.explained_variance_ratio_


# In[34]:


l1=list(pca.explained_variance_ratio_)


# In[35]:


len(l1)


# In[36]:


pca.explained_variance_ratio_.sum()


# In[37]:


# explained_variance_ratio_: how much variance each principal component explains.


# In[38]:


import numpy as np


# In[39]:


np.sum(l1[0:55])


# In[40]:


df=pd.DataFrame(x_pca1[:,0:55])


# In[41]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[42]:


linreg.fit(df,pp1.Sale_Price)


# In[43]:


linreg.score(df,pp1.Sale_Price)


# In[44]:


df1=pd.DataFrame(x_pca1)


# In[45]:


df1.corr().round(3) # since there is no co-relation between cols


# In[46]:


# Shows correlation between all PCA components. Since PCA creates orthogonal (uncorrelated)
# components, you should see near-zero correlations.


# In[47]:


# PCA cannot be used for feature selection bcause there is not one on one mapping your
# original column and the columns after transformation


# In[48]:


#PCA is not feature selection; it's dimensionality reduction.

#First few components explain the most variance.

#PCA reduces multicollinearity since components are uncorrelated.


# In[49]:


# So the result is a 74×74 correlation matrix.


# In[50]:


pp.shape


# In[51]:


pp1.shape


# In[ ]:




