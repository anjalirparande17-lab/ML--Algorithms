#!/usr/bin/env python
# coding: utf-8

# # Hierarchical clustering

# In[8]:


import warnings
warnings.filterwarnings("ignore")


# In[9]:


import pandas as pd


# In[10]:


mall = pd.read_csv(r'C:\Users\krish\Downloads\mall_kmeans.csv')


# In[11]:


mall


# In[12]:


mall.head()


# In[13]:


mall.drop('CustomerID', axis=1, inplace=True)


# In[14]:


mall.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)


# In[15]:


mall['Genre'].replace({'Female': 0, 'Male': 1}, inplace=True)


# In[17]:


mall.head()


# In[18]:


from sklearn.cluster import AgglomerativeClustering


# In[19]:


agg_mall = AgglomerativeClustering(n_clusters=5, linkage='ward', metric='euclidean')
#mall['Cluster'] = agg_mall.fit_predict(mall)


# In[20]:


agg_mall.fit(mall)


# In[21]:


agg_mall.labels_


# In[22]:


len(agg_mall.labels_)


# In[23]:


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# In[24]:


mall['Cluster'] = agg_mall.fit_predict(mall)


# In[25]:


mall['Cluster'].value_counts()


# In[26]:


# Features only (excluding cluster labels)
X = mall.drop('Cluster', axis=1)
labels = mall['Cluster']


# In[27]:


# Evaluation metrics
sil_score = silhouette_score(X, labels)
sil_score


# In[28]:


mall.head()


# In[29]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


# In[30]:


# single, complete,ward


# In[31]:


# Dendrogram to help choose number of clusters (optional but useful)
plt.figure(figsize=(12, 6))
link_result = linkage(mall.drop('Cluster', axis=1), method='single')
dendrogram(link_result)
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.grid()
plt.show()


# In[32]:


# Scatter plot: Age vs Spending Score colored by cluster
colors = np.array(['red', 'green', 'blue', 'black', 'purple'])

plt.figure(figsize=(8, 8))
plt.scatter(mall['Age'], mall['Score'], c=colors[mall['Cluster']])
plt.title("Clusters (Age vs Spending Score)")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.grid()
plt.show()


# In[ ]:




