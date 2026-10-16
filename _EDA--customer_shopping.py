#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


sp=pd.read_csv(r"C:\Users\krish\Downloads\customer_shopping_data.csv")


# In[4]:


sp


# In[5]:


sp.columns


# In[6]:


sp.value_counts()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


sp.shape


# In[9]:


sp.head()


# In[10]:


sp.tail()


# In[11]:


sp.columns


# In[12]:


sp['age'].unique()


# In[13]:


sp['age'].nunique()


# In[14]:


sp.info()


# In[15]:


sp[['gender','age','category']]


# In[16]:


sp.describe()


# In[17]:


sp.describe(include="object")


# In[18]:


sp.columns


# In[19]:


sp.loc[(sp['gender']=='female') & (sp['category']=='Shoes')][['gender','category']]


# In[20]:


sp.iloc[np.where (sp["age"]>=40)]


# In[21]:


sp.isnull()


# In[22]:


sp.isnull().sum()


# In[23]:


sp.isnull().sum()[sp.isnull().sum()>0]


# In[24]:


sp


# In[25]:


sp


# # Data cleaning and data transfromation
# 

# In[27]:


sp.isnull().sum()[sp.isnull().sum()>0]


# In[28]:


sp.duplicated().sum()


# # Seperate categorical vs continuous data

# In[29]:


sp.columns


# In[30]:


sp.info()


# In[31]:


f1=sp.dtypes=='object'


# In[32]:


f1


# # Percentage

# In[33]:


(sp.isnull().sum()/len(sp))*100


# # check unique values

# In[34]:


sp.nunique()


# # for individual categorical columns

# In[35]:


sp["gender"].unique()


# In[36]:


sp["customer_id"].unique()


# In[37]:


sp["payment_method"].unique()


# In[38]:


sp["shopping_mall"].unique()


# # Create revenue column

# In[40]:


sp["revenue"]=sp["quantity"]* sp["price"]


# In[42]:


sp


# In[43]:


sp[["quantity","price","revenue"]]


# # univariate analysis

# In[ ]:


# gender


# In[45]:


sp["gender"].value_counts()


# In[46]:


sns.countplot(data=sp,x='gender')
plt.title('Customer Distribution')
plt.show()


# # category

# In[49]:


sp["category"].value_counts()


# In[50]:


plt.figure(figsize=(10,5))
sns.countplot(data=sp,x="category")
plt.xticks(rotation=45)
plt.title("Sales BY category")
plt.show()


#   # Payment method

# In[52]:


sns.countplot(data=sp,x="payment_method")
plt.title("payment method Distribution")
plt.show()


# # Shopping mall

# In[57]:


plt.figure(figsize=(10,5))
sns.countplot(data=sp,x="shopping_mall")
plt.xticks(rotation=45)
plt.title("Customers by shopping mall")
plt.show()


# # NUmerical analysis

# # Age Distribution

# In[58]:


sns.histplot(data=sp,x='age',bins=20,kde=True)
plt.title("Age Distribution")
plt.show()


# # Quantity

# In[59]:


sns.histplot(data=sp,x="quantity",bins=10,kde=True)
plt.title("Quantity Distribution")
plt.show()


# # price

# In[60]:


sns.histplot(data=sp,x="price",bins=30,kde=True)
plt.title("price Distribution")
plt.show()


# # revenue

# In[61]:


sns.histplot(data=sp,x="revenue",bins=30,kde=True)
plt.title("Revenue Distribution")
plt.show()


# # CENTRAL TENDENCY

# # Mean

# In[62]:


sp["age"].mean()


# In[66]:


sp['price'].mean()


# In[68]:


sp["quantity"].mean()


# In[69]:


sp["revenue"].mean()


# # Median

# In[64]:


sp['age'].median()


# In[70]:


sp['price'].median()


# In[71]:


sp['quantity'].median()


# In[72]:


sp['revenue'].median()


# # mode

# In[65]:


sp['age'].mode()


# In[73]:


sp['price'].median()


# In[74]:


sp['quantity'].median()


# In[75]:


sp['revenue'].median()


# # Category wise revenue

# In[82]:


category_revenue=(sp.groupby("category")["revenue"].sum().sort_values(ascending=False))
category_revenue


# In[83]:


# visualization


# In[85]:


category_revenue.plot(kind="bar",figsize=(10,5))
plt.title("revenue by category")
plt.xlabel("category")
plt.ylabel("revenue")
plt.xticks(rotation=45)
plt.show()


# # Gender VS revenue

# In[86]:


gender_revenue=sp.groupby("gender")["revenue"].sum()
gender_revenue


# In[87]:


gender_revenue.plot(kind="bar")
plt.title("Revenue by Gender")
plt.ylabel("Revenue")
plt.show()


# # Gender x category

# In[88]:


gender_category=pd.pivot_table(sp,values='revenue',index='category',columns='gender',aggfunc='sum')
gender_category


# In[89]:


# heatmap


# # Shopping mall analysis

# In[93]:


plt.figure(figsize=(10,6))
sns.heatmap(gender_category,annot=True,fmt=".0f")
plt.title("Revenue by category and gender")
plt.show()


# In[94]:


mall_revenue=sp.groupby("shopping_mall")["revenue"].sum().sort_values(ascending=False)
mall_revenue


# In[95]:


mall_revenue.plot(kind="bar",figsize=(12,5))
plt.title("revenue by shopping mall")
plt.ylabel("revenue")
plt.xticks(rotation=45)
plt.show()


# # Payment method vs revenue

# In[98]:


payment_revenue=sp.groupby("payment_method")["revenue"].sum()
payment_revenue


# In[101]:


payment_revenue.plot(kind="pie",autopct="%1.1f%%")
plt.title("revenue distribution by payment method")
plt.ylabel(" payment method")
plt.show()


# # Age group Analysis

# In[103]:


bins = [0, 18, 25, 35, 45, 55, 100]

labels = [
    "Below 18",
    "18-25",
    "26-35",
    "36-45",
    "46-55",
    "55+"
]

sp["age_group"] = pd.cut(
    sp["age"],
    bins=bins,
    labels=labels
)


# In[104]:


sp["age_group"].value_counts().sort_index()


# In[105]:


# revenue


# In[107]:


age_revenue = sp.groupby("age_group", observed=True)["revenue"].sum()

age_revenue


# # correation analysis

# In[108]:


numeric_sp = sp[["age", "quantity", "price", "revenue"]]

numeric_sp.corr()


# In[109]:


plt.figure(figsize=(8,6))

sns.heatmap(
    numeric_sp.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()


# # outlier analysis

# In[110]:


sns.boxplot(data=sp, x="price")

plt.title("Price Outliers")
plt.show()


# In[111]:


# for revenue
sns.boxplot(data=sp, x="revenue")

plt.title("Revenue Outliers")
plt.show()


# # top customers

# In[112]:


top_customers = (
    sp.groupby("customer_id")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_customers


# In[113]:


top_invoices = (
    sp.groupby("invoice_no")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_invoices


# In[ ]:




