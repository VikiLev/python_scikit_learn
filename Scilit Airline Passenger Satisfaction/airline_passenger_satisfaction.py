#!/usr/bin/env python
# coding: utf-8

# In[665]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[666]:


## Loading data


# In[667]:


df = pd.read_csv('airline_passenger_satisfaction.csv')
pd.set_option('display.max_columns', None)
df.head()


# In[668]:


df.describe().round(2)


# In[669]:


## Data cleaning


# In[670]:


df.info()


# In[671]:


df.isnull().sum()


# In[672]:


df['Arrival Delay in Minutes']


# In[673]:


df['Arrival Delay in Minutes'].mean()


# In[674]:


df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)


# In[675]:


df.isnull().sum()


# In[676]:


## Charts


# In[677]:


plt.pie(df['satisfaction'].value_counts(), labels=['neutral or dissatisfied', 'satisfied'], autopct='%1.1f%%')


# In[678]:


## Encoding data



# In[679]:


df.dtypes


# In[680]:


df.select_dtypes(include='object')


# In[681]:


df['Gender'].unique()


# In[682]:


df['Customer Type'].unique()


# In[683]:


df['Type of Travel'].unique()


# In[684]:


df['Class'].unique()


# In[685]:


# option 1


#df.replace({
#     'Gender':{
#         'Male': 1,
#         'Female': 2
#     },
#     'Customer Type':{
#         'Loyal Customer': 1,
#         'disloyal Customer': 2
#     },
#     'Type of Travel':{
#         'Business travel': 1,
#         'Personal Travel': 2
#     },
#     'Class':{
#         'Eco': 1,
#         'Business': 2,
#         'Eco Plus': 3
#     },
# }, inplace=True)


# In[686]:


# option 2
columns = df.select_dtypes(include='object').drop(columns='satisfaction').columns

label_encoder = LabelEncoder()

for column in columns:
    df[column] = label_encoder.fit_transform(df[column])


df.head()


# In[687]:


df.dtypes


# In[688]:


## Additional Charts


# In[689]:


plt.figure(figsize=(16, 8))
sns.heatmap(df.drop(columns='satisfaction').corr(), annot=True, fmt='.2f',)
plt.show()


# In[690]:


sns.catplot(data=df, x='Age', height=4, aspect=4, kind='count', hue='satisfaction')


# In[691]:


## models


# In[692]:


X = df.drop(columns='satisfaction')


# In[693]:


X.head()


# In[694]:


y = df['satisfaction']
y.head()


# In[695]:


### Decision tree


# In[696]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier()


# In[697]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[698]:


model.fit(X_train, y_train)


# In[699]:


predictions = model.predict(X_test)
predictions


# In[700]:


model_score = accuracy_score(y_test, predictions)
model_score


# In[701]:


## Random forest


# In[702]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[703]:


model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)


# In[704]:


predictions = model.predict(X_test)
predictions


# In[705]:


model_score = accuracy_score(y_test, predictions)
model_score


# In[706]:


### KNeighbordsClassifier


# In[707]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[708]:


model = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)


# In[709]:


predictions = model.predict(X_test)
predictions


# In[710]:


model_score = accuracy_score(y_test, predictions)
model_score


# In[711]:


### Logistic Regression


# In[712]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[713]:


model = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)


# In[714]:


predictions = model.predict(X_test)
predictions


# In[715]:


model_score = accuracy_score(y_test, predictions)
model_score


# In[ ]:




