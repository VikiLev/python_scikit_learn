#!/usr/bin/env python
# coding: utf-8

# In[186]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt


# In[159]:


## Loading data


# In[160]:


df = pd.read_csv('vehicles.csv')
df


# In[161]:


## Cleaning


# In[162]:


df.isnull().sum()


# In[163]:


df['Income'].fillna(0.0, inplace=True)
df.head()


# In[164]:


## Encoding


# In[165]:


## option 1
# df.replace({
#     'Gender': {
#         'male': 0,
#         'female': 1
        
#     }
# }, inplace=True)
# df.head()


# In[166]:


## option 2
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df.head()


# In[167]:


df.dtypes


# In[168]:


## Model


# In[169]:


X = df.drop(columns='Favorite Transport')
X.head()


# In[170]:


y = df['Favorite Transport']
y.head()


# In[171]:


model = DecisionTreeClassifier()
model.fit(X, y)


# In[172]:


## Prediction


# In[173]:


test_df = pd.DataFrame({
    'Age': [12],
    'Gender': [0],
    'Income': [0.0]
})
test_df


# In[174]:


model.predict(test_df)


# In[175]:


## Evaluation


# In[176]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape


# In[177]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[178]:


X_test


# In[179]:


predictions = model.predict(X_test)
predictions


# In[180]:


model_accuracy_score = accuracy_score(y_test, predictions)
model_accuracy_score


# In[182]:


## Exporting to the DOT file


# In[185]:


tree.export_graphviz(model, out_file='decision_tree_model.dot', feature_names=['Age', 'Gender', 'Income'], filled=True, class_names=sorted(y.unique()))


# In[187]:


## Charts


# In[188]:


sns.countplot(x=df['Gender'], hue=df['Favorite Transport'])


# In[190]:


sns.histplot(x=df['Income'], hue=df['Favorite Transport'])


# In[ ]:




