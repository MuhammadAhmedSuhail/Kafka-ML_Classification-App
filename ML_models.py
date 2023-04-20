#!/usr/bin/env python
# coding: utf-8

# In[25]:


import shutil
import os
import zipfile
import rarfile
import patoolib
from pyunpack import Archive
import librosa
import soundfile as sf
import numpy as np
import librosa.display
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
from sklearn.decomposition import PCA
import pickle


# In[2]:


df=pd.read_csv('combined.csv')


# In[3]:


df


# In[4]:


le=LabelEncoder()
df['Label']=le.fit_transform(df.Label)
df


# In[12]:


df.fillna(df.mean(),inplace=True)


# In[13]:


X = df
X=X.drop('Label',axis=1)
y = df['Label']


# In[14]:


X


# In[15]:


y


# In[16]:


print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)


# In[17]:


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.09, random_state=51)
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)
sc = StandardScaler()
sc.fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
clf1=KNeighborsClassifier(n_neighbors=3)
clf1.fit(X_train, y_train)
pred=clf1.predict(X_test)
clf1.score(X_test,y_test)
pickle.dump(clf1,open("knn.pkl","wb"))


# In[20]:


print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[27]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
pred=clf.predict(X_test)
clf.score(X_test,y_test)
pickle.dump(clf,open("decsiontree.pkl","wb"))


# In[22]:


print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)
pred=clf.predict(X_test)
print (pred)
clf.score(X_test,y_test)
pickle.dump(clf,open("randomforest.pkl","wb"))


# In[24]:


print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)

