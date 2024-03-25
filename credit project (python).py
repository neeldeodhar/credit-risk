#!/usr/bin/env python
# coding: utf-8

# In[68]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from scipy.io import arff

from sklearn.datasets import fetch_openml





# In[ ]:





# In[69]:


# opening the dataset
# converting dataset into pandas dataframe from a dictionary type file.

credit = fetch_openml(name='credit-g', version=1)

df_features = pd.DataFrame(data = credit['data'])

#defining target
df_target = pd.DataFrame(data = credit['target'])
#appending df_features and df_target
df = pd.concat([df_features, df_target], axis = 1)

# drop missing values
df.dropna()
df.dropna(inplace = True)


# In[70]:


# displaying dataset
# displaying first few rows.
# counting null values
df
df.head()
df.isnull().sum()
df.value_counts()





# In[71]:


# encoding X_train and X_test
enc = OrdinalEncoder()


df[['class','credit_history', 'employment', 'personal_status']] = enc.fit_transform(df[['class', 'credit_history', 'employment', 'personal_status' ]]
)

selected_features = df[['credit_amount', 'age', 'existing_credits', 'num_dependents', 'credit_history', 'employment', 'personal_status']]
display(selected_features)



# In[72]:


#selecting and scaling X variable (selected features)
scaler = StandardScaler()

X = scaler.fit_transform(selected_features)


# In[73]:


# defining y variable

y = df['class'].values


# In[74]:


#creating a split (train/test)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=45)


#creating a split (train/val)

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=45)

# now the train/validate/test split will be 80%/10%/10%


# In[75]:


# training the KNN model and predicting (test)
from sklearn.neighbors import KNeighborsClassifier 
#
K = 11
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score= model.score(X_test, y_test)
# training the same KNN model and predicting (validation)
y_predict = model.predict(X_val)

valscore= model.score(X_val, y_val)

print(str(score))
print ((y_pred))

print(str(valscore))
print ((y_predict))


# In[76]:


#OPTIMIZING THE KNN MODEL

#making a loop to run a range of nearest neighbours to find the best model for testing
err_rate = []
accuracy = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    predictions_i = knn.predict(X_test)
    
    err_rate.append(np.mean(predictions_i != y_test))
    accuracy.append(knn.score(X_test, y_test))
    
#making a loop to run a range of nearest neighbours to find the best model for validating
valerror_rate = []
accuracyval = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    predictions_ival = knn.predict(X_val)
    
    valerror_rate.append(np.mean(predictions_ival != y_val))
    accuracyval.append(knn.score(X_val, y_val))


# In[77]:


# printing accuracy results (testing)
accuracy


# In[78]:


#printing accuracy results for validating
accuracyval


# In[79]:


#plotting accuracy (testing) percentage VS K values
plt.figure(figsize = (10,4))

plt.plot(range(1,20), accuracy, color ="red" )
plt.title("Accuracy percent vs: K value")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy percentage")

plt.show()


# In[80]:


#plotting accuracy (validation) percentage vs K values
plt.figure(figsize = (10,4))

plt.plot(range(1,20), accuracyval, color ="orange" )
plt.title("ValidationAccuracy percent vs: K value")
plt.xlabel("Number of Neighbors")
plt.ylabel("Validation Accuracy Rate")

plt.show()


# In[81]:


print ("Conclusion: Based on above graphs, k = 10 is the optimal point for number of neighbors")


# In[82]:


# printing maximum accuracy (testing) score
max (accuracy)


# In[83]:


# printing maximum accuracy (validating) score
max (accuracyval)


# In[84]:


# running the model for best accuarcy (testing)
bestaccuracy_knn = accuracy.index(max(accuracy)) + 1  
print(bestaccuracy_knn)

# Run the best model

bestaccuracy_knn_model = KNeighborsClassifier(n_neighbors = bestaccuracy_knn)
bestaccuracy_knn_model.fit(X_train, y_train)


print(bestaccuracy_knn_model.score(X_test, y_test))


# In[85]:


# running the model for best accuracy (validation)
bestaccuracyval_knn = accuracyval.index(max(accuracyval)) + 1 
print(bestaccuracyval_knn)

# Run the best model

bestaccuracyval_knn_model = KNeighborsClassifier(n_neighbors = bestaccuracyval_knn)
bestaccuracyval_knn_model.fit(X_train, y_train)

print(bestaccuracyval_knn_model.score(X_val, y_val))


# In[86]:


# displaying confusion matrix (testing)
clf = make_pipeline(StandardScaler(), LogisticRegression(random_state = 0))
clf.fit(X_train, y_train)
y_predclf = clf.predict(X_test)
cm = confusion_matrix(y_test, y_predclf)
cm_display = ConfusionMatrixDisplay(cm).plot()


# In[87]:


# printing confusion matrix for validating
clf = make_pipeline(StandardScaler(), LogisticRegression(random_state = 0))
clf.fit(X_train, y_train)
y_predclf1 = clf.predict(X_val)
cm1 = confusion_matrix(y_val, y_predclf1)
cm_display1 = ConfusionMatrixDisplay(cm1).plot()

