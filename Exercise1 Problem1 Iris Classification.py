#!/usr/bin/env python
# coding: utf-8

# In[110]:


from sklearn.datasets import load_iris #our labels and targets comes from here
from sklearn.model_selection import train_test_split # allows us to split training and testing data
from sklearn.neighbors import KNeighborsClassifier



flower_data = load_iris()
#print(dir(flower_data))
#['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

#print(flower_data['filename'])
#iris.csv

#print(type(flower_data))
#<class 'sklearn.utils._bunch.Bunch'>

#print(len(flower_data))
#8

#print(len(flower_data['data']))
#150

#print(flower_data['target'])

#print(flower_data['target_names'])


X = flower_data.data
# print(X)
y = flower_data.target
# print(y)

trainX, testX, trainy, testy = train_test_split(X,y, test_size=0.1, random_state=42)

# print(testy)

#define KNeighborsClassifier and set values for nearest neighbors and power for calculating distance.
knn = KNeighborsClassifier(n_neighbors = 3, p = 2)

#train the model
knn.fit(trainX,trainy)

#make predictions based on training set
knn.predict(testX)

#lines 48 and 49 compare the actual values with our predictions
#print(knn.predict(testX))
#print(testy)


# In[ ]:




