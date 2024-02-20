#!/usr/bin/env python
# coding: utf-8

# In[39]:


from sklearn.datasets import load_iris #our labels and targets comes from here
from sklearn.model_selection import train_test_split # allows us to split training and testing data
from sklearn.neighbors import KNeighborsClassifier # allows us to use KNN Algorithm
from sklearn.preprocessing import StandardScaler, minmax_scale # allows us to scale our features



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


X1 = flower_data.data
# print(X)
#X = minmax_scale(X1)
scaler = StandardScaler() 
X = scaler.fit_transform(X1)
#print(X)
#print(X1)
y = flower_data.target
# print(y)

trainX, testX, trainy, testy = train_test_split(X,y, test_size=0.1, random_state=1)

# print(testy)

#define KNeighborsClassifier and set values for nearest neighbors and power for calculating distance.
knn = KNeighborsClassifier(n_neighbors =1, p = 1)

#train the model
knn.fit(trainX,trainy)

#make predictions based on training set
knn.predict(testX)

#lines below compare the actual values with our predictions
print(knn.predict(testX))
print(testy)


# In[ ]:





# In[ ]:




