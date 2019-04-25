#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 19:05:17 2019

@author: ravindra
"""

#benign tumor or malignant tumor

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# import pandas as pd
# cancer_data=pd.read_json('dataset.txt')

#genfromtxt is used to import tabular data from txt file
cancer_data=np.genfromtxt(fname='breast_cancer.data',delimiter=',',dtype=float)

'''
print(cancer_data)
print(len(cancer_data))
print(cancer_data.shape)
'''

# we don't need patient id so,deleting column one
#obj=0 means first column and axis=1 means vertical
cancer_data=np.delete(arr=cancer_data,obj=0,axis=1)

X=cancer_data[:,range(0,9)]
Y=cancer_data[:,9]

#using Imputer to remove nan value
imputer=Imputer(missing_values="NaN",strategy='median',axis=0)
X=imputer.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
#to convert Y_train and Y_test to 1 D using ravel function
Y_train=Y_train.ravel()
Y_test=Y_test.ravel()

kf=0;
max1=0;
n=10
for k in range(n):
    k_value=k+1
    neighbor=KNeighborsClassifier(n_neighbors=k_value,weights='uniform',algorithm='auto')
    neighbor.fit(X_train,Y_train)
    y_pred=neighbor.predict(X_test)
    if(accuracy_score(Y_test,y_pred)*100>max1):
        max1=accuracy_score(Y_test,y_pred)*100
        k_f=k;
    print('Accuracy is',accuracy_score(Y_test,y_pred)*100,'% for k value:',k_value)
print("max:",max1," k:",k) 
# k_value Vs accuracy    