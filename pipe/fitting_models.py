#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:16:54 2020

@author: ordovas
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error



from sklearn.model_selection import train_test_split
def first_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8 )
    
    linreg = linear_model.LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_train)
    print("LinearRegression")
    rmse = mean_squared_error(10**y_train.values, 10**y_pred, squared=False)
    print("RMSE (train) =",rmse)
    y_pred = linreg.predict(X_test)
    rmse = mean_squared_error(10**y_test.values, 10**y_pred, squared=False)
    print("RMSE (test) =",rmse)
    return linreg