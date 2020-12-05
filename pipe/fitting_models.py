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
from pipe.grid_searching import *
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split


def first_test(X,y):
    print("LinearRegression")
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.85 )
    linreg = linear_model.LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_train)
    
    y_pred = linreg.predict(X_test)
    rmse = mean_squared_error(10**y_test.values, 10**y_pred, squared=False)
    print("RMSE (test) =",rmse)
    return linreg, rmse

def forest_fit(X,y):
    print("Forest")
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8 )
    func = random_tree_grid(X_train, y_train)
    y_pred = func.predict(X_test)
    rmse = mean_squared_error(10**y_test.values, 10**y_pred, squared=False)
    print("RMSE (test) =",rmse)
    return func, rmse

def l2l1_fit(X,y):
    print("L2 L1")
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8 )
    func = l2_l1_grid(X_train, y_train)
    y_pred = func.predict(X_test)
    rmse = mean_squared_error(10**y_test.values, 10**y_pred, squared=False)
    print("RMSE (test) =",rmse)
    return func, rmse

def knn_fit(X,y):
    print("KNN")
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8 )
    func = knn_grid(X_train, y_train)
    y_pred = func.predict(X_test)
    rmse = mean_squared_error(10**y_test.values, 10**y_pred, squared=False)
    print("RMSE (test) =",rmse)
    return func, rmse

def best_func(X,y):
    best_one = None
    best_rmse = 1_000_000
    #lambda a,b: l2l1_fit(a,b),
    f_list=[lambda a,b: first_test(a,b),
            
            lambda a,b: knn_fit(a,b),
            lambda a,b: forest_fit(a,b) ]
    
    for func in f_list:
        fnc,rmse = func(X,y)
        if rmse < best_rmse:
            print("Choosen one best",rmse, best_rmse,rmse < best_rmse)
            best_rmse=rmse
            best_one=fnc
    
    return best_one,best_rmse