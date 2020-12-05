#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:21:38 2020

@author: ordovas
"""
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def random_tree_grid(X,y):
    rfc = RandomForestRegressor(n_jobs=-1,max_features= 'sqrt'
                                 ,n_estimators=50, criterion = "mse") 
    
    param_grid = { 
        'n_estimators': [10,25,40,50,60,75,100],
        'min_samples_split': [5,25,50,100],
        'min_samples_leaf': [5,25,50,100],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    CV_rfc = GridSearchCV(estimator=rfc,verbose=1, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X, y)
    
    rfc = RandomForestRegressor(n_jobs=-1,**CV_rfc.best_params_, criterion = "mse")
    rfc.fit(X,y)
    
    return rfc


def l2_l1_grid(X,y):
    parametersGrid = {"max_iter": [1_000, 5_000],
                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0,1.0,0.1)}

    eNet = ElasticNet()
    grid = GridSearchCV(eNet, parametersGrid,verbose=1, scoring='r2', cv=5)
    grid.fit(X, y)
    
    l2l1 = ElasticNet(**grid.best_params_)
    l2l1.fit(X,y)
    return l2l1



def knn_grid(X,y):
    params = {'n_neighbors':[3,5,9,13,15,31,51,61,75],
              'weights':['uniform', 'distance']}

    grid = GridSearchCV(
     estimator=KNeighborsRegressor(),
     param_grid=params,
     verbose=1,
     scoring="neg_mean_squared_error",
     return_train_score=True
     )
    
    grid.fit(X,y)
    knn=KNeighborsRegressor(**grid.best_params_)
    knn.fit(X,y)
    return knn

