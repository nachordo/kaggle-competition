#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:16:54 2020

@author: ordovas
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8 )
from sklearn import linear_model
linreg = linear_model.LinearRegression().fit(X_train, y_train)