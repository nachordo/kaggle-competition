#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:36:07 2020

@author: ordovas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def y_processing(y):
    return np.log10(y)

def cleaning_1(diamonds):
    diamonds=diamonds.drop(columns=["depth","table"])
    
    diamonds = pd.get_dummies(diamonds,drop_first=True)
    dims=["x","y","z"]
    for i in range(len(dims)):
        for j in range(i,len(dims)):
            col1 = dims[i]
            col2 = dims[j]
            diamonds[f"{col1}*{col2}"] = diamonds[col1]*diamonds[col2]
    
    return diamonds