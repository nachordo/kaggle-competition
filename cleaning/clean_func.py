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


cut={"Fair":1./5, "Good":2./5, "Very Good":3./5, "Premium":4./5, "Ideal":5./5}
color={"D":1./7,"E":2./7,"F":3./7,"G":4./7,"H":5./7,"I":6./7,"J":7./7}
clarity={"I1":1./8, "SI2":2./8, "SI1":3./8, "VS2":4./8, "VS1":5./8, "VVS2":6./8, "VVS1":7./8, "IF":8./8}

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

def cleaning_2(diamonds):
    cut={"Fair":1./5, "Good":2./5, "Very Good":3./5, "Premium":4./5, "Ideal":5./5}
    color={"D":1./7,"E":2./7,"F":3./7,"G":4./7,"H":5./7,"I":6./7,"J":7./7}
    clarity={"I1":1./8, "SI2":2./8, "SI1":3./8, "VS2":4./8, "VS1":5./8, "VVS2":6./8, "VVS1":7./8, "IF":8./8}

    diamonds=diamonds[["carat","cut","color","clarity"]]
    diamonds["color"]=diamonds["color"].replace(color)
    diamonds["cut"]=diamonds["cut"].replace(cut)
    diamonds["clarity"]=diamonds["clarity"].replace(clarity)
    
    dims=["cut","color","clarity"]
    for i in range(len(dims)):
        for j in range(i,len(dims)):
            col1 = dims[i]
            col2 = dims[j]
            diamonds[f"{col1}*{col2}"] = diamonds[col1]*diamonds[col2]
    diamonds["all_cat"]=diamonds["color"]*diamonds["cut"]*diamonds["clarity"]
    return diamonds

def cleaning_3(diamonds):
    diamonds=diamonds[["carat","cut","color","clarity"]]
    diamonds = pd.get_dummies(diamonds,drop_first=True)
    return diamonds    




def cleaning_4(diamonds):
    cut={"Fair":1./5, "Good":2./5, "Very Good":3./5, "Premium":4./5, "Ideal":5./5}
    color={"D":1./7,"E":2./7,"F":3./7,"G":4./7,"H":5./7,"I":6./7,"J":7./7}
    clarity={"I1":1./8, "SI2":2./8, "SI1":3./8, "VS2":4./8, "VS1":5./8, "VVS2":6./8, "VVS1":7./8, "IF":8./8}

    diamonds=diamonds[["carat","cut","color","clarity"]]
    diamonds["color"]=diamonds["color"].replace(color)
    diamonds["cut"]=diamonds["cut"].replace(cut)
    diamonds["clarity"]=diamonds["clarity"].replace(clarity)

    diamonds["all_cat"]=diamonds["color"]*diamonds["cut"]*diamonds["clarity"]
    return diamonds



def cleaning_5(diamonds):
    diamonds=diamonds[["x","z","cut","color","clarity"]]
    diamonds = pd.get_dummies(diamonds,drop_first=True)
    
    return diamonds

def cleaning_6(diamonds):
    cut={"Fair":1./5, "Good":2./5, "Very Good":3./5, "Premium":4./5, "Ideal":5./5}
    color={"D":1./7,"E":2./7,"F":3./7,"G":4./7,"H":5./7,"I":6./7,"J":7./7}
    clarity={"I1":1./8, "SI2":2./8, "SI1":3./8, "VS2":4./8, "VS1":5./8, "VVS2":6./8, "VVS1":7./8, "IF":8./8}

    diamonds=diamonds[["x","cut","color","clarity"]]
    diamonds["color"]=diamonds["color"].replace(color)
    diamonds["cut"]=diamonds["cut"].replace(cut)
    diamonds["clarity"]=diamonds["clarity"].replace(clarity)
    
    dims=["cut","color","clarity"]
    for i in range(len(dims)):
        for j in range(i,len(dims)):
            col1 = dims[i]
            col2 = dims[j]
            diamonds[f"{col1}*{col2}"] = diamonds[col1]*diamonds[col2]
    diamonds["all_cat"]=diamonds["color"]*diamonds["cut"]*diamonds["clarity"]
    return diamonds