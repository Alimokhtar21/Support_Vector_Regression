# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:52:07 2020

@author: Santosh Sah
"""

from sklearn.svm import SVR
from SupportVectorRegressionUtils import (saveSupportVectorRegressionModel, readIndepentDataset, readDependentDataset)

"""
Train SupportVector regression model 
"""
def trainSupportVectorRegressionModel():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    
    # Fitting SupportVector Regression to the dataset
    supportVectorRegressor = SVR(kernel = "rbf")
    supportVectorRegressor.fit(X, y)
    
    saveSupportVectorRegressionModel(supportVectorRegressor)

if __name__ == "__main__":
    trainSupportVectorRegressionModel()