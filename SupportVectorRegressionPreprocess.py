# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:50:10 2020

@author: Santosh Sah
"""

from sklearn.preprocessing import StandardScaler
from SupportVectorRegressionUtils import (importSupportVectorRegressionDataset, saveDataSetInPickle, 
                                          saveSupportVectorRegressionStandardScalerForXTrain, saveSupportVectorRegressionStandardScalerForYTrain)

def preprocess():
    
    X, y = importSupportVectorRegressionDataset("Support_Vector_Regression_Position_Salaries.csv")
    
    standardScaler_X = StandardScaler()
    standardScaler_y = StandardScaler()
    
    X = standardScaler_X.fit_transform(X)
    y = standardScaler_y.fit_transform(y)
    
    saveDataSetInPickle(X, y)
    saveSupportVectorRegressionStandardScalerForXTrain(standardScaler_X)
    saveSupportVectorRegressionStandardScalerForYTrain(standardScaler_y)
    

if __name__ == "__main__":
    preprocess()