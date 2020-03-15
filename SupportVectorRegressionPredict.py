# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:50:53 2020

@author: Santosh Sah
"""

import pandas as pd
from SupportVectorRegressionUtils import (readSupportVectorRegressionModel, readSupportVectorRegressionStandardScalerForXTrain,
                                          readSupportVectorRegressionStandardScalerForYTrain)

def predictSupportVectorRegression():
    
    supportVectorRegressionModel = readSupportVectorRegressionModel()
    supportVectorRegressionStandardScalerForXTrain = readSupportVectorRegressionStandardScalerForXTrain()
    supportVectorRegressionStandardScalerForYTrain = readSupportVectorRegressionStandardScalerForYTrain()
    
    inputValue = [[6.5]]
    inputValueDataframe = pd.DataFrame(supportVectorRegressionStandardScalerForXTrain.transform(inputValue))
    
    predictedValue = supportVectorRegressionModel.predict(inputValueDataframe.values)
    
    inversePredictedValue = supportVectorRegressionStandardScalerForYTrain.inverse_transform(predictedValue)
    
    print(inversePredictedValue)

if __name__ == "__main__":
    predictSupportVectorRegression()