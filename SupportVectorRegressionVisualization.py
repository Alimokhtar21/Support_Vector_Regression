# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:53:07 2020

@author: Santosh Sah
"""

import matplotlib.pyplot as plt
import numpy as np
from SupportVectorRegressionUtils import (readSupportVectorRegressionModel, readIndepentDataset, readDependentDataset)

"""
Visualising the DecisionTree Regression results

"""

def visualisingSupportVectorRegression():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    supportVectorRegressionModel = readSupportVectorRegressionModel()
    
    # Visualising the SVR results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, supportVectorRegressionModel.predict(X), color = 'blue')
    plt.title('Truth or Bluff (SVR)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    
    plt.savefig("support_vector__trainingsetresult.png")
    
    plt.show()

"""
Visualising the DecisionTree Regression results (for higher resolution and smoother curve)

"""

def visualisingSupportVectorRegressionInHighResolution():
    
    X = readIndepentDataset()
    y = readDependentDataset()
    supportVectorRegressionModel = readSupportVectorRegressionModel()
    
    # Visualising the SVR results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, supportVectorRegressionModel.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (SVR)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    
    plt.savefig("support_vector__trainingsetresult_high_resolution.png")
    
    plt.show()

if __name__ == "__main__":
    #visualisingSupportVectorRegression()
    visualisingSupportVectorRegressionInHighResolution()
