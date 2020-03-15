# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:46:52 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle

"""
Import dataset and read specific column. Split the dataset in training and testing set.
Data set is very small and hence we are not going to divide the dataset in training and test set.
We will train our model on the whole dataset
"""
def importSupportVectorRegressionDataset(supportVectorRegressionDatasetFileName):
    
    supportVectorRegressionDataset = pd.read_csv(supportVectorRegressionDatasetFileName)
    X = supportVectorRegressionDataset.iloc[:, 1:2].values
    y = supportVectorRegressionDataset.iloc[:, 2:3].values
    
    """
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test
    
    """
    
    return X, y

"""
Save dataset as pickle file
"""
def saveDataSetInPickle(X, y):
    
    #Write X in a picke file
    with open("X.pkl",'wb') as X_Pickle:
        pickle.dump(X, X_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("y.pkl",'wb') as y_Pickle:
        pickle.dump(y, y_Pickle, protocol = 2)

"""
Save SupportVectorRegressionModel as a pickle file.
"""
def saveSupportVectorRegressionModel(supportVectorRegressionModel):
    
    #Write SupportVectorRegressionModel as a picke file
    with open("SupportVectorRegressionModel.pkl",'wb') as SupportVectorRegressionModel_Pickle:
        pickle.dump(supportVectorRegressionModel, SupportVectorRegressionModel_Pickle, protocol = 2)


"""
read SupportVectorRegressionModel from pickle file
"""
def readSupportVectorRegressionModel():
    
    #load SupportVectorRegressionModel model
    with open("SupportVectorRegressionModel.pkl","rb") as SupportVectorRegressionModel:
        supportVectorRegressionModel = pickle.load(SupportVectorRegressionModel)
    
    return supportVectorRegressionModel

"""
read X from pickle file
"""
def readIndepentDataset():
    
    #load y_test
    with open("X.pkl","rb") as X_pickle:
        X = pickle.load(X_pickle)
    
    return X

"""
read y from pickle file
"""
def readDependentDataset():
    
    #load y
    with open("y.pkl","rb") as y_pickle:
        y = pickle.load(y_pickle)
    
    return y


"""
Save standard scalar object as a pickel file for ForXTrain. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveSupportVectorRegressionStandardScalerForXTrain(supportVectorRegressionStandardScalarForXTrain):
    
    #Write SupportVectorRegressionStandardScalerForXTrain in a picke file
    with open("SupportVectorRegressionStandardScalerForXTrain.pkl",'wb') as SupportVectorRegressionStandardScalerForXTrain_Pickle:
        pickle.dump(supportVectorRegressionStandardScalarForXTrain, SupportVectorRegressionStandardScalerForXTrain_Pickle, protocol = 2)


"""
Save standard scalar object as a pickel file for ForYTrain. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveSupportVectorRegressionStandardScalerForYTrain(supportVectorRegressionStandardScalarForYTrain):
    
    #Write SupportVectorRegressionStandardScalerForYTrain in a picke file
    with open("SupportVectorRegressionStandardScalerForYTrain.pkl",'wb') as SupportVectorRegressionStandardScalerForYTrain_Pickle:
        pickle.dump(supportVectorRegressionStandardScalarForYTrain, SupportVectorRegressionStandardScalerForYTrain_Pickle, protocol = 2)


"""
read SupportVectorRegressionStandardScalarForXTrain from pickel file
"""
def readSupportVectorRegressionStandardScalerForXTrain():
    
    #load SupportVectorRegressionStandardScalerForXTrain object
    with open("SupportVectorRegressionStandardScalerForXTrain.pkl","rb") as SupportVectorRegressionStandardScalerForXTrain:
        supportVectorRegressionStandardScalarForXTrain = pickle.load(SupportVectorRegressionStandardScalerForXTrain)
    
    return supportVectorRegressionStandardScalarForXTrain

"""
read SupportVectorRegressionStandardScalarForYTrain from pickel file
"""
def readSupportVectorRegressionStandardScalerForYTrain():
    
    #load SupportVectorRegressionStandardScalerForYTrain object
    with open("SupportVectorRegressionStandardScalerForYTrain.pkl","rb") as SupportVectorRegressionStandardScalerForYTrain:
        supportVectorRegressionStandardScalarForYTrain = pickle.load(SupportVectorRegressionStandardScalerForYTrain)
    
    return supportVectorRegressionStandardScalarForYTrain
