#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:26:20 2021

@author: dyveke
"""
import numpy as np
import pandas as pd

# Splitting the data into training, validation and test sets (using offset to shift the training and validation sets)
# and splitting the datasets based on date, which is specified by the iTrain_u, iVal_u and iTest_u values.

def Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset):
    
    dX = dX.sort_values(['date']).reset_index(drop=True) #Sorting the data on date
    dY = dY.sort_values(['date']).reset_index(drop=True) #Sorting the data on date
    vDates = dX['date']
    dfDates = pd.DataFrame(vDates)
    
    mask_train = (vDates < max(iTrain_u))
    mask_val = (vDates >= max(iTrain_u)) & (vDates < max(iVal_u))
    mask_test = (vDates >= max(iVal_u)) & (vDates < max(iTest_u))
    
    # Offsetting of the data in the training and validation sets. The test sets are not offset.
    # First: splitting the dates into the three sets, by the upper bounds for each set.
    # Second: we split the datasets into training, validation and test sets, and use dropna to remove empty values:
        
    dTrain_X = dX[mask_train]  
    dTrain_X = dTrain_X.shift(periods = iOffset, freq = None, axis = 0).dropna()
    dTrain_Y = dY[mask_train] 
    dTrain_Y = dTrain_Y.shift(periods = -iOffset, freq = None, axis = 0).dropna()
    dVal_X = dX[mask_val]
    dVal_X = dVal_X.shift(periods = iOffset, freq = None, axis = 0).dropna()
    dVal_Y = dY[mask_val]
    dVal_Y = dVal_Y.shift(periods = -iOffset, freq = None, axis = 0).dropna()
    dTest_X = dX[mask_test] # The test sets are not offset
    dTest_Y = dY[mask_test] # The test sets are not offset
    
    # Third: the values are standardized (within each subset and not overall):
    
    dTrain_X = dTrain_X.values
    dTrain_Y = dTrain_Y.values
    dVal_X = dVal_X.values
    dVal_Y = dVal_Y.values
    dTest_X = dTest_X.values
    dTest_Y = dTest_Y.values
    
    # Fourth: the columns with date and permno is removed from the dataset.
    
    dTrain_X = np.delete(dTrain_X, np.s_[0:2], axis=1)
    dTest_X = np.delete(dTest_X, np.s_[0:2], axis=1)
    dVal_X = np.delete(dVal_X, np.s_[0:2], axis=1)
    dTrain_Y = np.delete(dTrain_Y, np.s_[0:2], axis=1)
    dVal_Y = np.delete(dVal_Y, np.s_[0:2], axis=1)
    dTest_Y = np.delete(dTest_Y, np.s_[0:2], axis=1)
    
    return dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates

def Test_Val_Train_NN1Loop(dX, dY, iTrainu, iValu, iTestu, iOffset, i):
    
    dX = dX.sort_values(['date']).reset_index(drop=True) #Sorting the data on date
    dY = dY.sort_values(['date']).reset_index(drop=True) #Sorting the data on date
    vDates = dX['date']
    dfDates = pd.DataFrame(vDates)
    
    # Offsetting of the data in the training and validation sets. The test sets are not offset.
    # First: splitting the dates into the three sets, by the upper bounds for each set.
    # Second: we split the datasets into training, validation and test sets, and use dropna to remove empty values:
    
    mask_train = (vDates < iTrainu[i])
    mask_val = (vDates >= iTrainu[i]) & (vDates < iValu[i])
    mask_test = (vDates >= iValu[i]) & (vDates < iTestu[i])
    
    # Offsetting of the data in the training and validation sets. The test sets are not offset.
    # First: splitting the dates into the three sets, by the upper bounds for each set.
    # Second: we split the datasets into training, validation and test sets, and use dropna to remove empty values:
        
    dTrain_X = dX[mask_train]  
    dTrain_X = dTrain_X.shift(periods = iOffset, freq = None, axis = 0).dropna()
    dTrain_Y = dY[mask_train] 
    dTrain_Y = dTrain_Y.shift(periods = -iOffset, freq = None, axis = 0).dropna()
    dVal_X = dX[mask_val]
    dVal_X = dVal_X.shift(periods = iOffset, freq = None, axis = 0).dropna()
    dVal_Y = dY[mask_val]
    dVal_Y = dVal_Y.shift(periods = -iOffset, freq = None, axis = 0).dropna()
    dTest_X = dX[mask_test] # The test sets are not offset
    dTest_Y = dY[mask_test] # The test sets are not offset
    
    # Third: the values are standardized (within each subset and not overall):
    
    dTrain_X = dTrain_X.values
    dTrain_Y = dTrain_Y.values
    dVal_X = dVal_X.values
    dVal_Y = dVal_Y.values
    dTest_X = dTest_X.values
    dTest_Y = dTest_Y.values
    
    # Fourth: the columns with date and permno is removed from the dataset.
    
    dTrainX = np.delete(dTrain_X, np.s_[0:2], axis=1)
    dTestX = np.delete(dTest_X, np.s_[0:2], axis=1)
    dValX = np.delete(dVal_X, np.s_[0:2], axis=1)
    dTrainY = np.delete(dTrain_Y, np.s_[0:2], axis=1)
    dValY = np.delete(dVal_Y, np.s_[0:2], axis=1)
    dTestY = np.delete(dTest_Y, np.s_[0:2], axis=1)
    
    return dTrainX, dTrainY, dValX, dValY, dTestX, dTestY, dfDates

def Geom_nlayer(nlayer, basis_layer):
    N1 = int(basis_layer * pow(2, nlayer-1))
    N2 = int(N1/2)
    N3 = int(N2/2)
    N4 = int(N3/2)
    N5 = int(N4/2)
    N6 = int(N5/2)
    N7 = int(N6/2)
    N8 = int(N7/2)
    N9 = int(N8/2)
    N10 = int(N9/2)
    return N1, N2, N3, N4, N5, N6, N7, N8, N9, N10
        
        
#Exporting to latex
def latexTable(df):
    with open('mytable.tex','w') as tf:
        tf.write(df.to_latex())
    return print('Done')
    


