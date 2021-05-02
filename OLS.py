#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:40:18 2021

@author: dyveke
"""

import pickle
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib
import statsmodels
#import tensorflow.compat.v1.keras.backend as K
#import shap
from numpy import *
from matplotlib import pyplot as plt
from sklearn import linear_model
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation 
from keras import regularizers, initializers
from keras.layers.normalization import BatchNormalization
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from Data_Split import Test_Val_Train, Test_Val_Train_NN1Loop, Geom_nlayer
from NN_models import NN1, NN2, NN3, NN4, NN5, NNLoop

from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict

from statsmodels.tools.validation import (
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like)
from statsmodels import tsa
from statsmodels.tsa import ar_model
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

#Importing and sorting the data
iTrain_u =  np.arange(19890101, 20040101, 10000) # End dates for training set, preliminary
iVal_u = np.arange(19980101, 20130101, 10000) # End dates for validation set, preliminary
iTest_u = np.arange(19990101, 20170101, 10000) # End dates for test set, preliminary

#Splitting the data:
with open('normalized_factor.pkl', 'rb') as fh:
    dX = pickle.load(fh)
with open('excess_returns.pkl', 'rb') as fh:
    dY = pickle.load(fh)

dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates = Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset = 501)

def simple_regression(X,Y,val_X,val_Y):
    # Create linear regression object      
    regr = linear_model.LinearRegression()        
    # Fit      
    regr.fit(X, Y)
    # Calibration      
    Y_c = regr.predict(X)
    val_Y_c = regr.predict(val_X)        
    # Cross-validation      
    Y_cv = cross_val_predict(regr, X, Y, cv=5)
    val_Y_cv = cross_val_predict(regr, val_X, val_Y, cv=5)
    # Calculate scores for calibration and cross-validation
    ###score_c = r2_score(y, y_c)
    ###score_cv = r2_score(y, y_cv)        
    # Calculate mean square error for calibration and cross validation      
    mse_c = mean_squared_error(Y, Y_c)
    val_mse_c = mean_squared_error(val_Y, val_Y_c)
    mse_cv = mean_squared_error(Y, Y_cv)
    val_mse_cv = mean_squared_error(val_Y, val_Y_cv)
    print(regr.coef_)
    return(regr, regr.coef_, mse_c, val_mse_c, mse_cv, val_mse_cv)

model, coefs, mse_c, val_mse_c, mse_cv, val_mse_cv = simple_regression(X=dTrain_X, Y=dTrain_Y, val_X=dVal_X, val_Y=dVal_Y)

PTest_Y_OLS = model.predict(dTest_X)
mse_ols = mean_squared_error(dTest_Y,PTest_Y_OLS)

PTest_Y_OLS = pd.DataFrame(PTest_Y_OLS)

# Testing for serial correlation and heteroskedasticity
# First the heteroskedasticity

model = sm.OLS(dTrain_Y, dTrain_X)

residuals = dTrain_Y - model.predict(dTrain_X)

lm, lmpvalue, fvalue, fpvalue = statsmodels.stats.diagnostic.het_breuschpagan(residuals, exog_het = dTrain_X)

# Then the serial correlation:

model1 = statsmodels.regression.linear_model.OLSResults(model,coefs)
sb, sbpvalue = statsmodels.stats.diagnostic.acorr_ljungbox(dTrain_Y,6012)

# The new models:
    
model1 = statsmodels.regression.linear_model.RegressionResults(model, model.fit().params)#.HC0_se

reg = sm.OLS(dTrain_Y,dTrain_X).fit()
hac = reg.get_robustcov_results(cov_type = 'HAC', maxlags = 6012)

confidenceintervals = hac.conf_int()



plt.plot(series, plottingOLS.T[0,:], 'r.')
#plt.plot((confidenceintervals.T[0,:], confidenceintervals.T[1,:]), (series,series), 'b-')
plt.fill_between(series, (confidenceintervals.T[0,:]), (confidenceintervals.T[1,:]), color='b', alpha=.1)
plt.xlabel('Explanatory Variables')
plt.rcParams["font.family"] = "serif"
plt.ylabel('OLS Coefficient')
#plt.suptitle('Ordinary Least Squares, Coefficients for all Explanatory Variables')
plt.grid(True)
plt.show()

for i in range(94):
    if sign(confidenceintervals.T[0,i]) == sign(confidenceintervals.T[1,i]):
        print(i+1,'is significant')
    else:
        print(i+1,'is not')

tsta = np.zeros((94))

for i in range(94):
    tsta[i] = hac.params[i] / hac.bse[i]

plt.bar(x = series, height = tsta)
plt.xlabel('Explanatory Variables')
plt.rcParams["font.family"] = "serif"
plt.ylabel('t-stat for OLS Coefficient')
#plt.suptitle('Ordinary Least Squares, Coefficients for all Explanatory Variables')
plt.grid(True)
plt.show()

model = sm.OLS(dTrain_Y,dTrain_X)
results = model.fit()
results.bse

series = np.linspace(1,94,94)

plottingOLS = np.zeros((94,2))
t_stats = np.zeros((94))

for i in range(94):
    plottingOLS[i,0] = results.params[i]
    plottingOLS[i,1] = -results.bse[i]

plt.bar(x = series, height = plottingOLS.T[0,:], width = 0.2)
plt.scatter(series, plottingOLS.T[1,:], c='black', marker = '.')
plt.xlabel('Explanatory Variables')
plt.rcParams["font.family"] = "serif"
plt.ylabel('OLS Coefficient')
#plt.suptitle('Ordinary Least Squares, Coefficients for all Explanatory Variables')
plt.grid(True)
plt.show()


with open('PTest_Y_OLS.csv','w') as tf:
    tf.write(PTest_Y_OLS.to_csv())

series = np.linspace(0,94,94)

plt.bar(x = series, height = coefs.T[:,0])
plt.xlabel('Explanatory Variables')
plt.rcParams["font.family"] = "serif"
plt.ylabel('OLS Coefficient')
#plt.suptitle('Ordinary Least Squares, Coefficients for all Explanatory Variables')
plt.grid()
plt.show()

Asset200 = range(200,24248,501)
dValY = pd.DataFrame(dVal_Y)

# The AutoRegressive model (1) lag

armodel = AutoReg(endog = dTrain_Y, lags = [501])
armodel.fit(cov_type = 'nonrobust')

dVal_Y = pd.DataFrame(dVal_Y)
Pred_YVal_AR1 = armodel.predict(dVal_Y)

dTrainY = pd.DataFrame(dTrain_Y)
dValY = pd.DataFrame(dVal_Y)

shiftedY = dTrainY.shift(501).dropna()
dTrain_YAR = dTrain_Y[501:]
shiftedYVal = dValY.shift(501).dropna()
dVal_YAR = dVal_Y[501:]


dTrain_Y = pd.DataFrame(dTrain_Y)
dVal_Y = pd.DataFrame(dVal_Y)

mse_cAR, val_mse_cAR, mse_cvAR, val_mse_cvAR = simple_regression(X=dTrain_Y.shift(periods = 501).fillna(value = 0), Y=dTrain_Y, val_X=dVal_Y.shift(periods = 501).fillna(value=0), val_Y=dVal_Y)


model, coefs, mse_c, val_mse_c, mse_cv, val_mse_cv = simple_regression(
    X=shiftedY, Y=dTrain_YAR, val_X=shiftedYVal, val_Y=dVal_YAR)

dTest_Y = pd.DataFrame(dTest_Y)
PTest_Y_AR = model.predict(dTest_Y.shift(501).fillna(0))
mse_ar = mean_squared_error(dTest_Y,PTest_Y_AR)

PTest_Y_AR = pd.DataFrame(PTest_Y_AR)
with open('PTest_Y_AR.csv','w') as tf:
    tf.write(PTest_Y_AR.to_csv())

# The historical mean for the training sample is calculated.

hist = np.mean(dTrain_Y)

PTest_Y_H = np.empty((24048,1))
PTest_Y_H[:] = hist

mse_h = mean_squared_error(dTest_Y,PTest_Y_H)

PTest_Y_H = pd.DataFrame(PTest_Y_H)

with open('PTest_Y_H.csv','w') as tf:
    tf.write(PTest_Y_H.to_csv())
    
# Robustness checks:

histLoop = np.zeros((15))

for i in range(15):
    dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates = Test_Val_Train_NN1Loop(dX, dY, iTrain_u, iVal_u, iTest_u, 501)
    hist[i] = np.mean(dTrain_Y)
    






