#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:25:24 2021

@author: dyveke
"""

import pickle
import numpy as np
import pandas as pd
import keras
#import tensorflow as tf
import matplotlib
#import tensorflow.compat.v1.keras.backend as K
#import shap
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import linear_model
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation 
from keras import regularizers, initializers
from keras.layers.normalization import BatchNormalization
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Data_Split import Test_Val_Train, Test_Val_Train_NN1Loop, Geom_nlayer
from NN_models import NN1, NN2, NN3, NN4, NN5, NNLoop

# Importing and splitting the data:

with open('normalized_factor.pkl', 'rb') as fh:
    dX = pickle.load(fh)
with open('excess_returns.pkl', 'rb') as fh:
    dY = pickle.load(fh)

iTrain_u =  np.arange(19890101, 20080101, 10000) # End dates for training set, preliminary
iVal_u = np.arange(19980101, 20160101, 10000) # End dates for validation set, preliminary
iTest_u = np.arange(19990101, 20170101, 10000) # End dates for test set, preliminary

# Using the formula stated in Data_Split

dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates = Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset = 501)

# The relevant loops for finding the optimal weights

def Opt_Hyp_Ridge(X_train, X_tune,  y_train, y_tune,  lambd):
    MSE_R = np.zeros((len(lambd), 1))
    Alpha_R = np.zeros((len(lambd)))
    Coefs_R = np.zeros((len(lambd),94))
    for i in range(len(lambd)):
        reg = linear_model.RidgeCV(alphas = lambd[i]).fit(X_train, y_train)
        MSE_R[i] = mean_squared_error(y_tune, reg.predict(X_tune))
        Alpha_R[i] = reg.alpha_
        Coefs_R[i, :] = reg.coef_
        print(i, MSE_R[i], lambd[i])
    best_lambda_Ridge = lambd[np.argmin(MSE_R)]
    return  reg, MSE_R, Alpha_R, Coefs_R, best_lambda_Ridge


def Opt_Hyp_Lasso(X_train, X_tune,  y_train, y_tune,  lambd):
    MSE_L = np.zeros((len(lambd), 1))
    Alpha_L = np.zeros((len(lambd)))
    Coefs_L = np.zeros((len(lambd),94))
    for i in range(len(lambd)):
        reg = linear_model.LassoCV(alphas = lambd[i].reshape(-1,1), max_iter = 1000000).fit(X_train, y_train)
        MSE_L[i] = mean_squared_error(y_tune, reg.predict(X_tune))
        Alpha_L[i] = reg.alpha_
        Coefs_L[i, :] = reg.coef_
        print(i, MSE_L[i], lambd[i])
    best_lambda_Lasso = lambd[np.argmin(MSE_L)]
    return reg, MSE_L, Alpha_L, Coefs_L, best_lambda_Lasso

def Opt_Hyp_Elastic_Net(X_train, X_tune,  y_train, y_tune, lambd, weight):
    MSE_EN = np.zeros((len(weight)))
    Alpha_EN = np.zeros((len(weight)))
    Coefs_EN = np.zeros((len(weight),94))
    for j in range(len(weight)):
        reg = linear_model.ElasticNetCV(alphas = lambd, l1_ratio = weight[j], max_iter = 100000000).fit(X_train, y_train)
        MSE_EN[j] = mean_squared_error(y_tune, reg.predict(X_tune))
        print(j, MSE_EN[j], weight[j])
        Alpha_EN[j] = reg.alpha_
        Coefs_EN[j, :] = reg.coef_
    best_weight_EN = weight[np.argmin(MSE_EN)]
    return reg, MSE_EN, best_weight_EN, Alpha_EN, Coefs_EN

# Ridge is estimated, and the values for lambda is changed after every iteration to values around the minimum.

#lambd_R = np.logspace(-3,3,30)
#lambd_R = np.linspace(500,5000,50)
#lambd_R = np.linspace(5000,100000,50)
#lambd_R = np.linspace(100000,500000,50)
#lambd_R = np.linspace(393877,410204,50)
lambd_R = np.linspace(401500,403500,50)

Model_R, MSE_R, Alpha_R, Coefs_R, bestr = Opt_Hyp_Ridge(
    X_train = dTrain_X, X_tune = dVal_X, y_train = dTrain_Y, y_tune = dVal_Y, lambd = lambd_R)

series = lambd_R[::-1]

# When doing the plot below change the values of Lambd to be np.linspace(0,4400000), this gives the plot shown in the project.

plt.plot(series, Coefs_R)
plt.xlabel('Value of the L2 Penalty')
plt.ylabel('Ridge Coefficient')
plt.suptitle('Ridge, Coefficients for all Explanatory Variables given L2 Penalty')
plt.grid()
plt.show()

# Lasso is etimated, and the values for lambda is changed after every iteration to values around the minimum.

#lambd_R = np.logspace(-3,3,30)
#lambd_R = np.linspace(0,0.11,50)
#lambd_R = np.linspace(0,0.01,50)
#lambd_R = np.linspace(0,0.001,50)
#lambd_R = np.linspace(0,0.0005,50)
lambd_L = (np.linspace(0.0003331954989587672,0.0004997927484381507,50))

Model_L, MSE_L, Alpha_L, Coefs_L, bestl = Opt_Hyp_Lasso(
    X_train = dTrain_X, X_tune = dVal_X, y_train = dTrain_Y[:,0], y_tune = dVal_Y[:,0], lambd = lambd_L)

series = lambd_L[::-1]

plt.plot(series, Coefs_L)
plt.xlabel('Value of the L1 Penalty')
plt.ylabel('Lasso Coefficient')
plt.suptitle('Lasso, Coefficients for all Explanatory Variables given L1 Penalty')
plt.grid()
plt.show()

# Lastly, the Elastic Net are estimated.

lambd = (np.linspace(401500,403500,50))
l1 = np.linspace(0.00000000000000001,0.1,10)

Model_EN, MSE_EN, besten, Alpha_EN, Coefs_EN = Opt_Hyp_Elastic_Net(X_train = dTrain_X, X_tune = dVal_X, y_train = dTrain_Y[:,0], y_tune = dVal_Y[:,0], lambd = lambd[:], weight = l1)

# The Final Model
lambd = 0.0010101
l1 = 0.4055

mse_en = mean_squared_error(dVal_Y, Model_EN.predict(dVal_X))
PTest_Y_EN = Model_EN.predict(dTest_X)

PTest_Y_EN = pd.DataFrame(PTest_Y_EN)

with open('PTest_Y_EN.csv','w') as tf:
    tf.write(PTest_Y_EN.to_csv())

MSE_EN_Test = mean_squared_error(dTest_Y,PTest_Y_EN)

# The coefficients for the Elastic Net is plotted

series = range(94)

plt.bar(x = series, height = Coefs_EN[0])
plt.xlabel('Explanaotry Variables')
plt.ylabel('Coefficient, scaled by 1e-9')
plt.suptitle('Elastic Net, Coefficients for all Explanatory Variables')
plt.grid()
plt.show()

# The coefficients for the Lasso model is plotted.

lambd_L = np.linspace(0.0000000000001,0.0015) # Used for the plot

Model_L, MSE_L, Alpha_L, Coefs_L, bestl = Opt_Hyp_Lasso(
    X_train = dTrain_X, X_tune = dVal_X, y_train = dTrain_Y[:,0], y_tune = dVal_Y[:,0], lambd = lambd_L)

series = lambd_L

plt.plot(series, Coefs_L)
plt.rcParams["font.family"] = "serif"
plt.xlabel('Value of the L1 Penalty')
plt.ylabel('Lasso Coefficient')
#plt.suptitle('Lasso, Coefficients for all Explanatory Variables given L1 Penalty')
plt.grid()
plt.show()

# The coefficients for the ridge model is plotted.

lambd_R = np.linspace(0.00000001,450000,200) #Used for the plot

Model_R, MSE_R, Alpha_R, Coefs_R, bestr = Opt_Hyp_Ridge(
    X_train = dTrain_X, X_tune = dVal_X, y_train = dTrain_Y, y_tune = dVal_Y, lambd = lambd_R)

series = lambd_R

# When doing the plot below change the values of Lambd to be np.linspace(0,4400000), this gives the plot shown in the project.

plt.plot(series, Coefs_R)
plt.xlabel('Value of the L2 Penalty')
plt.rcParams["font.family"] = "serif"
plt.ylabel('Ridge Coefficient')
#plt.suptitle('Ridge, Coefficients for all Explanatory Variables given L2 Penalty')
plt.grid()
plt.show()
