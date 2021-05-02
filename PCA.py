#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:06:41 2021

@author: dyveke
"""

import pickle
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib
#import tensorflow.compat.v1.keras.backend as K
#import shap
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

#Importing and sorting the data
iTrain_u =  np.arange(19890101, 20040101, 10000) # End dates for training set, preliminary
iVal_u = np.arange(19980101, 20130101, 10000) # End dates for validation set, preliminary
iTest_u = np.arange(19990101, 20170101, 10000) # End dates for test set, preliminary

#Splitting the data:
with open('normalized_factor.pkl', 'rb') as fh:
    dX = pickle.load(fh)
with open('excess_returns.pkl', 'rb') as fh:
    dY = pickle.load(fh)

EV_pca = []

#iTrainu = iTrain_u[i]
#iValu = iVal_u[i]
#iTestu = iTest_u[i]
dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates = Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset = 501)

pca = PCA(n_components=94)
dTrain_X_PC_Fit = pca.fit(dTrain_X)
dTrain_X_PC_Transform = pca.transform(dTrain_X)
dVal_X_PC_Transform = pca.transform(dVal_X)
#dTrain_X_PC = pd.DataFrame(data = dTrain_X_PC_Fit)

dEV2 = pca.explained_variance_ratio_
sum_dEV2 = sum(dEV2[:])

Y = dTrain_Y
X = dTrain_X_PC_Transform
val_Y = dVal_Y
val_X = dVal_X_PC_Transform

# Create linear regression object      
regr = linear_model.LinearRegression()        
# Fit      
regr.fit(X, Y)
# Calibration      
Y_c = regr.predict(X)
val_Y_c = regr.predict(val_X)        
# Cross-validation      
Y_cv = cross_val_predict(regr, X, Y, cv=10)
val_Y_cv = cross_val_predict(regr, val_X, val_Y, cv=10)
# Calculate scores for calibration and cross-validation      
score_c = r2_score(Y, Y_c)      
score_cv = r2_score(Y, Y_cv)        
# Calculate mean square error for calibration and cross validation      
mse_c = mean_squared_error(Y, Y_c)
val_mse_c = mean_squared_error(val_Y, val_Y_c)
mse_cv = mean_squared_error(Y, Y_cv)
val_mse_cv = mean_squared_error(val_Y, val_Y_cv)


 

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
    return(mse_c, val_mse_c, mse_cv, val_mse_cv)

def pcr(train_X,train_Y,pc,val_X,val_Y):
    ''' Step 1: PCA on input data'''        
    # Define the PCA object      
    pca = PCA(n_components = pc)
    #pca.reshape(-1,1)     
    # Preprocess (2) Standardize features by removing the mean and scaling to unit variance      
    ###Xstd = StandardScaler().fit_transform(X)        
    # Run PCA producing the reduced variable Xred and select the first pc components      
    #Xreg = pca.fit_transform(train_X)
    pca.fit(train_X)
    #dTrain_X_PC_Fit.reshape(71643 ,pc)
    train_X = pca.transform(train_X)
    val_X = pca.transform(val_X)
    #dTrain_X_PC = pd.DataFrame(data = dTrain_X_PC_Fit)
    #dVal_X_PC = pd.DataFrame(data = dVal_X_PC_Transform)
    ''' Step 2: regression on selected principal components'''        
    mse_c, val_mse_c, mse_cv, val_mse_cv = simple_regression(X = train_X, Y = train_Y, val_X = val_X, val_Y = val_Y)        
    ev = pca.explained_variance_ratio_
    return(mse_c, val_mse_c, mse_cv, val_mse_cv, ev)

npc = 94
mse_c = np.zeros(shape = (npc))
val_mse_c = np.zeros(shape = (npc))
mse_cv = np.zeros(shape = (npc))
val_mse_cv = np.zeros(shape = (npc))
explained_variance = np.zeros(shape = (npc))

for k in range(1,npc+1):
    print (k)
    mse_c[k-1], val_mse_c[k-1], mse_cv[k-1], val_mse_cv[k-1], explained_variance[0:k] = pcr(train_X = dTrain_X, train_Y = dTrain_Y, pc = k, val_X = dVal_X, val_Y = dVal_Y)

#Plots
mean_mse = np.zeros(npc)
for j in range(npc):
    mean_mse[j] = np.mean(val_mse_c[j])
plt.plot(mean_mse)
plt.xlabel('Number of Principal Components')
plt.ylabel('Mean mse')
plt.suptitle('MSE in validation set with Principal Components')
plt.show()

plt.clf()

mean_mse_cv = np.zeros(npc)
for j in range(npc):
    mean_mse_cv[j] = np.mean(val_mse_cv[j])
plt.plot(mean_mse_cv)
plt.xlabel('Number of Principal Components')
plt.ylabel('Mean mse')
plt.suptitle('MSE in validation set with Principal Components, based on cross validation')
plt.show()

plt.clf()

series = np.linspace(1,94,94)

mean_ev = np.zeros(npc)
for j in range(npc):
    mean_ev[j] = explained_variance[j]
plt.bar(x=series,height=dEV2)
plt.rcParams["font.family"] = "serif"
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained variance')
plt.grid(True)
#plt.suptitle('Explained variance with Principal Components')
plt.show()

mean_ev = pd.DataFrame(mean_ev)

with open('mean_ev.tex','w') as kj:
    kj.write(mean_ev.to_latex())

# Plots for the regression where the explained variance of the different principal components based on each factor is compared.

pca_plot = PCA(n_components=(10))
pca_plot.fit(dTrain_X)
PCA_plot1 = np.array(pca_plot.components_[0,:])
plt.plot(PCA_plot1)

#Saved documents
explained_variance_out = pd.DataFrame(explained_variance)
explained_variance_out.to_csv('PCA.ExplainedVarianceForAll94.csv')

val_mse_out = pd.DataFrame(val_mse_c)
val_mse_out.to_csv('PCA.ValMSEForAll94.csv')

val_mse_cv_out = pd.DataFrame(val_mse_cv)
val_mse_cv_out.to_csv('PCA.ValMSE.CV.ForAll94.csv')

#Information criteria, Bai and Ng

counting = dfDates[dfDates["date"] < np.max(iTrain_u)]

iTime = float(len(Counter(counting['date'])))
iN = float(501)

#ICP 1 (k)
IC1_mse = np.zeros(npc)
IC1_mse_log = np.zeros(npc)
IC1_addition = np.zeros(npc)
IC1_Final = np.zeros(npc)
for i in range(npc):
    IC1_mse[i] = mse_c[14,i]
    IC1_mse_log[i] = np.log(IC1_mse[i])
    IC1_addition[i] = i*((iN+iTime)/(iN*iTime))*np.log((iN*iTime)/(iN+iTime))
    IC1_Final[i] = IC1_mse_log[i] + IC1_addition[i]
IC1_best = np.unravel_index(np.argmin(IC1_Final), IC1_Final.shape)

plt.plot(IC1_Final, '.')
plt.xlabel('Number of Principal Components, k')
plt.ylabel('IC_P1 (k)')
plt.suptitle('Values of IC_P1 for different values for k')
plt.show()

#ICP 2 (k)
IC2_mse = np.zeros(npc)
IC2_mse_log = np.zeros(npc)
IC2_addition = np.zeros(npc)
IC2_Final = np.zeros(npc)
for i in range(npc):
    IC2_mse[i] = mse_c[14,i]
    IC2_mse_log[i] = np.log(IC2_mse[i])
    IC2_addition[i] = i*((iN+iTime)/(iN*iTime))*np.log(min(iTime,iN))
    IC2_Final[i] = IC2_mse_log[i] + IC2_addition[i]
IC2_best = np.unravel_index(np.argmin(IC2_Final), IC2_Final.shape)

plt.plot(IC2_Final, '.')
plt.xlabel('Number of Principal Components, k')
plt.ylabel('IC_P2 (k)')
plt.suptitle('Values of IC_P2 for different values for k')
plt.show()


#ICP 3 (k)
IC3_mse = np.zeros(npc)
IC3_mse_log = np.zeros(npc)
IC3_addition = np.zeros(npc)
IC3_Final = np.zeros(npc)
for i in range(npc):
    IC3_mse[i] = mse_c[14,i]
    IC3_mse_log[i] = np.log(IC3_mse[i])
    IC3_addition[i] = i*((np.log(min(iTime,iN)))/(min(iTime,iN)))
    IC3_Final[i] = IC3_mse_log[i] + IC3_addition[i]
IC3_best = np.unravel_index(np.argmin(IC3_Final), IC3_Final.shape)

plt.plot(IC3_Final, '.')
plt.xlabel('Number of Principal Components, k')
plt.ylabel('IC_P3 (k)')
plt.suptitle('Values of IC_P3 for different values for k')
plt.show()

#Information Criteria based on validation MSE

#ICP 1 (k)
IC1_mse_VAL = np.zeros(npc)
IC1_mse_log_VAL = np.zeros(npc)
IC1_addition_VAL = np.zeros(npc)
IC1_Final_VAL = np.zeros(npc)
for i in range(npc):
    IC1_mse_VAL[i] = val_mse_c[14,i]
    IC1_mse_log_VAL[i] = np.log(IC1_mse_VAL[i])
    IC1_addition_VAL[i] = i*((iN+iTime)/(iN*iTime))*np.log((iN*iTime)/(iN+iTime))
    IC1_Final_VAL[i] = IC1_mse_log_VAL[i] + IC1_addition_VAL[i]
IC1_best_VAL = np.unravel_index(np.argmin(IC1_Final_VAL), IC1_Final_VAL.shape)

plt.plot(IC1_Final_VAL, '.')
plt.xlabel('Number of Principal Components, k')
plt.ylabel('IC_P1 (k)')
plt.suptitle('Values of IC_P1 (validation set) for k')
plt.show()

#ICP 2 (k)
IC2_mse_VAL = np.zeros(npc)
IC2_mse_log_VAL = np.zeros(npc)
IC2_addition_VAL = np.zeros(npc)
IC2_Final_VAL = np.zeros(npc)
for i in range(npc):
    IC2_mse_VAL[i] = val_mse_c[14,i]
    IC2_mse_log_VAL[i] = np.log(IC2_mse_VAL[i])
    IC2_addition_VAL[i] = i*((iN+iTime)/(iN*iTime))*np.log(min(iTime,iN))
    IC2_Final_VAL[i] = IC2_mse_log_VAL[i] + IC2_addition_VAL[i]
IC2_best_VAL = np.unravel_index(np.argmin(IC2_Final_VAL), IC2_Final_VAL.shape)

plt.plot(IC2_Final_VAL, '.')
plt.xlabel('Number of Principal Components, k')
plt.ylabel('IC_P2 (k)')
plt.suptitle('Values of IC_P2 (validation set) for k')
plt.show()

#ICP 3 (k)
IC3_mse_VAL = np.zeros(npc)
IC3_mse_log_VAL = np.zeros(npc)
IC3_addition_VAL = np.zeros(npc)
IC3_Final_VAL = np.zeros(npc)
for i in range(npc):
    IC3_mse_VAL[i] = val_mse_c[14,i]
    IC3_mse_log_VAL[i] = np.log(IC3_mse_VAL[i])
    IC3_addition_VAL[i] = i*((np.log(min(iTime,iN)))/(min(iTime,iN)))
    IC3_Final_VAL[i] = IC3_mse_log_VAL[i] + IC3_addition_VAL[i]
IC3_best_VAL = np.unravel_index(np.argmin(IC3_Final_VAL), IC3_Final_VAL.shape)

plt.plot(IC3_Final_VAL, '.')
plt.xlabel('Number of Principal Components, k')
plt.ylabel('IC_P3 (k)')
plt.suptitle('Values of IC_P3 (validation set) for k')
plt.show()

# Finding the number of principal components needed if 95% of the variance should be explained.

sum_EV = np.zeros(npc)
for i in range(1,npc):
    sum_EV[i] = float(sum_EV[i-1]) + float(explained_variance[14,i])




