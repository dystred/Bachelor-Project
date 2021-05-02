#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:55:12 2021

@author: dyveke
"""


import pickle
import numpy as np
import pandas as pd
import keras
import statistics
import statsmodels
#import tensorflow as tf
import matplotlib
#import tensorflow.compat.v1.keras.backend as K
#import shap
from scipy.stats import norm
from statsmodels.api import formula as form
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
from scipy.stats import t

# Importing the data to obtain the realized values of Y.

with open('normalized_factor.pkl', 'rb') as fh:
    dX = pickle.load(fh)
with open('excess_returns.pkl', 'rb') as fh:
    dY = pickle.load(fh)

iTrain_u =  np.arange(19890101, 20040101, 10000)     # End dates for training set, preliminary
iVal_u = np.arange(19980101, 20130101, 10000)        # End dates for validation set, preliminary
iTest_u = np.arange(19990101, 20170101, 10000)       # End dates for test set, preliminary

dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates = Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset = 501)

# Shifting the test set to evaluate the forecasts for time t with the realized value at time t+1.

dTest_Y = pd.DataFrame(dTest_Y)
dTest_Y = dTest_Y.shift(periods = -501, freq = None, axis = 0).dropna()

# Importing the predicted values for Y for the different models.

#Historical Mean
PTest_Y_H = pd.read_csv("PTest_Y_H.csv")
PTest_Y_H = PTest_Y_H.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_H = np.array(PTest_Y_H)
PTest_Y_H = PTest_Y_H[ : ,1]
PTest_Y_H = PTest_Y_H.reshape(-1,1)
L_H = (dTest_Y - PTest_Y_H)*(dTest_Y - PTest_Y_H)

#AR(1)
PTest_Y_AR = pd.read_csv("PTest_Y_AR.csv")
PTest_Y_AR = PTest_Y_AR.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_AR = np.array(PTest_Y_AR)
PTest_Y_AR = PTest_Y_AR[ : ,1]
PTest_Y_AR = PTest_Y_AR.reshape(-1,1)
L_AR = (dTest_Y - PTest_Y_AR)*(dTest_Y - PTest_Y_AR)

#OLS
PTest_Y_OLS = pd.read_csv("PTest_Y_OLS.csv")
PTest_Y_OLS = PTest_Y_OLS.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_OLS = np.array(PTest_Y_OLS)
PTest_Y_OLS = PTest_Y_OLS[ : ,1]
PTest_Y_OLS = PTest_Y_OLS.reshape(-1,1)
L_OLS = (dTest_Y - PTest_Y_OLS)*(dTest_Y - PTest_Y_OLS)

#Elastic Net
PTest_Y_EN = pd.read_csv("PTest_Y_EN.csv")
PTest_Y_EN = PTest_Y_EN.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_EN = np.array(PTest_Y_EN)
PTest_Y_EN = PTest_Y_EN[ : ,1]
PTest_Y_EN = PTest_Y_EN.reshape(-1,1)
L_EN = (dTest_Y - PTest_Y_EN)*(dTest_Y - PTest_Y_EN)

#PCA, 4factors
PTest_Y_PCAs = pd.read_csv("PTest_Y_PCAs.csv")
PTest_Y_PCAs = PTest_Y_PCAs.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_PCAs = np.array(PTest_Y_PCAs)
PTest_Y_PCAs = PTest_Y_PCAs[ : ,1]
PTest_Y_PCAs = PTest_Y_PCAs.reshape(-1,1)
L_PCAs = (dTest_Y - PTest_Y_PCAs)*(dTest_Y - PTest_Y_PCAs)

#PCA, 6factors
PTest_Y_PCA6s = pd.read_csv("PTest_Y_PCA6s.csv")
PTest_Y_PCA6s = PTest_Y_PCA6s.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_PCA6s = np.array(PTest_Y_PCA6s)
PTest_Y_PCA6s = PTest_Y_PCA6s[ : ,1]
PTest_Y_PCA6s = PTest_Y_PCA6s.reshape(-1,1)
L_PCA6s = (dTest_Y - PTest_Y_PCA6s)*(dTest_Y - PTest_Y_PCA6s)

#PCA, 4factors and additional lags (8)
PTest_Y_PCAe = pd.read_csv("PTest_Y_PCAe.csv")
PTest_Y_PCAe = PTest_Y_PCAe.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_PCAe = np.array(PTest_Y_PCAe)
PTest_Y_PCAe = PTest_Y_PCAe[ : ,1]
PTest_Y_PCAe = PTest_Y_PCAe.reshape(-1,1)
L_PCAe = (dTest_Y - PTest_Y_PCAe)*(dTest_Y - PTest_Y_PCAe)

#FFN, Neural Net
PTest_Y_FFN = pd.read_csv("PTest_Y_FFN.csv")
PTest_Y_FFN = PTest_Y_FFN.shift(periods = 501, freq = None, axis = 0).dropna()
PTest_Y_FFN = np.array(PTest_Y_FFN)
PTest_Y_FFN = PTest_Y_FFN[ : ,1]
PTest_Y_FFN = PTest_Y_FFN.reshape(-1,1)
L_FFN = (dTest_Y - PTest_Y_FFN)*(dTest_Y - PTest_Y_FFN)

# Mean squared error:

Loss = np.concatenate((L_H,L_AR,L_OLS,L_EN,L_PCAs,L_PCA6s,L_PCAe,L_FFN),axis = 1)

MSE = np.zeros((1,8))
MSFE = np.zeros((1,8))

for i in range(8):
    MSE[0,i] = (Loss[:,i]*Loss[:,i]).mean()
    MSFE[0,i] = MSE[0,i]/MSE[0,0]
    

MSE = pd.DataFrame(MSE)
with open('MSFE.tex','w') as tf:
    tf.write(MSE.to_latex())
    
std = np.zeros((1,8))
for i in range(8):
    std[0,i] = np.var(a = Loss[:,i])

# The function for calculating the test is:
def my_dm_test(y_test, y_pred1, y_pred2, N, h):
    """
    Description : this function runs the Diebold and Mariano test of hypothesis
                  for differnt accuracy in the forecasts of model 1 and model 2.
                  Specifically, it tests the null that the new have same accuracy against
                  the alternative that model 2 has better forecast accuracy.
                  I wrote the function starting from the one of John Tsang,
                  available in his Github repository. 
                  At each point in time, we take the cross sectional mean of the
                  difference in squared errors.
    Parameters
    ----------
    y_test : True values
    y_pred1 : Prediction of the first model
    y_pred2 : Prediction of the second model
    N : Number of assets
    d : number of time steps when forecasting. Defualt is 1

    Returns
    -------
    rt : test statistics and p value for the Diebold and Mariano test 

    """
    y_test = np.array(y_test)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    m = int(y_test.shape[0]/N) #Number of time perios
    y_test = y_test.reshape(m, N) #Reshape as matrix (num of time periods x num of assets)
    y_pred1 = y_pred1.reshape(m, N)
    y_pred2 = y_pred2.reshape(m, N)
    e1 = np.array([y_test[i, :]-y_pred1[i, :] for i in range(m)])#Get errors for model 1
    e2 = np.array([y_test[i, :]-y_pred2[i, :] for i in range(m)])#Get errors for model 2
    d = np.mean(e1**2-e2**2, 1)#Get time series of d
    mean_d = d.mean()#Get average of d
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d, len(d),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/m
    DM_stat=V_d**(-0.5)*mean_d
    #Adjustment following HArvey (1995)
    harvey_adj=((m+1-2*h+h*(h-1)/m)/m)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = m - 1)
    # Construct named tuple for return
    import collections
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    rt = dm_return(DM = DM_stat, p_value = p_value)
    return rt

# Pairwise comparison:

dmtest = my_dm_test(dTest_Y, PTest_Y_H, PTest_Y_AR,501,1)    

Comparison = np.zeros((6,8))

Loss = np.concatenate((L_H,L_AR,L_OLS,L_EN,L_PCAs,L_PCA6s,L_PCAe,L_FFN),axis = 1)
Pred = np.concatenate((dTest_Y,PTest_Y_H,PTest_Y_AR,PTest_Y_OLS,PTest_Y_EN,PTest_Y_PCAs,PTest_Y_PCA6s,PTest_Y_PCAe,PTest_Y_FFN),axis = 1)

for i in range(1,9):
    for j in range(0,3):
        DM = my_dm_test(Pred[:,0],Pred[:,j+1],Pred[:,i],501,1)
        Comparison[(j*2),i-1] = DM[0]
        Comparison[(j*2+1),i-1] = DM[1]

Final = pd.DataFrame(data = Comparison, index = ['Historical Mean', 'P-Value', 'AR(1)', 'P-Value', 'OLS', 'P-Value'], 
                     columns = ['Historical Mean', 'AR(1)', 'OLS', 'Elastic Net', 'PCA, 4 factors','PCA, 6 factors','PCA, entire', 'FFN'])

with open('Diebold_Mariano.tex','w') as tf:
    tf.write(Final.to_latex())

# Model confidence set:

from arch.bootstrap import MCS as getMCS
np.random.seed(17)

mcs = getMCS(Loss[:,0:8], size = 0.01, reps = 10000)
mcs.compute()
print('0.01',mcs.included, mcs.pvalues)

mcs = getMCS(Loss[:,0:8], size = 0.05, reps = 10000)
mcs.compute()
print('0.05',mcs.included, mcs.pvalues)

mcs = getMCS(Loss[:,0:8], size = 0.1, reps = 10000)
mcs.compute()
print('0.1',mcs.included, mcs.pvalues)

# Plots for performance

meanH = np.zeros((501))
meanAR = np.zeros((501))
meanOLS = np.zeros((501))
meanEN = np.zeros((501))
meanPCA4 = np.zeros((501))
meanPCA6 = np.zeros((501))
meanPCAe = np.zeros((501))
meanFFN = np.zeros((501))
meanY = np.zeros((501))

dTest_Y = np.array(dTest_Y)

for i in range(501):
    asset = range(0+i,len(dTest_Y)+i,501)
    meanH[i] = PTest_Y_H[asset].mean()
    meanAR[i] = PTest_Y_AR[asset].mean()
    meanOLS[i] = PTest_Y_OLS[asset].mean()
    meanEN[i] = PTest_Y_EN[asset].mean()
    meanPCA4[i] = PTest_Y_PCAs[asset].mean()
    meanPCA6[i] = PTest_Y_PCA6s[asset].mean()
    meanPCAe[i] = PTest_Y_PCAe[asset].mean()
    meanFFN[i] = PTest_Y_FFN[asset].mean()
    meanY[i] = dTest_Y[asset].mean()

# Plots

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanH,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by Hist. Mean')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanAR,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by AR(1)')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanOLS,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by OLS')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanEN,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by Elastic Net')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanPCA4,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by PCA, 4 factors')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanPCA6,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by PCA, 6 factors')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanPCAe,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by PCA, entire')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

plt.plot(meanY,meanY, linewidth=1, color = 'black')
plt.scatter(meanFFN,meanY, s=2, color = "C4")
plt.grid()
plt.rcParams["font.family"] = "serif"
plt.xlabel('Mean Returns Predicted by FFN')
plt.ylabel('Mean Realized Returns')
plt.xlim(-0.04,0.13)
plt.ylim(-0.04,0.13)

# Plotting of the cumulative sum of MSE for the different models.

def CumulativeSum(y_test,y_pred1,b1,b2,b3,N):
    y_test = np.array(y_test)
    y_pred1 = np.array(y_pred1)
    b1 = np.array(b1)
    b2 = np.array(b2)
    b3 = np.array(b3)
    m = int(y_test.shape[0]/N) #Number of time perios
    y_test = y_test.reshape(m, N) #Reshape as matrix (num of time periods x num of assets)
    y_pred1 = y_pred1.reshape(m, N)
    b1 = b1.reshape(m, N)
    b2 = b2.reshape(m, N)
    b3 = b3.reshape(m, N)
    e1 = np.array([y_test[i, :]-y_pred1[i, :] for i in range(m)])#Get errors for model 1
    eb1 = np.array([y_test[i, :]-b1[i, :] for i in range(m)])#Get errors for model 2
    eb2 = np.array([y_test[i, :]-b2[i, :] for i in range(m)])#Get errors for model 2
    eb3 = np.array([y_test[i, :]-b3[i, :] for i in range(m)])#Get errors for model 2
    d1 = np.mean(e1**2-eb1**2, 1)#Get time series of d
    d2 = np.mean(e1**2-eb2**2, 1)#Get time series of d
    d3 = np.mean(e1**2-eb3**2, 1)#Get time series of d
    dcumu1 = np.cumsum(d1)
    dcumu2 = np.cumsum(d2)
    dcumu3 = np.cumsum(d3)
    return dcumu1, dcumu2, dcumu3

dcumu = np.zeros((15,47))

for i in range(5):
    dcumu[(i*3),:], dcumu[(i*3+1),:],dcumu[(i*3+2),:] = CumulativeSum(Pred[:,0], Pred[:,4+i], Pred[:,1], Pred[:,2], Pred[:,3], 501)

# First the Elastic Net:

plt.plot(dcumu[0,:], linewidth=1, color = 'black', label='EN - Hist. Mean')
plt.plot(dcumu[1,:], linewidth=1, color = 'purple', label='EN - AR(1)')
plt.plot(dcumu[2,:], linewidth=1, color = 'red', label='EN - OLS')
plt.grid()
plt.hlines(y=0, xmin=0,xmax=46, color='black')
plt.legend(loc=3)
plt.rcParams["font.family"] = "serif"
plt.xlabel('Months after January 2013')
plt.ylabel('MSE')
plt.xlim(left=0,right=46)

plt.plot(dcumu[3,:], linewidth=1, color = 'black', label='PCA4s - Hist. Mean')
plt.plot(dcumu[4,:], linewidth=1, color = 'purple', label='PCA4s - AR(1)')
plt.plot(dcumu[5,:], linewidth=1, color = 'red', label='PCA4s - OLS')
plt.grid()
plt.hlines(y=0, xmin=0,xmax=46, color='black')
plt.legend(loc=2)
plt.rcParams["font.family"] = "serif"
plt.xlabel('Months after January 2013')
plt.ylabel('MSE')
plt.xlim(left=0,right=46)

plt.plot(dcumu[6,:], linewidth=1, color = 'black', label='PCA6s - Hist. Mean')
plt.plot(dcumu[7,:], linewidth=1, color = 'purple', label='PCA6s - AR(1)')
plt.plot(dcumu[8,:], linewidth=1, color = 'red', label='PCA6s - OLS')
plt.grid()
plt.hlines(y=0, xmin=0,xmax=46, color='black')
plt.legend(loc=2)
plt.rcParams["font.family"] = "serif"
plt.xlabel('Months after January 2013')
plt.ylabel('MSE')
plt.xlim(left=0,right=46)

plt.plot(dcumu[9,:], linewidth=1, color = 'black', label='PCAe - Hist. Mean')
plt.plot(dcumu[10,:], linewidth=1, color = 'purple', label='PCAe - AR(1)')
plt.plot(dcumu[11,:], linewidth=1, color = 'red', label='PCAe - OLS')
plt.grid()
plt.hlines(y=0, xmin=0,xmax=46, color='black')
plt.legend(loc=2)
plt.rcParams["font.family"] = "serif"
plt.xlabel('Months after January 2013')
plt.ylabel('MSE')
plt.xlim(left=0,right=46)

plt.plot(dcumu[12,:], linewidth=1, color = 'black', label='FFN - Hist. Mean')
plt.plot(dcumu[13,:], linewidth=1, color = 'purple', label='FFN - AR(1)')
plt.plot(dcumu[14,:], linewidth=1, color = 'red', label='FFN - OLS')
plt.grid()
plt.hlines(y=0, xmin=0,xmax=46, color='black')
plt.legend(loc=4)
plt.rcParams["font.family"] = "serif"
plt.xlabel('Months after January 2013')
plt.ylabel('MSE')
plt.xlim(left=0,right=46)







