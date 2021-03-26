#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:39:39 2021

@author: dyveke
"""

import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

# First: the functions are defined

def minimum_index(X, nfa):
    '''
    Finding the minimum of each column in X, thus returning the index value of the minimum value. Minimum of column i, in x[:,i].

    '''
    mins = X.argmin(axis=0)
    assert sum(X == X[mins]) == 1, 'Minimum value occurs more than once.'
    return mins

def Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset):
    
    dX = dX.sort_values(['date']).reset_index(drop=True) #Sorting the data on date
    dY = dY.sort_values(['date']).reset_index(drop=True) #Sorting the data on date
    vDates = dX['date']
    dfDates = pd.DataFrame(vDates)
    
    # Offsetting of the data in the training and validation sets. The test sets are not offset.
    # First: splitting the dates into the three sets, by the upper bounds for each set.
    # Second: we split the datasets into training, validation and test sets, and use dropna to remove empty values:
        
    dTrain_X = dX[dX["date"] < np.max(iTrain_u)]  
    dTrain_X = dTrain_X.shift(periods = iOffset, freq = None, axis = 0).dropna()
    dTrain_Y = dY[dY["date"] < np.max(iTrain_u)] 
    dTrain_Y = dTrain_Y.shift(periods = -iOffset, freq = None, axis = 0).dropna()
    dVal_X = dX[dX["date"] < np.max(iVal_u)]
    dVal_X = dVal_X.shift(periods = iOffset, freq = None, axis = 0).dropna()
    dVal_Y = dY[dY["date"] < np.max(iVal_u)]
    dVal_Y = dVal_Y.shift(periods = -iOffset, freq = None, axis = 0).dropna()
    dTest_X = dX[dX["date"] < np.max(iTest_u)].dropna() # The test sets are not offset
    dTest_Y = dY[dY["date"] < np.max(iTest_u)].dropna() # The test sets are not offset
    
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

def mrsq(Fhat,lamhat,ve2,series):
    ''' =========================================================================
    DESCRIPTION
    This function computes the R-squared and marginal R-squared from
    estimated factors and factor loadings.
     -------------------------------------------------------------------------
    INPUTS
               Fhat    = estimated factors (one factor per column)
               lamhat  = factor loadings (one factor per column)
               ve2     = eigenvalues of covariance matrix
               series  = series names
     OUTPUTS
               R2      = R-squared for each series for each factor
               mR2     = marginal R-squared for each series for each factor
               mR2_F   = marginal R-squared for each factor
               R2_T    = total variation explained by all factors
               t10_s   = top 10 series that load most heavily on each factor
               t10_mR2 = marginal R-squared corresponding to top 10 series
                         that load most heavily on each factor
    '''

    N, ic = lamhat.shape # N = number of series, ic = number of factors
    Fhat = Fhat.values

    print(N, ic)

    # Preallocate memory for output
    R2 = np.full(shape = (N, ic), fill_value = np.nan)
    mR2 = np.full(shape = (N, ic), fill_value = np.nan)

    # Compute R-squared and marginal R-squared for each series for each factor
    for i in range(ic):
        R2[:, i] = (np.var(Fhat[:, :i+1]@lamhat[:, :i+1].T, axis=0))
        mR2[:, i] = (np.var(Fhat[:, i:i+1]@lamhat[:, i:i+1].T, axis=0))

    # Compute marginal R-squared for each factor
    mR2_F = ve2/np.sum(ve2)
    mR2_F = mR2_F[0:ic]
    
    # Compute total variation explained by all factors
    R2_T = np.sum(mR2_F)

    return R2, mR2, mR2_F, R2_T

# Second: Preallocating memory and defining values

kmax = 60       # Maximum number of principle components estimated.
jj = 2          # The Information criteria used

iTrain_u =  np.arange(19890101, 20040101, 10000)     # End dates for training set, preliminary
iVal_u = np.arange(19980101, 20130101, 10000)        # End dates for validation set, preliminary
iTest_u = np.arange(19990101, 20170101, 10000)       # End dates for test set, preliminary

# Third: the data is loaded and splitted into training, validation and test set

with open('normalized_factor.pkl', 'rb') as fh:
    dX = pickle.load(fh)
with open('excess_returns.pkl', 'rb') as fh:
    dY = pickle.load(fh)

dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates = Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset = 501)
data = pd.DataFrame(dTrain_X)
X = pd.DataFrame(dTrain_X)

# Fourth: the data inputted are asserted

assert kmax <= X.shape[1] and  kmax >= 1 and np.floor(kmax) == kmax or kmax == 99, 'kmax is specified incorrectly'
assert jj in [1, 2, 3], 'jj is specified incorrectly'

# Fifth: The setup for the PCA is made

n = X.shape[1]
t = X.shape[0]/501
nt = n * t
nt1 = n + t

# Sixth: the overfitting penalty is calculated based on the value of jj

ct = np.zeros(kmax)
ii = range(1,1+kmax)
mnt = min(n,t)

if jj == 1:             # Criterion PC_p1
    ct[:] = np.log(nt / nt1) * ii * (nt1 / nt)

elif jj == 2:             # Criterion PC_p2
    ct[:] = np.log(mnt) * ii * (nt1 / nt)

elif jj == 3:             # Criterion PC_p3
    ct[:] = np.log(mnt) / mnt * ii

# Seventh: The PCA is ready to run

if t < n:
    ev, eigval, V = np.linalg.svd(np.dot(X, X.T))       #  Singular value decomposition
    Fhat0 = ev*np.sqrt(t)                               #  Components
    Lambda0 = np.dot(X.T, Fhat0) / t                    #  Loadings
else:
    ev, eigval, V = np.linalg.svd(np.dot(X.T, X))       #  Singular value decomposition
    Lambda0 = ev*np.sqrt(n)                             #  Loadings
    Fhat0 = np.dot(X, Lambda0) / n                      #  Components

# Eigth: The number of principal components is chosen based on the chosen information criteria

sigma = np.zeros(kmax)
ic1 = np.zeros(kmax)

for i in range(0,kmax):
    fhat = Fhat0[:, :i+1]
    lambda_ = Lambda0[:, :i+1]
    
    chat = np.dot(fhat,lambda_.T)
    ehat = X - chat
    sigma[i] = ((ehat*ehat/t).sum(axis=0)).mean() # Sum of squared residuals
    
    ic1[i] = np.log(sigma[i]) + ct[i]

minic1 = minimum_index(ic1)
minic = minic1+1

fhat = pd.DataFrame(Fhat0[:, :kmax])
lambda1 = Lambda0[:, :minic]

chat = np.dot(fhat,lambda1.T)

# Ninth: the values for the chosen information criteria are plotted (customi<e the plot to the chosen IC manually)

plt.plot(ic1, '.')
plt.xlabel('Number of Principal Components, k')
plt.ylabel('IC_P1 (k)')
plt.suptitle('Values of IC_P1 for k')
plt.grid()
plt.show()

# Tenth: The marginal R2 is computed by regressing the ith dataset on each factor and plotted (choose the factor and customize the plot manually)

series = range(1,n+1)

R2, mR2, mR2_F, R2_T = mrsq(Fhat = fhat, lamhat=lambda1, ve2=eigval, series = series)

plt.bar(x = series, height = mR2[:,0])
plt.xlabel('Explanatory factor')
plt.ylabel('R2')
plt.suptitle('Explained Variance of each factor, First Principle Component')
plt.grid()
plt.show()

