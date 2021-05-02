#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:06:40 2021

@author: dyveke
"""

import pickle
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model
from keras import optimizers
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import regularizers, initializers
from keras.layers.normalization import BatchNormalization
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from Data_Split import Test_Val_Train, Geom_nlayer

# The model used for the testing:
def NNLoop(dTrain_X, dTrain_Y, dVal_X, dVal_Y, nlayer, basis_layer, L_activation, L_bias, B_activation, epo, dr, lr, pa, a):
    neurons = np.array(Geom_nlayer(nlayer, basis_layer))
    NNLoop = Sequential()
    for i in range(nlayer):
        if L_activation == 'relu':
            NNLoop.add(Dense(units = neurons[i], kernel_initializer = initializers.he_normal(), use_bias = L_bias))
            NNLoop.add(LeakyReLU(alpha = a))
        else:
            NNLoop.add(Dense(units = neurons[i], activation = L_activation, kernel_initializer = initializers.he_normal(), use_bias = L_bias))
        NNLoop.add(BatchNormalization())
        NNLoop.add(Activation(B_activation))
        NNLoop.add(Dropout(dr))
    NNLoop.add(Dense(1, activation = 'linear'))
    opt = optimizers.Adam(learning_rate = lr, beta_1=0.9, beta_2 = 0.999, decay = 0.0, amsgrad = True)
    NNLoop.compile(optimizer = opt, loss = keras.losses.MeanSquaredError(), metrics = ['mse'])
    es = EarlyStopping(monitor='val_loss', mode='auto', patience=pa, restore_best_weights=True)
    mc = ModelCheckpoint(filepath = 'Best_FFN.h5', monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)
    historyLoop = NNLoop.fit(x = dTrain_X, y = dTrain_Y, batch_size = 10000, epochs = epo, callbacks = [es, mc], shuffle = False, validation_data = (dVal_X, dVal_Y))
    Best_NNLoop = load_model('Best_FFN.h5')
    Best_NNLoop.summary()
    return Best_NNLoop, historyLoop

# Setting the random seed for reproducibility.

seed = 17
np.random.seed(seed)

# Importing the data.

with open('normalized_factor.pkl', 'rb') as fh:
    dX = pickle.load(fh)
with open('excess_returns.pkl', 'rb') as fh:
    dY = pickle.load(fh)

# Setting the cut of dates for the samples (the latest date in the interval is chosen)

iTrain_u =  np.arange(19890101, 20040101, 10000) # End dates for training set, preliminary
iVal_u = np.arange(19980101, 20130101, 10000) # End dates for validation set, preliminary
iTest_u = np.arange(19990101, 20170101, 10000) # End dates for test set, preliminary

# Dates used for robustness checks
#iTrain_u =  np.arange(19890101, 20000101, 10000) # End dates for training set, preliminary
#iVal_u = np.arange(19980101, 20090101, 10000) # End dates for validation set, preliminary
#iTest_u = np.arange(19990101, 20170101, 10000) # End dates for test set, preliminary

dTrain_X, dTrain_Y, dVal_X, dVal_Y, dTest_X, dTest_Y, dfDates = Test_Val_Train(dX, dY, iTrain_u, iVal_u, iTest_u, iOffset = 501)

# The possible values for the different hyperparameters.

bias = [True, False]
dr  = np.linspace(0.05, 0.55, 11)
epo_loop = [100, 200, 250, 500, 750]
alpha_activation = np.linspace(0.0, 1, 11)
patience = [5,10,25,50,100,500]
layer = np.linspace(2,7,6)
basis = [2,4,8,16]
lr = [0.01, 0.05, 0.1, 0.25, 0.5]

mse = np.zeros(shape=(8,11))

# Bias:

for i in range(2):
    print (i, "bias or not")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = 4, basis_layer = 8, L_activation = 'relu', L_bias = bias[i], 
                             B_activation = 'linear', epo = 100, dr = 0.2, lr = 0.01, pa = 100, a = 0.0)
    mse[0,i] = (history.history['val_mse'][-1])

best_bias = np.unravel_index(np.argmin(mse[0,0:2]), mse.shape)
# In the first iteration, the best value is 1. (Thus using bias in the model)
# In teh Robust Checks, the bias was also chosen to be implemented in each hidden layer.

# Dropout rate:

for i in range(11):
    print (i, "dropout")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = 4, basis_layer = 8, L_activation = 'relu', L_bias = bias[best_bias[1]], 
                             B_activation = 'linear', epo = 100, dr = dr[i], lr = 0.01, pa = 100, a = 0.0)
    mse[1,i] = (history.history['val_mse'][-1])

best_dr = np.unravel_index(np.argmin(mse[1,0:12]), mse.shape)
# In the first iteration, the best value was 4, thus having a dropout rate of 0.25.
# In the robustness checks, the Dropout rate was chosen to be 0.1


# epo:

for i in range(5):
    print (i, "epo")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = 4, basis_layer = 8, L_activation = 'relu', L_bias = bias[best_bias[1]], 
                             B_activation = 'linear', epo = epo_loop[i], dr = dr[best_dr[1]], lr = 0.01, pa = 100, a = 0.0)
    mse[2,i] = (history.history['val_mse'][-1])

best_epo = np.unravel_index(np.argmin(mse[2,0:5]), mse.shape)
# in the first iteration, the best value was 1000, thus having an epoch of 750.
# In the robustness checks, the apochs was chosen to be 250.

# Alpha:

for i in range(11):
    print (i, "alpha")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = 4, basis_layer = 8, L_activation = 'relu', L_bias = bias[best_bias[1]], 
                             B_activation = 'linear', epo = epo_loop[best_epo[1]], dr = dr[best_dr[1]], lr = 0.01, 
                             pa = 100, a = alpha_activation[i])
    mse[3,i] = (history.history['val_mse'][-1])
    
best_alpha = np.unravel_index(np.argmin(mse[3,0:11]), mse.shape)
# in the first iteration, the best value is 3, therefore meaning that the alpha value should be 0.3

# Learning rate:

for i in range(5):
    print (i, "Learning Rate")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = 4, basis_layer = 8, L_activation = 'relu', L_bias = bias[best_bias[1]], 
                             B_activation = 'linear', epo = epo_loop[best_epo[1]], dr = dr[best_dr[1]], lr = lr[i], 
                             pa = 100, a = alpha_activation[best_alpha[1]])
    mse[4,i] = (history.history['val_mse'][-1])

best_lr = np.unravel_index(np.argmin(mse[4,0:5]), mse.shape)

# Patience:

for i in range(6):
    print (i, "patience")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = 4, basis_layer = 8, L_activation = 'relu', L_bias = bias[best_bias[1]], 
                             B_activation = 'linear', epo = epo_loop[best_epo[1]], dr = dr[best_dr[1]], lr = lr[best_lr[1]], 
                             pa = patience[i], a = alpha_activation[best_alpha[1]])
    mse[5,i] = (history.history['val_mse'][-1])

best_patience = np.unravel_index(np.argmin(mse[5,0:6]), mse.shape)
# In the first iteration, the best value for patience was: 10

# Layer:

for i in range(6):
    print (i, "layer")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = int(layer[i]), basis_layer = 8, L_activation = 'relu', L_bias = bias[best_bias[1]], 
                             B_activation = 'linear', epo = epo_loop[best_epo[1]], dr = dr[best_dr[1]], lr = lr[best_lr[1]], 
                             pa = patience[best_patience[1]], a = alpha_activation[best_alpha[1]])
    mse[6,i] = (history.history['val_mse'][-1])

best_layer = np.unravel_index(np.argmin(mse[6,0:6]), mse.shape)
# in the first iteration, the best value for layer was: 7, thus meaning a value of i of 5.

# Basis:

for i in range(4):
    print (i, "basis")
    model, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = int(layer[best_layer[1]]), basis_layer = basis[i], L_activation = 'relu', L_bias = bias[best_bias[1]], 
                             B_activation = 'linear', epo = epo_loop[best_epo[1]], dr = dr[best_dr[1]], lr = lr[best_lr[1]], 
                             pa = patience[best_patience[1]], a = alpha_activation[best_alpha[1]])
    mse[7,i] = (history.history['val_mse'][-1])

best_basis = np.unravel_index(np.argmin(mse[7,0:4]), mse.shape)
# in the first iteration the best basis layer is 16, but 8 is a close contender. (thus a value of 3)


FinModelFFN, history = NNLoop(dTrain_X = dTrain_X, dTrain_Y = dTrain_Y, dVal_X = dVal_X, dVal_Y = dVal_Y, 
                             nlayer = int(layer[2]), basis_layer = basis[1], L_activation = 'relu', L_bias = bias[0], 
                             B_activation = 'linear', epo = epo_loop[3], dr = dr[2], lr = lr[1], 
                             pa = patience[2], a = alpha_activation[1])

FinModelFFN = load_model('Best_FFN.h5')


PTest_Y_FFN = FinModelFFN.predict(dTest_X)
PTest_Y_FFN = pd.DataFrame(PTest_Y_FFN)

with open('PTest_Y_FFN.csv','w') as tfl:
    tfl.write(PTest_Y_FFN.to_csv())


errorT = dTest_Y-PTest_Y_FFN

MSE_T = mean_squared_error(dTest_Y, PTest_Y_FFN)

Asset200 = range(200,24248,501)

plt.plot(dTest_Y[Asset200])
plt.plot(PTest_Y_FFN[Asset200])
plt.show()

series = range(24048)
plt.plot(series, errorT, '.')

PTrain_Y = FinModelFFN.predict(dTrain_X)
error_Train = dTrain_Y - PTrain_Y
plt.plot(error_Train, 'r.')
plt.plot(errorT, 'b.')
plt.show()
