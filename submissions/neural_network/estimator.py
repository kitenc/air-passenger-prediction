import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def get_estimator(X, Y, LR=1000):
    '''
    use the 1 layer neural network to achieve the linear model
    X : the explanation dataframe with (n_sample, n_feature)
    Y : the respones dataframe with (n_sample, )
    LR : Learning rate 
    '''
    # define the training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    X_train = preprocessing(X_train)
    X_test = preprocessing(X_test)

    # Build the neural network
    model = Sequential()
    model.add(Dense(1, input_shape=(X_train.shape,), activation = 'relu'))
    # Choose loss function and optimizer
    model.compile(optimizer=Adam(learning_rate=LR), loss='sgd')
    
    passenger = model.fit(X_train, y_train, epochs = 500, validation_split = 0.1,verbose = 0)
    passenger_dict = passenger.history
    
    #loss_values = history_dict['loss']
    #val_loss_values=history_dict['val_loss']
    
    # predict the y value
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    
    return mse

def get_nonlinear_estimator(X, Y, LR=1000):
    '''
    use the 1 layer neural network to achieve the linear model
    X : the explanation dataframe with (n_sample, n_feature)
    Y : the respones dataframe with (n_sample, )
    LR : Learning rate 
    '''
    # define the training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    X_train = preprocessing(X_train)
    X_test = preprocessing(X_test)

    # Build the neural network
    model = Sequential()
    model.add(Dense(10, input_shape=(X_train.shape[1],), activation = 'relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1,))
    # Choose loss function and optimizer
    model.compile(optimizer=Adam(learning_rate=LR), loss='sgd')
    
    passenger = model.fit(X_train, y_train, epochs = 500, validation_split = 0.1,verbose = 0)
    passenger_dict = passenger.history
    
    #loss_values = history_dict['loss']
    #val_loss_values=history_dict['val_loss']
    
    # predict the y value
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    
    return mse