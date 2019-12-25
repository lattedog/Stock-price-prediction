# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:58:24 2017

@author: Yuxing Tang
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Activation, Dropout, RepeatVector, TimeDistributed, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential


warnings.filterwarnings("ignore")


# scale train and test data to [0, 1]
# keep track of the scaler, need to use them to scale back the numbers 
# in the main code
#def normalise(window_data):
#    normalised_data = []
#    scalers = []
#    for window in window_data:
#        scaler = MinMaxScaler()
#        scaler.fit(window)
#        norm_wind = scaler.transform(window)
#        normalised_data.append(norm_wind)
#        scalers.append(scaler)
#    return normalised_data, scalers

def normalise(window_data):
    normalised_data = []
    scalers = []
    for window in window_data:
        scaler = StandardScaler()
        scaler.fit(window)
        norm_wind = scaler.transform(window)
        normalised_data.append(norm_wind)
        scalers.append(scaler)
    return normalised_data, scalers


def invert_scale_1_feature(scaler, yhat):
    new_row = [y for y in yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return np.squeeze(inverted)


def invert_scale_N_feature(scaler, X, yhat):
#   the 1st column is the 'Adj Close' price column
    result = np.zeros([X.shape[0]+yhat.shape[0], X.shape[1]])
    result[:X.shape[0], :] = X
    result[-yhat.shape[0]:, 0] = yhat
    
    inverted = scaler.inverse_transform(result)
    
    return inverted[:-yhat.shape[0], 0], inverted[-yhat.shape[0]:, 0]


def load_data(filename, seq_len, hold_out, normalise_window = True):
    
#filename= 'SPY.csv'
#seq_len = 50
#hold_out = 30
#normalise_window = True
    

    df = pd.read_csv(filename, header= 0)
    df = df[['Adj Close']]  
      
    df_lately = df.iloc[-seq_len:]   # for forecasting use
    df = df.iloc[:-seq_len]          # for model fitting and testing use
    
    
    data = []
    size = df.shape[0]
    
#    use the 'seq_len' long data to forecast the next hold_out step data
#    sequence_length = seq_len + 1
    sequence_length = seq_len + hold_out


    #form a sliding window of size 'seq_len', until the data is exhausted
    for index in range(size - sequence_length + 1):
        data.append(df[index: index + sequence_length])
    
    if normalise_window:
        data, scaler = normalise(data)
        
    data = np.array(data) 
    
    
    test_size = 0.2
    row = round((1-test_size) * data.shape[0])
    train = data[:row, :]
    
    np.random.shuffle(train)
    X_train = train[:, :-hold_out]
    y_train = train[:, -hold_out:]   
    X_test = data[row:, :-hold_out]
    y_test = data[row:, -hold_out:]
    
#    reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
    
    print("Train data shape: %r, Train target shape: %r" 
          % (X_train.shape, y_train.shape))
    print("Test data shape: %r, Test target shape: %r" 
          % (X_test.shape, y_test.shape))
    
    return [X_train, y_train, X_test, y_test, data, scaler, df_lately]


def build_model(input_n, output_n, drop_rate, units_n, feature_n):
    
    """
    input_n: the length of the input sequence
    output_n: the length of the predicted sequence
    feature_n: how many features we have in the model

    """
    
    print("input_n", input_n)
    print("output_n", output_n)
    print("units_n", units_n)
    print("feature_n", feature_n)
    
#    model = Sequential()
    
#    model.add(LSTM(batch_input_shape = (409, input_n, feature_n), 
#    output_dim = layers[1], return_sequences = True, stateful = True))
    
#    model.add(LSTM(input_shape = (input_n, feature_n), 
#                   output_dim = layers[1], return_sequences = True,
#                   activation = 'relu'
#                   ))
#    model.add(Dropout(drop_rate))
    

#    model.add(LSTM(input_shape = (input_n, layers[1]), 
#                   output_dim = layers[2], return_sequences = True 
##                   activation = 'sigmoid'
#                   ))
#    model.add(Dropout(0.2))
    
#    model.add(LSTM(layers[1], return_sequences = True, activation = 'relu'))
#    
#    model.add(LSTM(input_shape = (input_n, layers[1]), 
#                   output_dim = layers[2], return_sequences = False,
#                   activation = 'relu'
#                   ))
#    model.add(Dropout(drop_rate))
    
#    model.add(Dense(output_n))   
#    model.add(Activation("linear"))
    
    
# =============================================================================
#     Encoder-decoder structure
# =============================================================================
    
    model = Sequential()
    
#    The encoder is traditionally a Vanilla LSTM model
    model.add(LSTM(units_n, input_shape = (input_n, feature_n), activation='relu'))
    
#    the fixed-length output of the encoder is repeated, once for each required time step in the output sequence.
    model.add(RepeatVector(output_n))
    
    model.add(LSTM(units_n, activation='relu', return_sequences = True))
    
#    model.add(Flatten())
#    
    model.add(TimeDistributed(Dense(1)))
    model.add(Dropout(drop_rate))
    model.add(Flatten())
    
    
#    start = time.time()
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', matrics =["MAPE"])
    
#    print("Compilation Time:" + str(time.time()-start) )
    print(model.summary())

    return model


