# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:30:29 2017

This code is used to import the stock price data from online Yahoo Finance.

It provides the function to select how far back you want the data.

"tf": choose how far in the past we want to use (default = 5 years)
You can select the timeframe such as '5Y', '12M', '10D', the code will know it.

Updated: 12/9/2019 modified the data splitting function.

@author: Yuxing Tang
"""

import numpy as np
import pandas as pd
import re
import talib as tb
from sklearn.preprocessing import StandardScaler
#import pandas_datareader as pdr


class Stock:
    def __init__(self, stock_symbol, tf = None):
        
        
        """
        
#        1. Load in the stock data from Yahoo Finance

# if the time frame is not provided, then we assume to use 5 year old data
#        self.stock = stock_symbol
#        self.df = pdr.get_data_yahoo(stock_symbol)
#        
##        plot the 'Adj Close' for viewing
#        self.df['Adj Close'].plot(
#                title="Historical stock prices for {}".format(stock_symbol),
#                figsize=(8,6),
#                fontsize = 16)
#        self.df = self.gen_TimeFrame(tf)
        
        
#       2. load the data from the local downloaded file
        

        """

        self.stock = stock_symbol
        data = pd.read_csv('{}.csv'.format(stock_symbol), header = 0, 
                           index_col =  "Date",
                           parse_dates=True)
        
        if tf == None:
            self.df = data
        else:
            self.df = self.gen_TimeFrame(data, tf)
            
            
        print("The data is from {} to {}".format(self.df.index[0], self.df.index[-1]))
        
        
        ax = self.df ['Adj Close'].plot(title="Historical stock prices for {}".format(stock_symbol), figsize=(8, 6), fontsize = 14, lw = 2)
        ax.set_xlabel("Date", fontsize = 14)
        ax.set_ylabel("USD", fontsize = 14)
        
        return



    def gen_TimeFrame(self, data, tf):
        
        """
        tf is the recent timeframe we look back to filter the input data, 
        the function returns the filtered version, per user's choice of the past time frame
        
        """
        tf_num = int(re.search(r'\d+', tf).group())   # how many
        tf_type = str.capitalize(tf[-1])  #  select 'Y', 'M', 'D'
        df2 = data[data.index > (data.index[-1] - np.timedelta64(1,tf_type)*tf_num)]
#        print(df2.head())
#        print(df2.tail())
        
        return df2
    

    
    def prepare_technical(self):
        
        """
# =============================================================================
#  Prepare the features we use for the model
# =============================================================================
#1. HL_PCT: the variation of the stock price in a single day
#2. PCT_change: the variation between the open price and the close price
#3. Adj close price of the day


        
        """
        df3 = self.df.copy()
        # transform the data and get %change daily       
        df3['HL_PCT'] = (df3['High'] - df3['Low']) / df3['Low'] * 100.0
        # spread/volatility from day to day
        df3['PCT_change'] = (df3['Adj Close'] - df3['Open']) / df3['Open'] * 100.0
        
#obtain the data from the technical analysis function and process it into useful features
#        open = df3['Open'].values
        close = df3['Adj Close'].values
        high = df3['High'].values
        low = df3['Low'].values
        volume = df3['Volume'].values
        
#The technical indicators below cover different types of features:
#1) Price change – ROCR, MOM
#2) Stock trend discovery – ADX, MFI
#3) Buy&Sell signals – WILLR, RSI, CCI, MACD
#4) Volatility signal – ATR
#5) Volume weights – OBV
#6) Noise elimination and data smoothing – TRIX
        
# define the technical analysis matrix
        
#        https://www.fmlabs.com/reference/default.htm?url=ExpMA.htm

#  Overlap Studies
        
        # make sure there is NO forward looking bias.
        #moving average
        df3['MA_5']  = tb.MA(close, timeperiod=5) 
        df3['MA_20'] = tb.MA(close, timeperiod=20)
        df3['MA_60'] = tb.MA(close, timeperiod=60)
#        df3['MA_120'] = tb.MA(close, timeperiod=120)
        
        # exponential moving average
        df3['EMA_5'] = tb.MA(close, timeperiod=15)  
        # 5-day halflife. the timeperiod in the function is the "span".

        df3['up_band'] = tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
        df3['mid_band'] = tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
        df3['low_band'] = tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
            
# Momentum Indicators
        df3['ADX'] = tb.ADX(high, low, close, timeperiod=20)            
#        df3['ADXR'] = tb.ADXR(high, low, close, timeperiod=20)                         
        df3['MACD'] = tb.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2]  
        df3['RSI'] = tb.RSI(close, timeperiod=14)
#       df3['AD'] = tb.AD(high, low, close, volume)
        df3['ATR'] = tb.ATR(high, low, close, timeperiod=14)
        df3['MOM'] = tb.MOM(close, timeperiod=10)
        df3['WILLR'] = tb.WILLR(high, low, close, timeperiod=10)
        df3['CCI'] = tb.CCI(high, low, close, timeperiod=14)
            
#   Volume Indicators
        df3['OBV'] = tb.OBV(close, volume*1.0)

#   drop the NAN rows in the dataframe
#        df3.dropna(axis = 0, inplace = True)
#        df3.fillna(value=-99999, inplace=True)
        
        df3 = df3[['Adj Close',
                   'HL_PCT',
                   'PCT_change',
                   'Volume',
                   'MA_5',
                   'MA_20',
                   'MA_60',
#                   'MA_120',
                   'EMA_5',
                   'up_band',
                   'mid_band',
                   'low_band',
                   'ADX',
                   'MACD',
                   'RSI',
                   'ATR',
                   'MOM',
                   'WILLR',
                   'CCI',
                   'OBV'
                   ]]
        
#        forecast_col = 'Adj Close' 
#        df3.loc[:,'label'] = df3[forecast_col].shift(-forecast_out)
        
        return df3
        


    def export_data(self):      
        """
        export the data to csv file
        
        """
        
        self.df_out = self.prepare_technical()
        self.df_out.to_csv('Data_for_{}.csv'.format(self.stock), sep=',')
        
        
###############################################################################


def normalise(window_data):
    """
# keep track of the scaler, need to use them to scale back the numbers 
# in the main code
    """
    normalised_data = []
    scalers = []
    for window in window_data:
        scaler = StandardScaler()
#        scaler = MinMaxScaler()
        scaler.fit(window)
        norm_wind = scaler.transform(window)
        normalised_data.append(norm_wind)
        scalers.append(scaler)
    return normalised_data, scalers


    

# maybe add some log transformation to the data befroe standardization???


def data_split(input_data, 
               rolling_window,    #  time length of window for fitting the model
               forecast_horizon,  # time length for forecasting
               test_size = 0.3, 
               stand_norm = True):
    
    """
#    This function implements Rolling-Window data splitting:
    
    1. save the recent "forecast_horizon" data for making prediction for the future
    stock prices, which we can use for trading.
    2. split the data into 2 parts by a certain time point in history.
    All the data after the time point is hold-out data, and must NOT be used 
    in the model training process.
    
    3. The first part of the data is to be used for training the model.
    We need a rolling window of (40 days) to predict the next (20 days) of 
    stock prices. So we walk forward 1 day each time, make a new data sample.
    
    4. scale each feature within each data; keep track of all the scalers used
    
    5. return all the splitted data and the scalers used; the scalers will be used
    to back out the real stock prices.
    
    
    Reference:
        
    https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
    """

    assert isinstance(input_data, pd.DataFrame), "THe input data is not dataframe!!!"
        
    input_data2 = input_data.dropna(axis = 0).copy()
    
    df_for_predict = input_data2.iloc[-rolling_window:]   # for future forecasting use
    df = input_data2.iloc[:-rolling_window]          # for model training and testing use
    
    
    # split the data into train and test data.
    row = round((1-test_size) * df.shape[0])
    
    df_train = df.iloc[:row, :]
    df_test = df.iloc[row:, :]
    
    print("Training data starts at", df_train.index[0])
    print("The data is split at", df_train.index[-1])
    print("Testing data ends at", df_test.index[-1])
    
    
    
#    use the 'rolling_window' long data to forecast the next "forecast_horizon" data
    sequence_length = rolling_window + forecast_horizon
    
    # collect the splitted training and testing data.
    
    def rolling_dataframe(dat, sequence_length,  
                          step = 1,   # each time move forward 1 day
                          stand_norm = stand_norm):
        
        transformed_data = []
        size = dat.shape[0]
        
        
        #form a sliding window of size 'sequence_length', until the data is exhausted
        for index in range(0, size - sequence_length + 1, step):
            transformed_data.append(dat[index: index + sequence_length])
        
    #    standard normalise each feature in each time window, including the train and forecast
        if stand_norm:
            transformed_data, scalers = normalise(transformed_data)
            
            return transformed_data, scalers
        
        else:
            return transformed_data
    
    if stand_norm:
        train_data, train_scalers = rolling_dataframe(df_train, sequence_length, 
                                                      step = 1,   # each time move forward 1 day
                                                      stand_norm = stand_norm)
    
        test_data, test_scalers = rolling_dataframe(df_test, sequence_length, 
                                                      step = 1,   # each time move forward 1 day
                                                      stand_norm = stand_norm)
        
    else:
        train_data = rolling_dataframe(df_train, sequence_length, 
                                      step = 1,   # each time move forward 1 day
                                      stand_norm = stand_norm)
    
        test_data = rolling_dataframe(df_test, sequence_length, 
                                      step = 1,   # each time move forward 1 day
                                      stand_norm = stand_norm)

    
    

#    forecast_col = 'Adj Close' is always in the 1st column
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    X_train = train_data[:, :-forecast_horizon, 1:]  # should not have the 1st column
    y_train = train_data[:, -forecast_horizon: , 0]   
    X_test  =  test_data[:, :-forecast_horizon, 1:]  # should not have the 1st column
    y_test  =  test_data[:, -forecast_horizon: ,0]
    
    #    reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(df.columns) - 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],len(df.columns) - 1))
    
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
    
    print("Train data shape: %r, Train target shape: %r" 
          % (X_train.shape, y_train.shape))
    print("Test data shape: %r, Test target shape: %r" 
          % (X_test.shape, y_test.shape))
    
    if stand_norm:
        
        # train_data and test_data are just for rescaling the stock prices.
        
        return [X_train, y_train, X_test, y_test, train_data, train_scalers, 
                test_data, test_scalers, df_for_predict]
    
    else:
        return [X_train, y_train, X_test, y_test, df_for_predict]







# this part below only operates when we do unit test of this module. When others
#        import this script, those codes below won't be executed.

if __name__ == '__main__':
    
    my_stock = Stock("IPGP")

    my_df2 = my_stock.prepare_technical()

    X_train, y_train, X_test, y_test, train_data, train_scalers, test_data, test_scalers, df_for_predict = data_split(my_df2, 
           rolling_window = 40,    #  time length of window for fitting the model
           forecast_horizon = 20,  # time length for forecasting
           test_size = 0.3, 
           stand_norm = True)