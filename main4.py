# -*- coding: utf-8 -*-
"""
Spyder Editor

This code follows the version at:
    https://www.bing.com/videos/search?q=keras+lstm&view=
detail&mid=B76271D08D63FEF8A00DB76271D08D63FEF8A00D&FORM=VIRE
    
This code first creates training and testing data using a rolling window, in which
the first period of data is used to train the model and the second part of the data
is for testing. Every time the window moves forward 1-day time step. 

Modify 1: change the forecast horizon to multiple days rather than just 1 day

Modify 2: 11/29/2017
The current scaler justs transforms the data into the range of [0,1], which
limits the range the stock price can be.
Try to use the standard scaler, which gives 0-mean and 1-variance

Modify 3: 11/30/2017
Add the plotting part to plot the test data and compare it with the real data.
Also plot the forecast on the hold out data!

Modify 4: 12/12/2017
Add the Bayesian Optimization method to tune the parameters for LSTM

Modify 5: 12/9/2019
Optimization the Stock class, make sure the training the testing data are
split first, then build up the rolling window. In this way, we make sure
there is no overapping between the 2 datasets.


@author: Yuxing Tang

"""

import lstm4 as lstm
import stock4 as stk
import portfolio as port

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV

from bayes_opt import BayesianOptimization

def plot_result(predicted_data, true_data, ax):
#    fig = plt.figure(facecolor = 'white')
    ax.plot(true_data, 'b', label = "True data",linewidth=2)
    ax.plot(predicted_data, 'o', label = 'Prediction',linewidth=2)
#    plt.legend()
#    plt.show()
    
    
#make sure we reload the modules.
import importlib
importlib.reload(lstm)
importlib.reload(stk)
importlib.reload(port)
    
# Main Run Thread
if __name__ == '__main__':

    
# =============================================================================
# use 'Stock' module to load and prepare the data
# =============================================================================

#    stock_symbol = 'IPGP'
    
    stock_symbol = 'UNH'
    
    #    how far in the past we retrieve the data
    time_frame = '10y'              
    
    print("Loading data...")

    my_stock = stk.Stock(stock_symbol, time_frame)
    
    my_df = my_stock.prepare_technical()
    
    print("Data loaded !!!")
    

# =============================================================================
#     Build LSTM models
# =============================================================================

#    rolling_window = 20      #  time length of window for fitting the model
#    forecast_horizon = 20    # time length for forecasting
    
    
    
    #    hyper-parameters
#    drop_rate = 0.2
    epochs = 60
#    batch = 64
#    n_units = 128  # neurons in hidden layer

    def lstm_train(batch, 
                   n_units, 
                   rolling_window, 
#                   forecast_horizon,
                   drop_rate):
        
        # the parameters return by Bayes Optimization may not be integers.
        n_units = int(n_units)
        rolling_win = int(rolling_window)
#        forecast_horizon = int(forecast_horizon)
        forecast_horizon = 20
        
        test_size = 0.3  
        
        # split the data according to the sampled rolling window and forecast horizon.
        
        X_train, y_train, X_test, y_test, train_scalers, test_scalers, hold_out = stk.data_split(my_df, rolling_win, forecast_horizon, test_size, stand_norm = True)

        
#        layer = [1, n, forecast_horizon]
        
        # number of features
        n_features = X_train.shape[2]
        
        model = lstm.build_model(rolling_win, 
                                 20, 
                                 drop_rate, 
                                 n_units,
                                 n_features)

        result = model.fit(X_train, y_train, batch_size = int(batch), verbose = 1,
              epochs = epochs, validation_data = (X_test, y_test))
        
        # the objective is the best validation loss in the validation data.
        best_score = -min(result.history['val_loss'])
        
        return best_score
    

    pbds = {'batch': (40,200),
            'n_units': (40,200), 
            'rolling_window': (20,80),
#            'forecast_horizon': (5,30),
            'drop_rate': (0.2,0.5)}


    # bounded regions of the parameter space
    lstm_BayesOptimizer = BayesianOptimization(f = lstm_train,
                                               pbounds = pbds,
                                               verbose = 2,
                                               random_state = 23)
    

#    free parameter kappa which control the balance between exploration and 
#    exploitation; we will set kappa which, in this case, makes the algorithm quite bold.
    lstm_BayesOptimizer.maximize(n_iter=10, kappa = 5)
    
    print('Best combination of parameters:', lstm_BayesOptimizer.max)
    print('The best validation error is {}'.format(-lstm_BayesOptimizer.max['target']))
    
    # all the parameters probed
#    for i, res in enumerate(lstm_BayesOptimizer.res):
#        print("Iteration {}: \n\t{}".format(i, res))
    

    

#Best combination of parameters: {'target': -0.3517188376506514, 'params': {'batch': 200.0, 'drop_rate': 0.2, 'n_units': 200.0, 'rolling_window': 20.0}}
#  
#    
#Best combination of parameters: {'target': -0.3496553568836484, 'params': {'batch': 75.36725812186441, 'drop_rate': 0.4372444170474934, 'n_units': 26.685568124002252, 'rolling_window': 35.697698658601226}}

# =============================================================================
#     set the optimal parameters and re-train the model
# =============================================================================
    
    epochs = 60
    batch = 200
    n_units = 200                   # neurons in hidden layer
    rolling_window = 20
    forecast_horizon = 20
    drop_rate = 0.2
    
    test_size = 0.3
    
    X_train, y_train, X_test, y_test, train_data, train_scalers, test_data, test_scalers,  hold_out = stk.data_split(my_df, rolling_window, forecast_horizon, test_size, stand_norm = True)
    
    n_features = X_train.shape[2]
    
    model = lstm.build_model(rolling_window, 
                             20, 
                             drop_rate, 
                             n_units,
                             n_features)
   
    model.summary()
    
    model.fit(X_train, y_train, batch_size = batch, 
              nb_epoch = epochs, validation_data = (X_test, y_test))
    

    
# =============================================================================
# Save the trained model to a file and later you can load it
# =============================================================================
    
    from keras.models import load_model
    model.save("LSTM_{}_{}_{}.h5".format(stock_symbol, rolling_window, forecast_horizon))  
    # creates a HDF5 file 'my_model.h5'
#    del model  # deletes the existing model
    
    # returns a compiled model
    # identical to the previous one
    model = load_model("LSTM_{}_{}_{}.h5".format(stock_symbol,rolling_window, forecast_horizon))
    
# =============================================================================
#  make forecast on the training data set
# =============================================================================
    
#    predicts = model.predict(X_train)
    
#    invert the predicted and original test value to USD
    
#    invert_predict_train = []
#    invert_true_train = []
#    for i in range(len(predicts)):
#        inverted_pred = lstm.invert_scale_N_feature(scalers[i], X_train[i,:,:], predicts[i])
#        invert_predict_train.append(inverted_pred)
#        inverted_true = lstm.invert_scale_N_feature(scalers[i], X_train[i,:,:], y_train[i])
#        invert_true_train.append(inverted_true)
#
#
#    total_rmse = 0
#    for i in range(len(invert_predict_train)):
#        total_rmse = total_rmse + math.sqrt(
#                mean_squared_error(invert_predict_train[i], invert_true_train[i]))
#    accuracy = total_rmse/len(invert_predict_train)
#    print("The avearge root mse on training data is: %.2f" %(accuracy))
#    
#    for i in range(500, 800, 50):
#        plot_result(invert_predict_train[i], invert_true_train[i] )
    
    
# =============================================================================
#  make forecast on the test data set
# =============================================================================
    
    predicts = model.predict(X_test)
    
#    invert the predicted and original test value to USD

    invert_predict_test = []     # predicted future test values (hold_days)
    invert_true_test = []        # true future test values (hold_days)
    actual_test = []             # actual test data in the previous days (seq_len)
    
    
    for i in range(len(predicts)):
#    for i in range(1):
        actual, inverted_pred = lstm.invert_scale_N_feature(test_scalers[i], 
                                                    test_data[i,:,:], predicts[i])
        invert_predict_test.append(inverted_pred)
        
        actual, inverted_true = lstm.invert_scale_N_feature(test_scalers[i], 
                                                    test_data[i,:,:], y_test[i])
        invert_true_test.append(inverted_true)
        actual_test.append(actual)


    total_rmse = 0
    for i in range(len(invert_predict_test)):
        total_rmse = total_rmse + math.sqrt(
                mean_squared_error(invert_predict_test[i], invert_true_test[i]))
    accuracy = total_rmse/len(invert_predict_test)
    print("The avearge root mse on test data is: %.2f" %(accuracy))
    
    
    window_size = rolling_window + forecast_horizon
    
    
    
    
    # plot some prediction results of the validation data set.
    nrows = 4
    ncols = 3
    
    f, axes = plt.subplots(nrows, ncols, sharex=True, figsize = (10,8))
    f.suptitle("Price predictions for {} in the next 20 days".format(stock_symbol), fontsize=16)
    
    i = 0
    
    for ax in axes.flatten():
        
        ax.plot(invert_true_test[i], 'b', label = "True data",linewidth=2)
        ax.plot(invert_predict_test[i], 'r', label = 'Prediction',linewidth=2)
        
        i += window_size
        
        
    # compare the price change in the next 20 days
    
    gain = []  # the gain/loss is percentage terms for the next 20 days
    
    for p in range(0, len(invert_predict_test), 20):
        prices = invert_predict_test[p]
        gain.append( (prices[-1]/prices[0] - 1 ) * 100)
        
    _ = plt.hist(gain, bins='auto') 
    
    
    
    
    
    
    
    
# =============================================================================
# make use of the prediction in 20 days in the strategy to see the performance
# =============================================================================
    
    trade_threshold =  2.0    # only when my model predicts the changes 
                                # between the prediction in 20 days and 
                                # today's price >= threshold. the trades
                                # are to be made.
                            
    my_wgt = []       # monthly weight using my strategy
    stock_return = []
                            
    for p in range(0, len(invert_predict_test), 20):
        
        pred_prices = invert_predict_test[p]
        true_prices = invert_true_test[p]
        
        pred_chg_pct = (pred_prices[-1]/pred_prices[0] - 1 ) * 100
        
        true_chg_pct = (true_prices[-1]/true_prices[0] - 1 ) * 100
        
        stock_return.append(true_chg_pct)
        
        # if the predicted gain > thres, we take long position
        if pred_chg_pct > trade_threshold: 
            
            my_wgt.append(1.0)
            
        elif pred_chg_pct < -trade_threshold:
            
            my_wgt.append(-1.0)
            
        else:  # no position
            
            my_wgt.append(0.0)
            

    my_wgt = pd.DataFrame(my_wgt, columns = [stock_symbol])
    
    bench_wgt = [1.0 for i in range(len(my_wgt))]
    bench_wgt = pd.DataFrame(bench_wgt, columns = [stock_symbol])
    
    stock_return = pd.DataFrame(stock_return, columns = [stock_symbol])
    
    
    my_port = port.Portfolio(my_wgt, stock_return, stock_symbol, 12)
    IR = my_port.IR(bench_wgt, stock_return)
        
    
# =============================================================================
#  Make plot with actual test data + prediction data
# =============================================================================
        
    window_size = rolling_window + forecast_horizon
    
    total_actual = []
    total_predict = []
#    for i in range(0, X_test.shape[0], window_size ):
    for i in range(0, 200, window_size):
        actual = np.append(actual_test[i], invert_true_test[i])
        predict= np.append(actual_test[i], invert_predict_test[i])
        total_actual.append(actual)
        total_predict.append(predict)
    

    total_actual = np.reshape(total_actual, (window_size*len(total_actual),1))
    total_predict = np.reshape(total_predict, (window_size*len(total_predict),1))
    
    plot_result(total_predict, total_actual)


# =============================================================================
#     make prediction on the new data--hold_out
# =============================================================================
    H = np.array(hold_out)
    scaler = StandardScaler()
    scaler.fit(H)
    norm_hold = scaler.transform(H)
    norm_hold_1 = np.reshape(norm_hold[:, :-1], (1, norm_hold.shape[0], norm_hold.shape[1]-1))
    predict_hold = np.reshape(model.predict(norm_hold_1), (forecast_horizon,))
    acutal_hold, predict_hold = lstm.invert_scale_N_feature(scaler, norm_hold, predict_hold)
    
#    since the normalization process is different from the model building, we 
#    need to adjust the starting price of the forecast: shift the whole curve
#    to make sure the price starts at the last 'Adj Close' in 'my_df'
    shift_size = acutal_hold[-1] - predict_hold[0]
#    shift_size = my_df.iloc[-1]['Adj Close'] - predict_hold[0]
    predict_hold = predict_hold + shift_size

    
    
# =============================================================================
# Try to plot the original data up to today and the forecasted data
# =============================================================================
    
    plt.plot(predict_hold)
    plt.title('Prediction for {} on {}'.format(stock_symbol, 
              str(my_df.index[-1]).split(' ')[0]), fontsize = 18)
    plt.xlabel("Days", fontsize=16)
    plt.ylabel("USD", fontsize=16)
    plt.legend()
    plt.show()
    
    
    from datetime import datetime
    
    df2 = hold_out.copy()
    df2.loc[:, 'Forecast'] = np.nan
    last_date = np.datetime64(df2.iloc[-1].name)
    
    #find the next weekday
    next_date = last_date + np.timedelta64(1,'D')
    while next_date.astype(datetime).isoweekday() not in range(1, 6):
        next_date = next_date + np.timedelta64(1,'D')
    
    
    count = 1
    for i in predict_hold:
        new_date = next_date
        while new_date.astype(datetime).isoweekday() not in range(1, 6):
            new_date = new_date + np.timedelta64(1,'D') 
            
        next_date = new_date + np.timedelta64(1,'D')
        df2.loc[next_date.astype(datetime)] = [np.nan for _ in range(len(df2.columns)-1)] + [i]
        count += 1
    
    df2['Adj Close'].plot(color='blue', linewidth = 2, figsize = (8,6))
    df2['Forecast'].plot(color='red', linewidth = 2, figsize = (8,6))
    plt.legend(loc=4)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('USD', fontsize=16)
    plt.title('Prediction of {} prices for the next {} days'.format(stock_symbol, 
              forecast_horizon), fontsize=16)
    plt.show()  
        
    