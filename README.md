# Stock-price-prediction

# Project description
This project is to employ encoder-decoder seq2seq model with LSTM units in stock price prediction. From the findings in the behavior finance, investors have this "herding" behavior that they tend to look at what other people do and do the same things. The idea of this language translation model is that the input sentence is embedded into a fixed-length vector, incorporating the information in the sentence, then the information in the vector is transformed into another language. Similar thought can be applied to the above herding behavior modeling. All the information in the past few days about the market are extracted into a vector, then it provides with the hints of next moves in the stock market. 


## Table of contents
* [Feature engineering](#Feature-engineering)
* [Data splitting](#Data-splitting)
* [Hyperparameters](#Hyperparameters)



# Feature engineering

The most available information we get freely is the stock prices each day and the trading volume. Experts in Wall Street have developed lots of technical indicators to help them make calls of when to buy or when to sell the stocks. We can make use of these technical indicators as the feature input for the models, such as MACD, RSI, moving averages, etc.


# Data splitting
The way we make predictions is to stand at the current time point, look back for some periods, gather the information about the current market’s situation, then make prediction for the next certain period. Here we want to make the strategy a monthly trading strategy, so we fix the forecast period to be 20 days, instead of making it as a hyperparameter to tune in the model.

The total length of the time window = rolling window + forecast horizon (20 days).

We use the information in the “rolling window” to forecast in the “forecast horizon”.
(1)	We first save the last “rolling window” days of data, to make the prediction for our next trading in the future. 
(2)	Based on the testing data size (for example 0.3 here), we split the rest of the data into 2 big chunks, the training data and the testing data. The testing data is the hold-out data, which we presumably cannot look at before we are certain about the model we want to use.

Inside each chunk, we use the “time window” to select out the data, making it as one sample, then moving one day ahead, and get another sample. In the case for stock “IPGP”, we formulate 2159 training samples and 892 testing samples.

(3)	Inside each “time window”, we scale each column to have mean 0 and variance 1 and save the scalers. After we make predictions under this new scale, we use the saved scalers to rescale back the numbers into the USD units.



# Hyperparameters
(1)	Rolling window: how many days’ of data we want to use to make prediction for the next 20 days
(2)	Drop_rate: the drop rate in the drop layer in the model, which can help reduce over-fitting.
(3)	N_units: the length of the vector we incorporate the input stock price information into.
(4)	Batch size


# Model assessment
After training and selecting the best model, we want to see how it actually helps us to make trading decisions. 
(1)	First we can compare the predictions we make and the true data in the units of USD.
The blue lines are the true data and the red lines are the predictions. We can see most of them capture the correct trends.


(2)	We can form a trading strategy using these predictions of stock prices. A simple idea is to look at the predictions, if the model says the price will increase after 20 days, we buy now; otherwise, we can short the stock now. If we do this, and use long-only as the benchmark, we can achieve an information ratio of 1.28, which is a very good strategy. 


![alt text](https://github.com/lattedog/Stock-price-prediction/blob/master/comparison%20of%20my%20strategy%20on%2012-10-2019.png)

