import FinanceDataReader as fdr
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()

from datetime import datetime

import model as model
import ProcessDataset as prd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras

window_size = 4
number_of_features = 1

start_date = datetime(2024, 1, 1)
end_date = datetime.today()

SamsungStockData = web.get_data_yahoo('005930.KS', start = start_date, end = end_date)
# MNQData = fdr.DataReader('MNQ=F', start_date, end_date)

SamsungStockData = SamsungStockData.reindex(['Open', 'High', 'Low', 'Volume', 'Close', 'Adj Close'], axis = 1)
max_value = prd.get_max_temp(SamsungStockData['Adj Close'])
min_value = prd.get_min_temp(SamsungStockData['Adj Close'])
SamsungStockData['Adj Close'] = prd.scale_data(SamsungStockData['Adj Close'])

stock_data = SamsungStockData['Adj Close'].to_numpy()
_, _, current = prd.train_test_split(stock_data, train_ratio = 0.6, valid_ratio = 0.75)

plot_x, _ = prd.create_dataset(current, shuffle = False, window_size = window_size, number_of_features = number_of_features)

LSTM_model = keras.models.load_model('weights2.h5')

predicted = np.array([np.nan]*(window_size-1))
for i in range(plot_x.shape[0]):
    result = LSTM_model.predict(plot_x[i].reshape(1,window_size,number_of_features))
    predicted = np.append(predicted, result)


def predicted_value():
    return predicted[-1]

def get_current_data():
    result = plot_x[:,0]
    return result

def get_result_string():
    if plot_x[:,0][-1] > predicted[-1]:
        return "Sell"
    else:
        return "Buy"
    

def plot_picture():
    plt.figure()
    plt.plot(np.arange(len(plot_x[:,0])), plot_x[:,0])
    plt.plot(predicted)
    plt.title('Recent Stock Data')
    plt.xlabel('Time')
    plt.ylabel('Adjusted Close')
    plt.savefig('static/stock_data.png')
    plt.close()
    return 'static/stock_data.png'
