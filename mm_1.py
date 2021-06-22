import math

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from Extract import extract
def build():
    dataframes = extract()
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].iloc[:, 1:].T
        #print(dataframes[i])
    df = pd.concat(dataframes)
    '''
    plt.figure()
    plt.plot(df.iloc[:, 15])
    plt.title('Monthly Car Sales', fontsize = 16)
    plt.xlabel('Year', fontsize = 14)
    plt.ylabel('Sales', fontsize = 14)
    plt.savefig('Plot4.jpg')
    #print(df)
    '''
    return df

def adf_fuller_test(X):
    #print(df)
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return None

def autocorr(X):
    #df = build()
    #X = df.iloc[:, -1].values
    adf_fuller_test(X)
    autocorrelation_plot(X)
    plt.title('Autocorrelation')
    plt.show()
    return None

def first_diff():
    df = build()
    X = pd.Series(df.iloc[:, -1].values)
    mt = (X-X.shift(12)).dropna()
    adf_fuller_test(mt)
    #plt.figure()
    #plt.plot(mt)
    #plt.title('First Difference')
    #plt.show()
    autocorr(mt)
    return None

def arima():
    df = build()
    series = pd.Series(df.iloc[:, -1].values)
    model = ARIMA(series, order=(2, 1, 0))
    model_fit = model.fit()
    # summary of fit model
    print(model_fit.summary())
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())
    return None

def arima_forecast():
    df = build()
    X = df.iloc[:, -1].values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(11, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    rmse = math.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes

    plt.plot(test, linestyle='dashed', linewidth=2,label = 'Observed')
    plt.plot(predictions, color='red', linewidth=2, label = 'Predicted')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Predicted v/s Observed')
    plt.show()
    return None



