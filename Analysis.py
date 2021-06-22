import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import statsmodels as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from Extract import extract

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'

def plot_1():
    df = extract()
    values = df.values
    groups = [i for i in range (0,14) ]
    i = 1
    # plot each column
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(df.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()
    return None

def plot_2():
    df = extract()
    plt.figure(num=None, figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.title('EU-15 Car Data', fontsize=30)
    print(df.columns)

    plt.plot(df.iloc[:, 14])
    plt.show()
    return None

def plot_3():
    df = extract()

    mpl.rcParams['figure.figsize'] = 18, 8
    plt.figure(num=None, figsize=(50, 20), dpi=80, facecolor='w', edgecolor='k')
    series = df.iloc[:,15]
    result = seasonal_decompose(series, model='multiplicative')
    result.plot()
    plt.show()
    return None

def plot_4():
    df = extract()
    fig = plt.figure(figsize=(15, 7))
    layout = (3, 2)
    pm_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    mv_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
    fit_ax = plt.subplot2grid(layout, (2, 0), colspan=2)

    series = df.iloc[:,15]
    result = seasonal_decompose(series, model='multiplicative')
    pm_ax.plot(result.trend)
    pm_ax.set_title("Automatic decomposed trend")

    mm = df.iloc[:,15].rolling(12).mean()
    mv_ax.plot(mm)
    mv_ax.set_title("Moving average 12 steps")

    X = [i for i in range(0, len(df.iloc[:,15]))]
    X = np.reshape(X, (len(X), 1))
    y = df.iloc[:,15].values
    model = LinearRegression()
    model.fit(X, y)
    # calculate trend
    trend = model.predict(X)
    fit_ax.plot(trend)
    fit_ax.set_title("Trend fitted by linear regression")

    plt.tight_layout()
    plt.show()
    return None

def plot_5():
    df = extract()
    fig = plt.figure(figsize=(12, 7))
    layout = (2, 2)
    hist_ax = plt.subplot2grid(layout, (0, 0))
    ac_ax = plt.subplot2grid(layout, (1, 0))
    hist_std_ax = plt.subplot2grid(layout, (0, 1))
    mean_ax = plt.subplot2grid(layout, (1, 1))

    df.iloc[:, 15].hist(ax=hist_ax)
    hist_ax.set_title("Original series histogram")
    series = df.iloc[:,15].values
    plot_acf(series, lags=30, ax=ac_ax)
    ac_ax.set_title("Autocorrelation")

    mm = df.iloc[:,15].rolling(7).std()
    mm.hist(ax=hist_std_ax)
    hist_std_ax.set_title("Standard deviation histogram")

    mm = df.iloc[:, 15].rolling(30).mean()
    mm.plot(ax=mean_ax)
    mean_ax.set_title("Mean over time")
    plt.show()
    return None

def plot_6():
    df = extract()
    rolmean = df.iloc[:,15].rolling(window=12).mean()
    rolstd = df.iloc[:,15].rolling(window=12).std()

    # Plot rolling statistics:
    orig = plt.plot(df.iloc[:,15], label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    return  None

def plot_7():
    df = extract()
    series = df.iloc[:, 15]
    plot_acf(series, lags=30)
    plot_pacf(series, lags=30)
    plt.show()
    
    return None


def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    df = extract()
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=(12, 7))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        mean_std_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        hypothesis_result = "We reject stationarity" if p_value <= 0.05 else "We can not reject stationarity"
        ts_ax.set_title('Time Series stationary analysis Plots\n Dickey-Fuller: p={0:.5f} Result: {1}'.format(p_value,
                                                                                                              hypothesis_result))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

        rolmean = df.iloc[:,15].rolling(window=12).mean()
        rolstd = df.iloc[:,15].rolling(window=12).std()

        # Plot rolling statistics:
        orig = plt.plot(df.iloc[:,15], label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show();
    return None

def plot_8():
    df = extract()
    lag3series = pd.Series(difference(df.iloc[:,15], interval=1, order=2))
    tsplot(lag3series, lags=30)
    return None

def difference(dataset, interval=1, order=1):
    for u in range(order):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        dataset=diff
    return diff

def plot_9():
    df = extract()
    lag1series = pd.Series(difference(df.iloc[:,15], interval=1, order=1))
    lag3series = pd.Series(difference(df.iloc[:,15], interval=3, order=1))
    lag1order2series = pd.Series(difference(df.iloc[:,15], interval=1, order=2))

    fig = plt.figure(figsize=(14, 11))
    layout = (3, 2)
    original = plt.subplot2grid(layout, (0, 0), colspan=2)
    lag1 = plt.subplot2grid(layout, (1, 0))
    lag3 = plt.subplot2grid(layout, (1, 1))
    lag1order2 = plt.subplot2grid(layout, (2, 0), colspan=2)

    original.set_title('Original series')
    original.plot(df.iloc[:,15], label='Original')
    original.plot(df.iloc[:,15].rolling(7).mean(), color='red', label='Rolling Mean')
    original.plot(df.iloc[:,15].rolling(7).std(), color='black', label='Rolling Std')
    original.legend(loc='best')

    lag1.set_title('Difference series with lag 1 order 1')
    lag1.plot(lag1series, label="Lag1")
    lag1.plot(lag1series.rolling(7).mean(), color='red', label='Rolling Mean')
    lag1.plot(lag1series.rolling(7).std(), color='black', label='Rolling Std')
    lag1.legend(loc='best')

    lag3.set_title('Difference series with lag 3 order 1')
    lag3.plot(lag3series, label="Lag3")
    lag3.plot(lag3series.rolling(7).mean(), color='red', label='Rolling Mean')
    lag3.plot(lag3series.rolling(7).std(), color='black', label='Rolling Std')
    lag3.legend(loc='best')

    lag1order2.set_title('Difference series with lag 1 order 2')
    lag1order2.plot(lag1order2series, label="Lag1order2")
    lag1order2.plot(lag1order2series.rolling(7).mean(), color='red', label='Rolling Mean')
    lag1order2.plot(lag1order2series.rolling(7).std(), color='black', label='Rolling Std')
    lag1order2.legend(loc='best')

    plt.show()
    return None
