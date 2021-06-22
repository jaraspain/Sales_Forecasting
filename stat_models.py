import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import pmdarima as pm
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model, svm
from sklearn.linear_model import LinearRegression

from Extract import extract
def train_test_split():
    df = extract()
    df.columns = ['Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Ireland', 'Italy', 'Luxembourg',
                  'Netherlands', 'Portugal', 'Spain', 'Sweden', 'UK', 'EU-15']
    n = len(df)
    train_df = df[0:int(n * 0.5)]
    test_df = df[int(n * 0.2):]
    return train_df, test_df

def hwes():
    df_training, df_test = train_test_split()
    index = len(df_training)
    df = extract()
    yhat = list()
    for t in tqdm(range(len(df_test.iloc[:,15]))):
        temp_train = df[:len(df_training) + t]
        model = ExponentialSmoothing(temp_train.iloc[:,15])
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    err1= mean_absolute_error(df_test.iloc[:,15], yhat.values)
    val_2 = yhat.values

    print(err1)
    plt.figure()
    plt.plot(df_test.iloc[:,15].values, label = 'Original')
    plt.plot(val_2,color = 'red', label = 'Predicted')
    plt.legend()
    plt.title('HWES Exponential Smoothing')
    plt.show()
    return None

def ar():
    df_training, df_test = train_test_split()
    index = len(df_training)
    df = extract()
    yhat = list()
    for t in tqdm(range(len(df_test.iloc[:,15]))):
        temp_train = df[:len(df_training) + t]
        model = AR(temp_train.iloc[:,15])
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    err1 = mean_absolute_error(df_test.iloc[:, 15], yhat.values)
    val_2 = yhat.values

    print(err1)
    plt.figure()
    plt.plot(df_test.iloc[:, 15].values, label='Original')
    plt.plot(val_2, color='red', label='Predicted')
    plt.legend()
    plt.title('AR')
    plt.show()
    return None

def arma():
    df_training, df_test = train_test_split()
    index = len(df_training)
    df = extract()
    yhat = list()
    for t in tqdm(range(len(df_test.iloc[:, 15]))):
        temp_train = df[:len(df_training) + t]
        model = ARIMA(temp_train.iloc[:,15], order=(1, 0,0))
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    err1 = mean_absolute_error(df_test.iloc[:, 15], yhat.values)
    val_2 = yhat.values

    print(err1)
    plt.figure()
    plt.plot(df_test.iloc[:, 15].values, label='Original')
    plt.plot(val_2, color='red', label='Predicted')
    plt.legend()
    plt.title('ARIMA')
    plt.show()
    return None

def auto_arima():
    df_training, df_test = train_test_split()
    autoModel = pm.auto_arima(df_training.iloc[:,15], trace=True, error_action='ignore', suppress_warnings=True, seasonal=True, m=6, stepwise=True)
    autoModel.fit(df_training.iloc[:,15])
    return autoModel

def auto():
    autoModel = auto_arima()
    df_training, df_test = train_test_split()
    df = extract()
    order = autoModel.order
    seasonalOrder = autoModel.seasonal_order
    yhat = list()
    for t in tqdm(range(len(df_test.iloc[:,15]))):
        temp_train = df[:len(df_training) + t]
        model = SARIMAX(temp_train.iloc[:,15], order=order, seasonal_order=seasonalOrder)
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    yhat = pd.concat(yhat)
    err1 = mean_absolute_error(df_test.iloc[:, 15], yhat.values)
    val_2 = yhat.values

    print(err1)
    plt.figure()
    plt.plot(df_test.iloc[:, 15].values, label='Original')
    plt.plot(val_2, color='red', label='Predicted AutoSARIMAX {}'.format(order))
    plt.legend()
    plt.title('AUTO-SARIMAX')
    plt.show()

    return None


def create_features(df, target = None):
    df['date'] = df.index
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    X = df.drop(['date'], axis=1)
    if target:
        y = df[target]
        X = X.drop([target], axis=1)
        return X, y

    return X

def train_test_mult():
    df_training, df_test = train_test_split()
    X_train_df, y_train = create_features(df_training, target='EU-15')
    X_test_df, y_test = create_features(df_test, target='EU-15')
    scaler = StandardScaler()
    scaler.fit(X_train_df)  # No cheating, never scale on the training+test!
    X_train = scaler.transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns)
    X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns)

    print(X_train_df.head())
    print(X_test_df.head())
    return X_train_df, y_train, X_test_df, y_test

def bay_reg():
    X_train, y_train, X_test,y_test = train_test_mult()
    df_training, df_test = train_test_split()

    reg= xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=1000)
    lightGBM = lgb.LGBMRegressor()
    reg.fit(X_train, y_train)
    lightGBM.fit(X_train, y_train)
    yhat = reg.predict(X_test)
    yhat_2 = lightGBM.predict(X_test)
    yhat = (yhat+yhat_2)/2
    err1 = mean_absolute_error(df_test.iloc[:, 15], yhat)
    val_2 = yhat

    print(err1)
    plt.figure()
    plt.plot(df_test.iloc[:, 15].values, label='Original')
    plt.plot(val_2, color='red', label='Predicted')
    plt.legend()
    plt.title('XGBoost + LightGBM Ensemble')
    plt.show()

    return None


def window_data(X, Y, window=7):
    '''
    The dataset length will be reduced to guarante all samples have the window, so new length will be len(dataset)-window
    '''
    x = []
    y = []
    for i in range(window - 1, len(X)):
        x.append(X[i - window + 1:i + 1])
        y.append(Y[i])
    return np.array(x), np.array(y)

def window_gen():
    BATCH_SIZE = 64
    BUFFER_SIZE = 10
    WINDOW_LENGTH = 12
    X_train, y_train, X_test, y_test = train_test_mult()
    X_w = np.concatenate((X_train, X_test))
    y_w = np.concatenate((y_train, y_test))

    X_w, y_w = window_data(X_w, y_w, window=WINDOW_LENGTH)
    X_train_w = X_w[:-len(X_test)]
    y_train_w = y_w[:-len(X_test)]
    X_test_w = X_w[-len(X_test):]
    y_test_w = y_w[-len(X_test):]

    print(f"Test set equal: {np.array_equal(y_test_w, y_test)}")

    train_data = tf.data.Dataset.from_tensor_slices((X_train_w, y_train_w))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((X_test_w, y_test_w))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    return X_train_w, X_test_w, train_data, val_data

def lstm_model():
    dropout = 0.0
    X_train_w,X_test_w, train_data, val_data = window_gen()
    X_train, y_train, X_test, y_test = train_test_mult()
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=X_train_w.shape[-2:], dropout=dropout),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='rmsprop', loss='mae')
    EVALUATION_INTERVAL = 100
    EPOCHS = 100

    model_history = simple_lstm_model.fit(train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data, validation_steps=50)
    yhat = simple_lstm_model.predict(X_test_w).reshape(1, -1)[0]
    err1 = mean_absolute_error(y_test, yhat)
    val_2 = yhat

    print(err1)
    plt.figure()
    plt.plot(y_test.values, label='Original')
    plt.plot(val_2, color='red', label='Predicted')
    plt.legend()
    plt.title('KNN, n_neighbors = 2')
    plt.show()

    return None