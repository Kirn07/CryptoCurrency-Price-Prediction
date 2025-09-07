from algorithms import *
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import math


def random_forest(df):
    days = 12
    df1 = df['Close']

    # splitting dataset into train 80% and test split 20%
    training_size = int(len(df1) * 0.80)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size], df1[training_size:len(df1)]

    # create features and target data for training set
    X_train, y_train = [], []
    for i in range(50, len(train_data)):
        X_train.append(train_data[i-50:i])
        y_train.append(train_data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # create features and target data for test set
    inputs = df1[len(df1) - len(test_data) - 50:].values
    inputs = inputs.reshape(-1, 1)
    X_test = []
    for i in range(50, 50+len(test_data)):
        X_test.append(inputs[i-50:i, 0])
    X_test = np.array(X_test)

    # create Random Forest model and train it on the training set
    model = RandomForestRegressor(
        n_estimators=1000, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    # make predictions on test set
    y_pred = model.predict(X_test)

    # calculate RMSE performance metric
    rmse = math.sqrt(mean_squared_error(test_data, y_pred))

    # generate forecast for the next 'days' days
    lst_output = []
    for i in range(days):
        if i == 0:
            x_input = X_test[-1]
        else:
            x_input = np.concatenate(
                (X_test[-1][1:], np.array([forecasted_stock_price[i-1]])))
        x_input = x_input.reshape(1, -1)
        yhat = model.predict(x_input)
        lst_output.append(yhat[0])
    forecasted_stock_price = np.array(lst_output).reshape(-1, 1)

    # Getting original prices back from scaled values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(np.array(df1).reshape(-1, 1))
    forecasted_stock_price = scaler.inverse_transform(forecasted_stock_price)

    return forecasted_stock_price, rmse
