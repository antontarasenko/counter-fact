__author__ = 'Anton Tarasenko'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestRegressor
import pickle, os, sys

pd.options.display.float_format = '{:,.2f}'.format

def main():
    # init if a prog
    data = pd.read_csv(os.path.realpath('') + "/data/gdp.csv").dropna()

    train_y = data[["rgdpwok"]][:-30].as_matrix()
    train_X = data[["kc", "kg", "ki"]][:-30].as_matrix()

    test_y = data[["rgdpwok"]][-30:].as_matrix()
    test_X = data[["kc", "kg", "ki"]][-30:].as_matrix()

    trained = train(train_y, train_X)
    mtab = get_metrics_table(train_y, train_X, trained)
    print(mtab)

def train(y, X):
    models = dict()

    # OLS
    lr = linear_model.LinearRegression()
    models['lr'] = lr.fit(X, y)

    # RandomForest
    rf = RandomForestRegressor()
    models['rf'] = rf.fit(X, y)

    return {k: pickle.dumps(v) for k, v in models.items()}

def predict(X, model):
    pred = pickle.loads(model)

    return pred.predict(X)

def get_metrics(y, y_pred):
    metr = {'mse': metrics.mean_squared_error(y, y_pred),
            'mae': metrics.mean_absolute_error(y, y_pred)}

    return metr

def get_metrics_table(y, X, models):
    mtab = pd.DataFrame()
    for name, model in models.items():
        metr = dict({'_model': name}.items() | get_metrics(y, predict(X, model)).items())
        mtab = mtab.append(metr, ignore_index=True)

    return mtab

if __name__ == "__main__":
    main()