__author__ = 'Anton'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

data = pd.read_csv("gdp.csv").dropna()

train_y = data[["rgdpwok"]][:-30]
train_X = data[["kc", "kg", "ki"]][:-30]

test_y = data[["rgdpwok"]][-30:]
test_X = data[["kc", "kg", "ki"]][-30:]

ols = linear_model.LinearRegression()
ols.fit(train_X, train_y)

print("R^2 train: %s\nR^2 test: %s" % (np.mean((ols.predict(train_X) - train_y) ** 2), \
                                       np.mean((ols.predict(test_X) - test_y) ** 2)) )
