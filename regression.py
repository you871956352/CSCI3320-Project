# CHEN Jiamin

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Read data from csv files
train_data = pd.read_csv('training.csv')

# Select features and convert dataframe to array
train_data = train_data[['actual_weight','declared_horse_weight','draw',
                        'win_odds','jockey_ave_rank','trainer_ave_rank',
                        'recent_ave_rank','race_distance','finish_time']].values
X = train_data[:,:8]
y = train_data[:,8]
def timeinseconds(timestring):
    min,sec,point = timestring.split(".")
    return float(min)*60+float(sec)+float(point)/100.0
y = map(lambda x: timeinseconds(x),y)


# SVR regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
