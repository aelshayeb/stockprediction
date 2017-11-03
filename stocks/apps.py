from django.apps import AppConfig

class StockPricesConfig(AppConfig):
    name = 'stock_prices'

#stats packages
#import statistics as stats
import numpy as np
import pandas as pd
import quandl as ql
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use("ggplot")


#for general purposes of printing...
stock = "AMZN"

ql.ApiConfig.api_key = "KXs7ei6aAkAu5zhWUMsQ"

df1 = ql.get('WIKI/AMZN', start_date="2000-12-31", end_date="2010-12-31")
df1 = df1[["Adj. Open", "Adj. Close", "Adj. High", "Adj. Low", "Adj. Volume"]]
df1['HL_PCT'] = (df1["Adj. High"] - df1["Adj. Close"]) / df1["Adj. Close"] * 100
df1["PCT_change"] = (df1["Adj. Close"] - df1["Adj. Open"]) / df1["Adj. Open"] * 100

df1 = df1[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

#set which colum to attribute label column with
forecast_col = "Adj. Close"

#teach the model to ignore outliers by filling all NaN with -999999 instead of removing the dataset
df1.fillna(-9999999, inplace=True)

#set model to frecast stock price into the future
forecast_out = int(math.ceil( 0.001 * len(df1) ))
df1["Label"] = df1[forecast_col].shift(-forecast_out)
df1.dropna(inplace=True)

#Features - (place in numpy array)
X = np.array(df1.drop(['Label'], 1))

#standardizing data. Gaussian with zero mean and unit variance
#for more info, see http://scikit-learn.org/stable/modules/preprocessing.html
X = preprocessing.scale(X)

#redefine labels as a numpy array
y = np.array(df1["Label"])

#set testing size (20%) verify cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False )

#define forecasted column
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

#define classifier for trainning and testing data
#MODEL: linear regression model
clf1 = LinearRegression(n_jobs=-1)
clf1.fit(X_train, y_train)
clf1.score(X_test, y_test)
#get accuracy (squared error)
accuracyLR = float("{0:.3f}".format(clf1.score(X_test, y_test) * 100))
#Set the forcasted stock price array as X_lately, as defined above
forecast_setLR = clf1.predict(X_lately)


#MODEL: SVM model, do more research to specify kernels
clf2 = svm.SVR(kernel = 'linear')
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)
#get accuracy (squared error)
accuracySVM = float("{0:.3f}".format(clf2.score(X_test, y_test) * 100))
#Set the forcasted stock price array as X_lately, as defined above
forecast_setSVM = clf2.predict(X_lately)



#make sure forecast as non NaN values

#output results in python shell
#print(X_train)
#print(y_train)
#print (df1)
#print("------------------------")
#print("ANALYSIS:")
#print("Analyzing", stock, "stock data for", len(df1), "days using", len(df1) * 5, "data points")
#print("Forecasting stock price" , forecast_out, "days into the future")
#print ("Model accuracy is", accuracyLR ,"% using Linear Regression" )
#print ("Model accuracy is", accuracySVM ,"% using SVM" )
#print("------------------------")
#print ("FORECASTED STOCK PRICES: LINEAR REGRESSION")
#print (forecast_setLR)
#print ("FORECASTED STOCK PRICES: SVM")
#print (forecast_setSVM)
#print("Feature coefficients:" , X_train)
#print("Label coefficients:" , y_train)


df1['forecast'] = np.NaN
last_date = df1.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#set forecasted stock price as i
for i in forecast_setLR:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df1.loc[next_date] = [np.NaN for _ in range(len(df1.columns)-1)] + [i]

df1['Adj. Close'].plot()
df1['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
