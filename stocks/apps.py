from django.apps import AppConfig

class StockPricesConfig(AppConfig):
    name = 'stock_prices'

#stats packages
import numpy as np
import pandas as pd
import quandl as ql
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use("ggplot")

#for general purposes of printing...
stock = "NYSE:AMZN"

ql.ApiConfig.api_key = "KXs7ei6aAkAu5zhWUMsQ"

df1 = ql.get('WIKI/AMZN', start_date="2000-12-31", end_date="2010-12-31")
df1 = df1[["Adj. Open", "Adj. Close", "Adj. Volume"]]
#df1['HL_PCT'] = (df1["Adj. High"] - df1["Adj. Close"]) / df1["Adj. Close"] * 100
df1["PCT_change"] = (df1["Adj. Close"] - df1["Adj. Open"]) / df1["Adj. Open"] * 100
#claculating standard devilation for 10, 50, 100 days
df1["std_10"] = df1["Adj. Close"].rolling(window=10, min_periods=0).std()
df1["std_50"] = df1["Adj. Close"].rolling(window=50, min_periods=0).std()
df1["std_100"] = df1["Adj. Close"].rolling(window=100, min_periods=0).std()
#claculating moving average for 10, 50, 100 days
df1["ma_10"] = df1["Adj. Close"].rolling(window=10, min_periods=0).mean()
df1["ma_50"] = df1["Adj. Close"].rolling(window=50, min_periods=0).mean()
df1["ma_100"] = df1["Adj. Close"].rolling(window=100, min_periods=0).mean()

#Features
df1 = df1[["Adj. Close", "Adj. Volume", "PCT_change", "ma_10", "ma_50", "ma_100", "std_10", "std_50", "std_100" ]]

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
X = preprocessing.scale(X)

#redefine labels as a numpy array
y = np.array(df1["Label"])

#set testing size (20%) verify cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False )

#define forecasted column
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#define classifier for trainning and testing data
#MODEL: linear regression model
clf1 = LinearRegression(n_jobs=-1)
clf1.fit(X_train, y_train)
#scoring: 1 score of 1 is a perfect testing
score = clf1.score(X_test, y_test)
#get accuracy (squared error)
accuracyLR = float("{0:.3f}".format(clf1.score(X_test, y_test) * 100))
#Set the forcasted stock price array as X_lately, as defined above
forecast_setLR = clf1.predict(X_lately)
#mean squared error
meanSquareErrorLR = np.mean((clf1.predict(X_test)-y_test)**2)



#MODEL: SVM model, do more research to specify kernels
clf2 = svm.SVR(kernel = 'linear')
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)
#get accuracy (squared error)
accuracySVM = float("{0:.3f}".format(clf2.score(X_test, y_test) * 100))
#Set the forcasted stock price array as X_lately, as defined above
forecast_setSVM = clf2.predict(X_lately)
#mean squared error
meanSquareErrorSVM = np.mean((clf2.predict(X_test)-y_test)**2)

#binary decision to analyze bull/bear signals
end_price = df1["Adj. Close"]+1
begin_price = df1["Adj. Close"]

if end_price > begin_price:
    y = 1
else:
    y = -1




#output results
print (df1)
print("------------------------")
print("ANALYSIS:")
print("Analyzing", stock, "stock data for", len(df1), "days using", len(df1) * 5, "data points")
print("Forecasting stock price" , forecast_out, "days into the future")
print ("Model accuracy is", accuracyLR ,"% using Linear Regression, with a mean squared error of", meanSquareErrorLR)
print ("Model accuracy is", accuracySVM ,"% using SVM, with a mean squared error of", meanSquareErrorSVM)
print("------------------------")
print ("FORECASTED STOCK PRICES: LINEAR REGRESSION")
print (forecast_setLR)
print ("FORECASTED STOCK PRICES: SVM")
print (forecast_setSVM)
print("Feature coefficients:" , X_train)
print("Label coefficients:" , y_train)

#plot forecasted price in date
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

#create two graphsL ax1 and ax2
ax1 = plt.subplot2grid((10,1), (0,0), rowspan=5, colspan=1)
plt.title(stock)
ax2 = plt.subplot2grid((10,1), (6,0), rowspan=1, colspan=1, sharex=ax1)

#plot graphs (ax1 and ax2)
ax1.plot(df1.index, df1["Adj. Close"], "r-")
ax1.plot(df1.index, df1['forecast'], color="black")
ax1.plot(df1.index, df1["ma_100"], color="blue")
ax1.plot(df1.index, df1["ma_50"], color="green")
ax1.plot(df1.index, df1["ma_10"], color="orange")
ax2.bar(df1.index, df1["Adj. Volume"])

#set labels on axes
ax2.set_xlabel('Date')
ax1.set_ylabel('P (USD)')
ax2.set_ylabel('Vol')
#create legend
ax1.legend(loc=0)
#show plots
plt.show()
