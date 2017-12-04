#To use machine learning algorithms to predict stock prices based on classification of buy/hold/sell stocks

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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use("ggplot")

#for general purposes of printing...
stock = "NYSE:AMZN"

ql.ApiConfig.api_key = "KXs7ei6aAkAu5zhWUMsQ"

#Extract data from Quandl
#df1 = ql.get('WIKI/AMZN', start_date="2000-12-31", end_date=pd.to_datetime('today'))
df1 = ql.get('WIKI/AMZN', start_date="2000-12-31", end_date="2010-12-31")
df1 = df1[["Adj. Open", "Adj. Close", "Adj. Volume"]]

#df1['HL_PCT'] = (df1["Adj. High"] - df1["Adj. Low"]) / df1["Adj. Low"] * 100
df1["PCT_change"] = (df1["Adj. Close"] - df1["Adj. Open"]) / (df1["Adj. Open"] * 100)

#claculating standard deviation for 10, 50, 100 days
df1["std_10"] = df1["Adj. Close"].rolling(window=10, min_periods=0).std()
df1["std_50"] = df1["Adj. Close"].rolling(window=50, min_periods=0).std()
df1["std_100"] = df1["Adj. Close"].rolling(window=100, min_periods=0).std()
#claculating moving average for 10, 50, 100 days
df1["ma_10"] = df1["Adj. Close"].rolling(window=10, min_periods=0).mean()
df1["ma_50"] = df1["Adj. Close"].rolling(window=50, min_periods=0).mean()
df1["ma_100"] = df1["Adj. Close"].rolling(window=100, min_periods=0).mean()

#Features
df1 = df1[["Adj. Close", "Adj. Volume","PCT_change", "ma_10", "ma_50", "ma_100", "std_10", "std_50", "std_100" ]]

#set which column to attribute label column with
forecast_col = "Adj. Close"

#teach the model to ignore outliers by filling all NaN with -999999 instead of removing the dataset
df1.fillna(-9999999, inplace=True)

#set model to forecast stock price into the future
#for ref: 0.001 predicts 3 days out, 0.01 predicts ~30 days out
forecast_out = int(math.ceil( 0.001 * len(df1) ))

#Shift the Adj. Close column up 3 days in order to predict
df1["Label"] = df1[forecast_col].shift(-forecast_out)
df1.dropna(inplace=True)

#Features - (place in numpy array) and exclude the label column
X = np.array(df1.drop(['Label'], 1))

#standardizing data. Gaussian with zero mean and unit variance
X = preprocessing.scale(X)

#redefine labels as a numpy array
y = np.array(df1["Label"])

#set testing size (20%) verify cross validation
#ref: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#define forecasted column
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#define classifier for training and testing data
#MODEL: linear regression model
#Optional: Save to pickle so that I don't have to train the model every time it is used
clf1 = LinearRegression(n_jobs=-1)
clf1.fit(X_train, y_train)

#scoring: 1 score of 1 is a perfect testing - Scoring = accuracy of the model
scoreLR = float("{0:.3f}".format(clf1.score(X_test, y_test) * 100))

#Set the forecasted stock price array as X_lately, as defined above
forecast_setLR = clf1.predict(X_lately)

#mean squared error using Sklearn function
meanSquaredErrorLR = mean_squared_error(y_test, clf1.predict(X_test))
meanAbsoluteErrorLR = mean_absolute_error(y_test, clf1.predict(X_test))
rootMeanSquareErrorLR = np.sqrt(mean_squared_error(y_test, clf1.predict(X_test)))

#MODEL: SVM model, do more research to specify kernels
clf2 = svm.SVR(kernel = 'linear')
clf2.fit(X_train, y_train)

#Use the score variable times 100 for the accuracy, it's the same as below
scoreSVM = float("{0:.3f}".format(clf2.score(X_test, y_test) * 100))

#Set the forcasted stock price array as X_lately, as defined above
forecast_setSVM = clf2.predict(X_lately)

#mean squared error using Sklearn function
meanSquareErrorSVM = mean_squared_error(y_test, clf2.predict(X_test))
meanSquaredErrorSVM = mean_squared_error(y_test, clf2.predict(X_test))
meanAbsoluteErrorSVM = mean_absolute_error(y_test, clf2.predict(X_test))
rootMeanSquareErrorSVM = np.sqrt(mean_squared_error(y_test, clf2.predict(X_test)))

#Function to label
def buyHoldSellStock():
    #Average of both model's predicted prices
    tomorrowPrice = (forecast_setLR[0] + forecast_setSVM[0])/2
    todayPrice = df1['Adj. Close'][-1]
    difference = ((tomorrowPrice - todayPrice)/(todayPrice)) * 100
    if difference > 10:
        decision = 2
    elif difference > 5:
        decision = 1
    elif difference < 0:
        decision = -1
    else:
        decision = 0

    if decision == 0:
        print("Percent change of tomorrow's predicted price and today's price: ", difference, '%', 'Hold stock.')
    elif decision == 2:
        print("Percent change of tomorrow's predicted price and today's price: ", difference, '%', 'Strong buy stock.')
    elif decision == 1:
        print("Percent change of tomorrow's predicted price and today's price: ", difference, '%', 'Buy stock.')
    elif decision == -1:
        print("Percent change of tomorrow's predicted price and today's price: ", difference, '%', 'Sell stock.')

buyHoldSellStock()


#output results
#print (df1)
print("------------------------")
print("ANALYSIS:")
print("------------------------")
print("Analyzing", stock, "stock data for", len(df1), "days using", len(df1) * 3, "data points")
print("Forecasting stock price" , forecast_out, "days into the future")
if scoreLR > 0 :
    print ("Model accuracy is", scoreLR ,"% using Linear Regression, with a mean squared error of", meanSquaredErrorLR, "root square error of", rootMeanSquareErrorLR, " and mean absolute error of", meanAbsoluteErrorLR)

print ("Model accuracy is", scoreSVM ,"% using SVM, with a mean squared error of", meanSquaredErrorSVM, "root square error of", rootMeanSquareErrorSVM, " and mean absolute error of", meanAbsoluteErrorSVM)
#print("------------------------")
print ("FORECASTED STOCK PRICES: LINEAR REGRESSION")
print (forecast_setLR)
print ("FORECASTED STOCK PRICES: SVM")
print (forecast_setSVM)
print(df1.tail(n=5))
#print("Feature coefficients:" , X_train)
#print("Label coefficients:" , y_train)

#plot forecasted price in dates
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
ax1.plot(df1.index, df1["ma_100"], color="blue", linewidth=1)
ax1.plot(df1.index, df1["ma_50"], color="green", linewidth=1)
ax1.plot(df1.index, df1["ma_10"], color="orange", linewidth=1)
ax2.bar(df1.index, df1["Adj. Volume"])

#set labels on axes
ax2.set_xlabel('Date')
ax1.set_ylabel('P (USD)')
ax2.set_ylabel('Vol')
#create legend
ax1.legend(loc=0)
#show plots
plt.show()
