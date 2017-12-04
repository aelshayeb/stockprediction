#To use machine learning algorithms to predict stock prices based on classification of buy/hold/sell stocks

from django.apps import AppConfig

class StockPricesConfig(AppConfig):
    name = 'stock_prices'

#stats packages
import numpy as np
import pandas as pd
import quandl as ql
import math
from sklearn import preprocessing, cross_validation, svm, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use("ggplot")


#for general purposes of printing...
stock = "NYSE:AMZN"

ql.ApiConfig.api_key = "KXs7ei6aAkAu5zhWUMsQ"

#Extract data from Quandl
df1 = ql.get('WIKI/AMZN', start_date="2000-12-31", end_date=pd.to_datetime('today'))
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

#Get bear/bull signal. If the following day is greater than the previous day,
#the value is 1, otherwise it is 0
df1['bear_bull'] = (df1['Adj. Close'].diff(-30) < 0).astype(int)
df1.fillna(-9999999, inplace=True)

#Set X matrix and y array as features and classification column respectively
X = np.array(df1.drop(['bear_bull'],1))
y = np.array(df1['bear_bull'])

#Split the data by 20% test and 80% train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Train the classifier: Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Accuracy score of the model
logreg_score = float("{0:.3f}".format(logreg.score(X_test, y_test) * 100))
print("Logistic Regression accuracy is", logreg_score, "%")

#same as above
#y_pred_class = clf.predict(X_test)
#print (metrics.accuracy_score(y_test, y_pred_class))


#Train the classifier: Logistic Regression
QDA = QDA()
QDA.fit(X_train, y_train)

#Accuracy score of the model
QDA_score = float("{0:.3f}".format(QDA.score(X_test, y_test) * 100))
print("Quadratic Discriminant Analysis accuracy is", QDA_score, "%")

#Train the classifier: Logistic Regression
GNB = GNB()
GNB.fit(X_train, y_train)

#Accuracy score of the model
GNB_score = float("{0:.3f}".format(GNB.score(X_test, y_test) * 100))
print("Gaussian Naive Bayes accuracy is", GNB_score, "%")

#Train the classifier: Logistic Regression
SVC = SVC()
SVC.fit(X_train, y_train)

#Accuracy score of the model
SVC_score = float("{0:.3f}".format(SVC.score(X_test, y_test) * 100))
print("Support Vector Classification accuracy is", SVC_score, "%")

#null accuracy for LogisticRegression
#count how many 0s and 1s
counter = df1['bear_bull'].value_counts()
print(counter)
#null accuracy: what a dumb model would predict
null_accuracy = 1 - df1['bear_bull'].mean()
print('Null accuracy: ', null_accuracy)


# Above is classification models, I will use the SVM regression model to predict
# actual values and then use that to determine strategy to buy, hold or sell.
forecast_col = "Adj. Close"

# 30 days out
forecast_out = int(math.ceil( 0.007 * len(df1) ))

df1["Label"] = df1[forecast_col].shift(-forecast_out)
df1.dropna(inplace=True)

X = np.array(df1.drop(['bear_bull', 'Label'],1))
X = preprocessing.scale(X)
y = np.array(df1["Label"])

# 30% of the data will be the test set as used in the Stanford paper
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

clf = svm.SVR(kernel = 'poly')
clf.fit(X_train, y_train)

predictedValues = clf.predict(X_lately)

# Uses the prediction over 30 days to report whether to buy, sell or hold the stock
def buySellHold():
    tomorrowPrice = predictedValues[0]
    priceInThirtyDays = predictedValues[-1]
    
    isHold = False if input('Are you holding this stock? (input yes or no): ').lower() == 'no' else True
    
    if tomorrowPrice < priceInThirtyDays and isHold:
        print('Continue to hold this stock.')
    elif tomorrowPrice < priceInThirtyDays and not isHold:
        print('Buy this stock.')
    elif tomorrowPrice > priceInThirtyDays and not isHold:
        print('Do not buy this stock.')
    else:
        print('Sell this stock.')


buySellHold()


        
        
        