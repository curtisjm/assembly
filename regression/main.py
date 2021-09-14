import pandas as pd
import quandl, math, datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
from matplotlib import style

print("\n")

style.use("ggplot")
# find tickers to use in get function from Quandl's website
# returns data set from stocks
# 	- https://www.quandl.com/
# df = data frame
df = quandl.get("WIKI/GOOGL")

# only get these values
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
# high-low percent
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100
# daily percent change
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100

# take only important values
# adjusted close is the only thing that really hinges on price
df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

# guess future values
forecast_col = "Adj. Close"
# replace NaN values
# treated as outliers in dataset
df.fillna(-99999, inplace=True)

# predict out 10% of the data frame, var will be number of days
forecast_out = int(math.ceil(0.1 * len(df)))

# shift index by desired number of periods (up in our case)
df["label"] = df[forecast_col].shift(-forecast_out)

# X = features : input (one column of the data in input set)
X = np.array(df.drop(columns="label", axis=1))

# scale x - standardize a dataset along any axis
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

# y = labels : output
y = np.array(df["label"])

# create training and testing sets
# shuffle features and labels, keeping x's and y's connected
# get different data for training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# find classifier (a classifier is a type of machine learning algorithm used to assign a class label to a data input)
# use the n_jobs arg to set how many processes to run training on (-1 = as many as possible)
clf = LinearRegression(n_jobs=-1)
# fit / train classifier, fit features and labels
clf.fit(X_train, y_train)

# use pickle to save classifier in a file
with open("./linearRegression.pickle", "wb") as f:
    pickle.dump(clf, f)

# use saved classifier
pickle_in = open("./linearRegression.pickle", "rb")
clf = pickle.load(pickle_in)

# test classifier
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print("Accuracy: " + str(accuracy))
print("FC out: " + str(forecast_out))
print("FC set: " + str(forecast_set))

# col full of NaN data
df["Forecast"] = np.nan

# predictions
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
# seconds in a day
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # take all first columns, sets them to NaNs, final column is i (forecast)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

print()
