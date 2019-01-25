import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import features

forecast_days = 30
moving_avg_window = 10

df = pd.read_csv("/Users/adityanarkar/Aditya/Project/Implementation/Data/M&MFIN.NS.csv")

# Selecting features that brings in value
df = df[['Open', 'Adj Close', 'High', 'Low']]
df['HL_PCT'] = ((df['High'] - df['Low']) / df['Low']) * 100
df['PCT_CHNG'] = ((df['Open'] - df['Adj Close']) / df['Adj Close']) * 100

df['Moving_Avg'] = np.nan

features.fifty_day_MA(df, moving_avg_window)
print(df.tail())
df.dropna(inplace=True)
print(df.tail())
# df.dropna(how='any', subset=['Open', 'Adj Close', 'High', 'Low'], inplace=True)


df['label'] = df['Adj Close'].shift(-forecast_days)

X_temp = np.array(df.drop(['label'], 1))

# Scaled X in the range of -1 to 1.
X_temp = preprocessing.scale(X_temp)

X = X_temp[:-forecast_days]

X_lately = X_temp[-forecast_days:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test, y_test))

forecast = reg.predict(X_lately)
print(forecast)
