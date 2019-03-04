# Stock-Prediction

Plan of action

1. Follow a simple tutorial to create a basic stock predictor

2. Add more features and test it in real market

3. Add support for more algorithms and test them in real stock market

[Tutorial](https://www.youtube.com/watch?v=r4mwkS2T9aI)

[Dataset](https://in.finance.yahoo.com/quote/M%26MFIN.NS/history?period1=1199167200&period2=1548050400&interval=1d&filter=history&frequency=1d)

**Jan 23 2019**
* Implemented Simple Linear Rgression
* Added 50 day moving average feature
    * By adding this feature, prediction changed and went closer to the actual price value.
* Learnt Pandas dataframe -- .iloc 
* iloc is exclusive of upper bound while accessing multiple rows or columns at the same time

**Feb 04 2019**
* Total features added --> Simple MA, Weighted MA, Momentum, Stochastic K, Stochastic D, RSI, MACD.

**Feb 10 2019**

* Todo: Discretize the features, test with other stocks, add SVM model (try different values to set its parameters)

**Mar 04 2019**
How ARMA process models can help predict the stock's future values. 
* [Basics](https://www.youtube.com/watch?v=v70-kLB3BLM)
* ARMA process models will try to find a better value for the `rho`.
