#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpaches
import matplotlib as mpl
import backtrader as bt 
import yfinance as yf





from datetime import datetime


# In[2]:


stock = 'BA'
startdate = '2021-08-10'
enddate = '2022-08-11'


df = yf.download(stock,start=startdate, end=enddate,progress=False)
ba = yf.download(stock,start=startdate, end=enddate,progress=False)
df.head()


# In[3]:


df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)# inplace meaning
df['simple_rtn']=df.adj_close.pct_change()# meaning 
df['log_rtn']=np.log(df.adj_close/df.adj_close.shift(1))


# In[ ]:





# In[4]:


df.head()


# In[5]:


#Following charts present the prices of MS as well as its simple and logarithmic
fig,ax= plt.subplots(3,1 ,figsize=(24,20),sharex=False)
plt.subplots_adjust(bottom=0.1,  top=0.9, hspace=0.4)


df.adj_close.plot(ax=ax[0],fontsize=16)


ax[0].set_title('MS time series',fontsize=20,color='brown')
ax[0].set_ylabel('Stock price ($)',fontsize=20,color='brown')
ax[0].set_xlabel('Date',fontsize=20,color='brown')


df.simple_rtn.plot(ax=ax[1],fontsize=16)

ax[1].set_title('MS time series',fontsize=20,color='brown')
ax[1].set_ylabel('Simple return (%)',fontsize=20,color='brown')
ax[1].set_xlabel('Date',fontsize=20,color='brown')

df.log_rtn.plot(ax=ax[2],fontsize=16)
ax[2].set_title('MS time series',fontsize=20,color='brown')
ax[2].set_ylabel('Log return (%)',fontsize=20,color='brown')
ax[2].set_xlabel('Date',fontsize=20,color='brown')


# In[6]:


df_rolling = df[['simple_rtn']].rolling(window=22).aggregate(['mean','std'])


df_rolling.columns = df_rolling.columns.droplevel()
df_outliers = df.join(df_rolling)
def indentify_outliers(row, n_sigmas=3) :
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma) :
        return 1
    else:
        return 0 


# In[7]:


df_outliers['outlier'] = df_outliers.apply(indentify_outliers,axis=1)
outliers = df_outliers.loc[df_outliers['outlier']==1,['simple_rtn']]

fig, ax = plt.subplots(figsize=(18,12))
ax.plot(df_outliers.index,df_outliers.simple_rtn,color='blue',label='Normal')
ax.scatter(outliers.index,outliers.simple_rtn, color='red',label='Anomaly')
ax.set_title("Boeing Company stock returns")
ax.legend(loc='upper left')


# In[ ]:





# In[8]:


class SmaStrategy (bt.Strategy):    #backtrader strategy
    params = (('ma_period', 20), )  #try running for 25, 30, 45,90 and compare profit
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.price = None
        self.comm = None
        self.sma = bt.ind.SMA(self.datas[0],period=self.params.ma_period)
        

    def log(self, txt):
        dt=self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Price:{order.executed.price:.2f},Cost:{order.executed.value:.2f},Commission:{order.executed.comm:.2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm 
            else:
                self.log(f' SELL EXECUTED --- Price:{order.executed.price:.2f},Cost:{order.executed.value:.2f},Commission:{order.executed.comm:.2f}')        
                self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                    self.log('Order Failed')
        self.order=None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT ---Gross: {trade.pnl:.2f},Net:{trade.pnlcomm:.2f}')

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.data_close[0] > self.sma[0]:
                self.log(f' BUY CREATED --- Price:{self.data_close[0]:.2f}')
                self.order = self.buy()
        else:
            if self.data_close[0] < self.sma[0]:
                    self.log(f' SELL CREATED --- Price:{self.data_close[0]:.2f}')
                    self.order = self.sell()


# In[ ]:





# In[9]:


pip install yfinance


# In[10]:


pip install backtesting


# In[11]:


import yfinance as yf
import backtrader as bt

cash = 10000000
commission = 0.001
percentageUsed = 10

cerebro = bt.Cerebro(stdstats=False)
data = bt.feeds.PandasData(dataname=yf.download(stock,startdate ,enddate ))
# data = bt.feeds.YahooFinanceData(dataname='GOOG',fromdate=datetime(2021,8,3),todate=datetime(2022,8,4))


# In[12]:


cerebro.adddata(data)


# In[13]:


cerebro.broker.setcash(cash)   #cash value is $10M
cerebro.addstrategy(SmaStrategy)
cerebro.broker.setcommission(commission)
cerebro.addsizer(bt.sizers.PercentSizer,percents=percentageUsed)  # only 10% of our portfolio is used
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)


# In[14]:


print (f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.broker.getvalue()
cerebro.run()
print(f'Final Portfolio Value:  {cerebro.broker.getvalue():.2f}')
profit = cerebro.broker.getvalue() - cash
print('+'*50)
print(f'Profit value is : {profit:.2f}')
print('+'*50)
cerebro.plot()

#if ImportError Cannot import name warnings from matplotlib.dates
#pip uninstall backtrader
#pip install git+https://github.com/mementum/backtrader.git@0fa63ef4a35dc53cc7320813f8b15480c8f85517


# In[15]:


# rerun backtesting without setting any commissions

cerebro = bt.Cerebro(stdstats=False)
data = bt.feeds.PandasData(dataname=yf.download(stock, startdate, enddate))
cerebro.adddata(data)
cerebro.broker.setcash(cash)   #cash value is $10M
cerebro.addstrategy(SmaStrategy)
cerebro.broker.setcommission(0)
cerebro.addsizer(bt.sizers.PercentSizer,percents=10)  # only 10% of our portfolio is used
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
print (f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.broker.getvalue()
cerebro.run()
print(f'Final Portfolio Value:  {cerebro.broker.getvalue():.2f}')
profitNocomm = cerebro.broker.getvalue() - cash
comm = profitNocomm - profit
print(f'profit without commission is :{profitNocomm:.2f}')
print('+'*50)
print(f'costs of the transactions is  :{comm:.2f}')
print('+'*50)


# In[16]:


# pip install fastquant


# In[17]:


# from __future__ import (absolute_import,division,print_function,unicode_literals,)

# import backtrader as bt
# from datetime import datetime
# from fastquant.strategies.base import BaseStrategy

# class MACDStrategy(BaseStrategy):
#     params = (("fast_period", 12),("slow_period", 26),("signal_period", 9),)
#     def __init__(self):
#         super().__init__()

#         self.fast_period = self.params. fast_period
#         self.slow_period = self.params.slow_period
#         self.signal_period = self.params.signal_period
#         self.commission = self.params.commission

#         if self.strategy_logging:
#             print("===Strategy level arguments===")
#             print("fast_period :", self.fast_period)
#             print("slow_period :", self.slow_period)
#             print("signal_period :", self.signal_period)

#         macd_ind = bt.ind.MACD(
#         period_mel=self.fast_period,
#         period_me2=self.slow_period,
#         period_signal=self.signal_period,)

#         self.macd = macd_ind.macd
#         self.signal = macd_ind.signal
#         self.crossover = bt.ind.CrossOver(self.macd, self.signal)
        
#     def buy_signal(self):
#         return self.crossover > 0

#     def sell_signal(self):
#         return self.crossover < 0



# In[18]:


#MACDStrategy

# cerebro = bt.Cerebro(stdstats=False)
# data = bt.feeds.PandasData(dataname=yf.download(stock, startdate, enddate))
# cerebro.adddata(data)
# cerebro.broker.setcash(cash)   
# cerebro.addstrategy(MACDStrategy)
# cerebro.broker.setcommission(commission)
# cerebro.addsizer(bt.sizers.PercentSizer,percents=10)  
# cerebro.addobserver(bt.observers.BuySell)
# cerebro.addobserver(bt.observers.Value)
# print (f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
# cerebro.broker.getvalue()
# cerebro.run()
# print(f'Final Portfolio Value:  {cerebro.broker.getvalue():.2f}')
# profitNocomm = cerebro.broker.getvalue() - cash
# comm = profitNocomm - profit
# print(f'profit without commission is :{profitNocomm:.2f}')
# print('+'*50)
# print(f'costs of the transactions is  :{comm:.2f}')
# print('+'*50)

# cerebro.plot()


# In[ ]:





# In[19]:


pip install bokeh


# In[20]:


pip install echo


# In[21]:


pip install backtrader


# In[22]:


pip install bt


# In[23]:


pip install ffn 


# In[ ]:





# In[24]:


import bt
 # fetch some data
data = bt.get('spy,agg', start='2010-01-01')
data.head()


# In[25]:


import ffn
returns = ffn.get('BA,aapl,msft,TSLA', start='2010-01-01').to_returns().dropna()
returns.calc_mean_var_weights().as_format('.2%')


# In[ ]:





# In[26]:


# create the strategy
s = bt.Strategy('s1', [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])


# In[27]:


# create a backtest and run it
test = bt.Backtest(s, data)
res = bt.run(test)


# In[28]:


res.plot()


# In[29]:


res.display()


# In[30]:


res.plot_histogram()


# In[31]:


res.plot_security_weights()


# In[32]:


from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA


class SmaCross(Strategy):
    n1 = 5
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


bt = Backtest(ba, SmaCross,
              cash=cash/10, commission=commission,
              exclusive_orders=True)

output = bt.run()

bt.plot()


# In[33]:


pip install ta-lib


# In[34]:


import numpy
import talib

close = numpy.random.random(100)


# In[35]:


output = talib.SMA(close)


# In[36]:


from talib import MA_Type

upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)


# In[37]:


output = talib.MOM(close, timeperiod=5)


# In[38]:


from backtesting import Backtest, Strategy
from backtesting.lib import crossover
#https://www.newtraderu.com/2022/05/14/moving-average-crossover-backtest-results/#:~:text=How%20do%20you%20do%20a,or%20break%20below%20each%20other.
from backtesting.test import SMA

class SmaCross(Strategy):
    n1 = 5
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


bt = Backtest(ba, SmaCross,
              cash=cash/10, commission=commission,
              exclusive_orders=True)



output = bt.run()
bt.plot()


# In[39]:


ba.tail()


# In[40]:


import pandas as pd


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


# In[41]:


from backtesting import Strategy
from backtesting.lib import crossover


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 5
    n2 = 20
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()
            
bt = Backtest(ba, SmaCross,
              cash=cash/10, commission=commission,
              exclusive_orders=True)



output = bt.run()
bt.plot()          


# In[42]:


from backtesting import Backtest

bt = Backtest(ba, SmaCross, cash=cash/10, commission=commission)
stats = bt.run()
stats


# In[43]:


bt.plot()


# In[44]:


get_ipython().run_cell_magic('time', '', "\nstats = bt.optimize(n1=range(5, 30, 5),\n                    n2=range(10, 70, 5),\n                    maximize='Equity Final [$]',\n                    constraint=lambda param: param.n1 < param.n2)\n")


# In[45]:


stats


# In[46]:


stats._strategy


# In[47]:


bt.plot(plot_volume=False, plot_pl=False)


# In[48]:


stats.tail()


# In[49]:


stats['_equity_curve'] 


# In[50]:


stats['_trades']

