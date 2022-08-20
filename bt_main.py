#import the libs

import pandas as pd
import numpy as np
import backtrader as bt 
import yfinance as yf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from termcolor import colored as cl 
from datetime import datetime
import plotly.express as px


stock = 'BA'
startdate = '2021-08-10'
enddate = '2022-08-11'

df = yf.download(stock,start=startdate, end=enddate,progress=False)
df.head()


df = df.loc[:,['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
df['simple_return']=df.adj_close.pct_change() 
df['log_return']=np.log(df.adj_close/df.adj_close.shift(1))
df.head()

# In[6]:

fig,ax= plt.subplots(3,1 ,figsize=(24,30),sharex=False)
plt.subplots_adjust(bottom=0.1,  top=0.9, hspace=0.4)

df.adj_close.plot(ax=ax[0],fontsize=16)

ax[0].set_title('Boeing Close price history',fontsize=20,color='brown')
ax[0].set_ylabel('close price (USD)',fontsize=20,color='brown')
ax[0].set_xlabel('Date',fontsize=20,color='brown')


df.simple_return.plot(ax=ax[1],fontsize=16)
ax[1].set_title('Boeing stock simple return',fontsize=20,color='brown')
ax[1].set_ylabel('Simple return (%)',fontsize=20,color='brown')
ax[1].set_xlabel('Date',fontsize=20,color='brown')

df.log_return.plot(ax=ax[2],fontsize=16)
ax[2].set_title('Boeing stock log return',fontsize=20,color='brown')
ax[2].set_ylabel('Log return (%)',fontsize=20,color='brown')
ax[2].set_xlabel('Date',fontsize=20,color='brown')

# In[7]:

df.describe()

# In[8]:


fig = px.box(df, y='adj_close')

fig.show()


# In[9]:

fig = px.histogram(df, x="adj_close")
fig.show()

# In[10]:

cash = 10000000
commission = 0.001
percentageUsed = 10

prev_cash = cash

class SMAStrategy (bt.Strategy):    
    params = (('p1', 5), )
   
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        
        print(f'{dt.isoformat()} | {txt}')
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.price = None
        self.comm = None
        self.sma = bt.ind.SimpleMovingAverage(self.data0,period=self.params.p1)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
             # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(cl(f'   BUY ORDER EXECUTED!  [Price:{order.executed.price:.2f} - Cost:{order.executed.value:.2f} - Commission:{order.executed.comm:.2f}]',color='green',attrs=['bold']))
                self.price = order.executed.price
                self.comm = order.executed.comm 
            else:
                self.log(cl(f'   SELL ORDER EXECUTED! [Price:{order.executed.price:.2f} - Cost:{order.executed.value:.2f} - Commission:{order.executed.comm:.2f}]',color='red',attrs=['bold']))        
                self.bar_executed = len(self)

        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order=None
        
    
    def notify_trade(self, trade):
        self.log(cl(f'OPERATION PROFIT: [GROSS : {trade.pnl:.2f} - NET :{trade.pnlcomm:.2f}]',attrs=['bold']))
    
    def notify_fund(self,cash, value, fundvalue, shares):
        global prev_cash
        if(cash == prev_cash):
            return
        else:
            self.log(cl(f'CURRENT FUNDS:   [CASH: {cash:.2f} - VALUE:{value:.2f} - FUND VALUE:{fundvalue:.2f} - SHARES:{shares:.2f}]\n',attrs=['bold']))
            prev_cash = cash
        

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

           
            if self.dataclose[0] > self.sma[0]:

                
                self.log(cl(f'▶ BUY ORDER IS CREATED!   Price:{self.datas[0].close[0]:.2f}',color='green',attrs=['bold']))

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
              
                self.log(cl(f'▶ SELL ORDER IS CREATED! Price:{self.datas[0].close[0]:.2f}',color='red',attrs=['bold']))

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


# In[11]:

def uselessFunction(uselessArg):
    print('\n')
    print( '+' * 50)
    print('+',' '*46,'+')
    print ('+',' '*4,uselessArg,' '*4,)
    print('+',' '*46,'+')
    print('+'*50,'\n')


# In[12]:

cerebro = bt.Cerebro(stdstats=False)
data = bt.feeds.PandasData(dataname=yf.download(stock,startdate ,enddate ))

cerebro.adddata(data)
cerebro.broker.setcash(cash)   

cerebro.addstrategy(SMAStrategy,p1=10)

cerebro.broker.setcommission(commission)

cerebro.addsizer(bt.sizers.PercentSizer,percents=percentageUsed)   #This sizer return percents of available cash
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Broker)
cerebro.addobserver(bt.observers.Trades)

uselessFunction(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')

cerebro.run()

uselessFunction(f'Final Portfolio Value:  {cerebro.broker.getvalue():.2f}\n+\n+      Profit is : {cerebro.broker.getvalue() - cash:.2f}')

plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.dpi']=100
plt.style.use('classic')

cerebro.plot()


# In[23]:


cash = 10000000
commission = 0.001
percentageUsed = 10

prev_cash = cash



class   EMAStrategy (bt.Strategy):    
    params = (('period_me1', 5), ('period_me2', 26), ('period_signal', 9),)
   
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        
        print(f'{dt.isoformat()} | {txt}')
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        
        self.sma1 = bt.ind.SimpleMovingAverage(self.data0,period=50)
        ema1 = bt.ind.ExponentialMovingAverage(self.data0, period=15)

        close_over_sma = data.close > self.sma1
        close_over_ema = data.close > ema1
        sma_ema_diff = self.sma1 - ema1

        self.buy_sig = bt.And(close_over_sma, close_over_ema, sma_ema_diff > 0)
        self.sell_sig = bt.And(close_over_sma, close_over_ema, sma_ema_diff < 0) 

    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            if self.buy_sig:
                
                self.log(cl(f'▶ BUY ORDER IS CREATED!   Price:{self.datas[0].close[0]:.2f}',color='green',attrs=['bold']))
                # Keep track of the created order to avoid a 2nd order
                self.buy()
        else:
            if self.sell_sig:    
                self.log(cl(f'▶ SELL ORDER IS CREATED! Price:{self.datas[0].close[0]:.2f}',color='red',attrs=['bold']))
                # Keep track of the created order to avoid a 2nd order
                self.sell()


# In[24]:

cerebro = bt.Cerebro(stdstats=False)
data = bt.feeds.PandasData(dataname=yf.download(stock,startdate ,enddate ))

cerebro.adddata(data)
cerebro.broker.setcash(cash)   

cerebro.addstrategy(EMAStrategy)

cerebro.broker.setcommission(commission)

cerebro.addsizer(bt.sizers.PercentSizer,percents=percentageUsed)   #This sizer return percents of available cash
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Broker)
cerebro.addobserver(bt.observers.Trades)

uselessFunction(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')

cerebro.run()

uselessFunction(f'Final Portfolio Value:  {cerebro.broker.getvalue():.2f}\n+\n+      Profit is : {cerebro.broker.getvalue() - cash:.2f}')

plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.dpi']=100
plt.style.use('classic')

cerebro.plot()
