import pandas as pd 
import numpy as np 
from collections import namedtuple
from datetime import datetime

Account = namedtuple('Account', ['Date', 'Holdings'])
Account.__doc__='''
Investment account information for holdings at given date.

date: datetime object, the version/standing-date of account information
holdings: a dict with tickers as keys and holding shares/quantity as values.
'''

Order = namedtuple('Order', ['Type', 'Ticker', 'Quantity', 'Price'])
Order.__doc__ = '''
Orders to place for target trades.

type: string, value in ['market', 'limit_up', 'limit_down', 'target']
ticker: string, the ticker/name of securities to trade
quantiy: integer, positve or negative, share/quantiy to trade
price: float,  price information for 'limit_up', 'limit_down' or 'target' orders
'''

class market:
    '''
    The market class replicate information (mostly price) from the market. The main attributes are the universe, which contains stocks available in this market, and price, which contain price information of stocks in the universe.
    '''
    '''
    def __init__(self, adj_close_price=None, open_price=None, high_price=None, low_price=None, close_price=None, volume = None):
        self.adj_close = adj_close_price
        self.O = open_price
        self.H = high_price
        self.L = low_price
        self.C =close_price
        self.V = volume
    '''

    def __init__(self):
        self.universe = set()
        self.price = dict()
    
    def add_stock(self, ticker, price_data):
        '''
        ticker: string, identifier of the new stock
        price_data: (part of) open, high, low, close, adj_close, volume data of given stock. Use standard column names.
        '''
        self.universe.add(ticker)
        self.price[ticker] = price_data

    def remove_stock(self, ticker):
        '''
        ticker: string, identifier of the stock to remove
        '''
        self.universe.discard(ticker)
        del self.price[ticker]
    
    def get_price(self, ticker, start_date=None, end_date=None, period=0):
        '''
        ticker: string, identifier of stock to look up
        start_date: string/datetime, start of price period in 'yyyy-mm-dd'
        end_date: string/datetime, end of price period in 'yyyy-mm-dd'
        period: integer, length of period, alternative way to define period
        '''
        price_data = self.price[ticker]
        if start_date and end_date:
            price_data = price_data.loc[start_date:end_date, :]
        elif start_date and period:
            price_data = price_data.loc[start_date:, :]
            price_data = price_data.iloc[:period, :]
        elif end_date and period:
            price_data = price_data.loc[:end_date, :]
            price_data = price_data.iloc[-period:, :]
        
        return price_data

    def execute(self, order, trading_system, date):
        '''
        Execute a given order at given date, for trading_system.
        '''
        # Market order: (execute at market open)
        if order.Type == 'market':
            price_open = self.price[order.Ticker].loc[date, 'open']
            trading_system._update_transaction(
                date = date, 
                ticker = order.Ticker, 
                quantity = order.Quantity, 
                price = price_open
            )
            return True

        # Target order: (execute at target if in range low:high)
        elif order.Type == 'target':
            price_high = self.price[order.Ticker].loc[date, 'high']
            price_low = self.price[order.Ticker].loc[date, 'low']
            if (order.Price <= price_high) and (order.Price>= price_low):
                trading_system._update_transaction(
                    date=date, 
                    ticker=order.Ticker, 
                    quantity=order.Quantity, 
                    price=order.Price
                )
                return True
            else:
                return False

        # Limit up order: (execute at target or lower if in range low:high)
        elif order.Type == 'limit_up':
            price_open = self.price[order.Ticker].loc[date, 'open']
            price_high = self.price[order.Ticker].loc[date, 'high']
            price_low = self.price[order.Ticker].loc[date, 'low']
            if order.Price >= price_open:
                trading_system._update_transaction(
                    date=date, 
                    ticker=order.Ticker, 
                    quantity=order.Quantity, 
                    price=price_open
                )
                return True
            elif order.Price >= price_low:
                trading_system._update_transaction(
                    date=date, 
                    ticker=order.Ticker, 
                    quantity=order.Quantity, 
                    price=order.Price
                )
                return True
            else:
                return False

        # Limit down order: (execute at target or above if in range low:high)
        elif order.Type == 'limit_down':
            price_open = self.price[order.Ticker].loc[date, 'open']
            price_high = self.price[order.Ticker].loc[date, 'high']
            price_low = self.price[order.Ticker].loc[date, 'low']
            if order.Price <= price_open:
                trading_system._update_transaction(
                    date=date, 
                    ticker=order.Ticker, 
                    quantity=order.Quantity, 
                    price=price_open
                )
                return True
            elif order.Price <= price_high:
                trading_system._update_transaction(
                    date=date, 
                    ticker=order.Ticker, 
                    quantity=order.Quantity, 
                    price=order.Price
                )
                return True
            else:
                return False

        else:
            raise TypeError('Unknown Order Type!')

    def execute_orders(self, trading_system, date):
        '''
        Execute all orders in order_book of trading_system at date.
        '''
        execution_results = list()
        for order in trading_system.order_book:
            execution = self.execute(order, trading_system, date)
            execution_results.append(execution)
        trading_system._update_account(date)
        trading_system._update_order_book(execution_results)

    def evaluate(self, account):
        '''
        No dividend version. (Use close price)
        '''
        value = account.Holdings.get('Cash', 0)
        value += sum(account.Holdings[ticker]*self.price[ticker].loc[account.Date, 'close'] for ticker in account.Holdings if ticker!='Cash')
        return value

    '''
    @property
    def ret(self):
        # Lazy calculation of return from adjusted price.
        try:       
            return self._daily_ret
        except AttributeError:
            self._daily_ret = np.log(self.adj_close/self.adj_close.shift(1))
            return self._daily_ret

    @property
    def price_change(self):
        # Lazy calculation of price change from adjusted price.
        try:
            return self._price_change
        except AttributeError:
            self._price_change = self.adj_close.diff()
            return self._price_change
    
    def get_data(self, data_attr, date=None, period=None):
        all_data = getattr(self, data_attr)
        # If date is given, return data available on or before given date:
        if date is not None:
            if type(date) == str:
                date = pd.to_datetime(date)
            assert (date in all_data.index), f'Out of range date: {date:%Y-%m-%d}'  
            all_data = all_data.loc[:date, :]
        # If period is given, return data within the period:
        if period is not None:
            assert period>0, f'Negative period: {period}'
            assert period<=all_data.shape[0], f'period {period} out of range {all_data.shape[0]}'
            all_data = all_data.iloc[-period:, :]
        return all_data


    def daily_ret(self, date=None, lag=0):
        ret_df = self.get_data(data_attr='ret', date=date)
        assert lag>=0, f"Negative lag: {lag}"
        return ret_df.shift(lag).iloc[[-1], :]

    def total_ret(self, date=None, period=None):
        ret_df = self.get_data(data_attr='ret', date=date, period=period)
        return ret_df.sum(axis=0).to_frame(name=ret_df.index[-1]).T

    def volatility(self, date=None, period=None):
        ret_df = self.get_data(data_attr='ret', date=date, period=period)
        return ret_df.std(axis=0).to_frame(name=ret_df.index[-1]).T

    def bollinger(self, date=None, period=None):
        adj_close_df = self.get_data(data_attr='adj_close', date=date, period=period)
        z_score_df = (adj_close_df.iloc[-1, :]-adj_close_df.mean(axis=0))/adj_close_df.std(axis=0)
        return z_score_df.to_frame(name=adj_close_df.index[-1]).T

    def oscillator(self, date=None, period=None):
        adj_close_df = self.get_data(data_attr='adj_close', date=date, period=period)
        close_df = self.get_data(data_attr='C', date=date, period=period)
        adj_high_df = self.get_data(data_attr='H', date=date, period=period)*adj_close_df/close_df
        adj_low_df = self.get_data(data_attr='L', date=date, period=period)*adj_close_df/close_df
        osc_df = (adj_close_df.iloc[-1, :]-adj_low_df.min(axis=0))/(adj_high_df.max(axis=0)-adj_low_df.min(axis=0))
        return osc_df.to_frame(name=adj_close_df.index[-1]).T 

    def RSI(self, date=None, period=None):
        close_change_df = self.get_data(data_attr='price_change', date=date, period=period)
        up_move = close_change_df.clip(lower=0).sum(axis=0)
        down_move = close_change_df.clip(upper=0).sum(axis=0)
        total_move = abs(up_move)+abs(down_move)
        rsi_df = abs(up_move)/total_move
        rsi_df.loc[total_move<1e-10] = np.nan
        return rsi_df.to_frame(name=close_change_df.index[-1]).T
    '''

class trading_system:
    '''
    Trading system is a class of investment account
    '''
    def __init__(self):
        self.__account = Account(None, dict())
        self.__transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        self.__transaction = self.__transaction.astype({
            'Ticker': str,
            'Quantity': float
        })
        self.order_book = list()

    # Reset methods:
    def _reset_transaction(self):
        self.__transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        
    def _reset_account(self, date):
        self.__account = Account(None, dict())

    def _reset_order_book(self):
        self.order_book = list()

    # Private account and transaction: gettable but not settable.
    @property
    def account(self):
        return self.__account
    @property
    def transaction(self):
        return self.__transaction

####################       Methods to Manage Orders              ######################################
    def create_order(self, order):
        self.order_book.append(order)
        return self.order_book
    
    def cancel_order(self, order):
        self.order_book.remove(order)
        return self.order_book
    
    def clear_order(self):
        self.order_book = []
        return self.order_book
    
    def filter_order(self, condition):
        '''
        condition: a function of order, return True or False.
        '''
        self.order_book = [o for o in self.order_book if condition(o)]
        return self.order_book

####################     Update methodes for Order Execution       ###############################
    # Methods for execution orders:
    def _update_transaction(self, date, ticker, quantity, price):
        trading_amount = - quantity * price
        self.__transaction = self.__transaction.append(
            {'Date': date, 'Ticker': ticker, 'Quantity':quantity},
            ignore_index=True
        )
        self.__transaction = self.__transaction.append(
            {'Date': date, 'Ticker': 'Cash', 'Quantity': trading_amount},
            ignore_index=True
        )

    def _update_account(self, date=None):
        transaction_record = self.transaction
        if self.__account.Date is not None:
            transaction_record = transaction_record.loc[transaction_record.Date > self.__account.Date, :]
        if date is not None:
            transaction_record = transaction_record.loc[transaction_record.Date <= date]
        else:
            date = max(self.__account.Date, max(transaction_record.index))
        
        if transaction_record.shape[0]>0:
            for _, tran in transaction_record.iterrows():
                holdings = self.__account.Holdings
                ticker = tran['Ticker']
                qty = tran['Quantity']
                holdings[ticker] = holdings.get(ticker, 0) + qty

            self.__account = Account(date, holdings)
        else:
            self.__account = Account(date, self.__account.Holdings)
        return self.__account

    def _update_order_book(self, execution_results):
        assert len(self.order_book)==len(execution_results), 'Inconsistent execution results!'
        self.order_book = [o for (o, r) in zip(self.order_book, execution_results) if not r]
        return self.order_book


    
    

    
#%%%%%%%%%%%%%%%%









# %%
