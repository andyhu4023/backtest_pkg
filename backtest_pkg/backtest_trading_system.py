import pandas as pd 
import numpy as np 
from collections import namedtuple

Account = namedtuple('Account', ['Date', 'Holdings'])
# date: datetime object, the version/standing point of account information
# holdings: a dict with tickers as keys and holding shares/quantity as values.
Order = namedtuple('Order', ['Type', 'Ticker', 'Quantity', 'Price'])
# type: string in ['market', 'limit', 'target']
# ticker: string, the ticker/name of securities to trade
# quantiy: integer, positve or negative, share/quantiy to trade
# price: float,  price information for 'limit' or 'target' orders

class market:
    def __init__(self, adj_close_price=None, open_price=None, high_price=None, low_price=None, close_price=None, volume = None):
        self.adj_close = adj_close_price
        self.O = open_price
        self.H = high_price
        self.L = low_price
        self.C =close_price
        self.V = volume

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
        
    def execute(self, order, trading_system, date):
        if order.Type == 'market':
            price_open = self.O.loc[date, order.Ticker]
            trading_system.add_transaction(date = date, ticker = order.Ticker, quantity = order.Quantity, price = price_open)
            return True

        elif order.Type == 'limit':
            if order.Quantity == 0:
                return True
            elif order.Quantiy >0:
                price_open = self.O.loc[date, order.Ticker]
                price_low = self.L.loc[date, order.Ticker]
                if order.Price > price_open:
                    trading_system.add_transaction(date=date, ticker=order.Ticker, quantity=order.Quantity, price=price_open)
                    return True
                elif order.Price > price_low:
                    trading_system.add_transaction(date=date, ticker=order.Ticker, quantity=order.Quantity, price=order.Price)
                    return True
                else:
                    return False
            
            elif order.Quantity <0:
                price_open = self.O.loc[date, order.Ticker]
                price_high = self.H.loc[date, order.Ticker]
                if order.Price < price_open:
                    trading_system.add_transaction(date=date, ticker=order.Ticker, quantity=order.Quantity, price=price_open)
                    return True
                    trading_system.add(date)
                elif order.Price<price_high:
                    trading_system.add_transaction(date=date, ticker=order.Ticker, quantity=order.Quantity, price=order.Price)
                    return True
                else:
                    return False

        elif order.Type == 'target':
            price_high = self.H.loc[date, order.Ticker]
            price_low = self.L.loc[date, order.Ticker]
            if (order.Price <= price_high) and (order.Price>= price_low):
                trading_system.add_transaction(date=date, ticker=order.Ticker, quantity=order.Quantity, price=order.Price)
                return True
            else:
                return False
            
        else:
            raise TypeError('Unknown Order Type!')

    def execute_orders(self, trading_system, date):
        execution_results = list()
        for order in trading_system.order_book:
            execution_results.append(self.execute(order, trading_system, date))
        trading_system._update_account(date)
        trading_system._update_order_book(execution_results)

    def evaluate(self, account):
        price = self.C.loc[account.Date]
        price['Cash'] = 1
        value = sum(account.Holdings[ticker]*price[ticker] for ticker in account.Holdings)
        return value

class trading_system:
    def __init__(self):
        self._account = Account(None, dict())
        self.transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
    
    def add_transaction(self, date, ticker, quantity, price):
        trading_amount = - quantity * price
        self.transaction.append(
            {'Date': date, 'Ticker': ticker, 'Quantity':quantity},
            ignore_index=True
        )
        trading_system.transaction.append(
            {'Date': date, 'Ticker': 'Cash', 'Quantity': trading_amount},
            ignore_index=True
        )

    def _update_account(self, date):
        transaction_record = self.transaction.copy()
        transaction_record = transaction_record.loc[(transaction_record.Date <= date) & (transaction_record.Date > self._account.Date), :]
        for _, tran in transaction_record.iterrows():
            holdings = self._account.Holdings
            ticker = tran['Ticker']
            qty = tran['Quantity']
            holdings[ticker] = holdings.get(ticker, 0) + qty
            self._account.Holdings = holdings
        self._account.Date = date
        return self._account

    def _reset_account(self, date):
        self._account = Account(date, dict())

    def create_order(self, order):
        try:
            self.order_book.append(order)
        except AttributeError:
            self.order_book = list()
            self.order_book.append(order)
        return self.order_book
    
    def cancel_order(self, order):
        try:
            self.order_book.remove(order)
        except AttributeError:
            self.order_book = list()
        return self.order_book
    
    def reset_order_book(self):
        self.order_book = list()
    
    def _update_order_book(self, execution_results):
        assert len(self.order_book)==len(execution_results), 'Inconsistent execution results!'
        self.order_book = [o for (o, r) in zip(self.order_book, execution_results) if not r]

        
    def holding(self, date):
        transaction_record = self.transaction.copy()
        transaction_record = transaction_record.loc[transaction_record.Date <= date, :]
        holdings = dict()
        for _, tran in transaction_record.iterrows():
            ticker = tran['Ticker']
            qty = tran['Quantity']
            holdings[ticker] = holdings.get(ticker, 0) + qty
        return holdings


    








