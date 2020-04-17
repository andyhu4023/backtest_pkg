import pandas as pd 
import numpy as np 

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
        

class trading_system:

    def __init__(self, capital=10**6):
        self.capital = 10**6
    
    def set_price(self, OHLC):
        OHLC.columns = list('OHLC')
        OHLC.index = pd.to_datetime(OHLC.index)
        self.__price = OHLC

 ###################   Create Orders   ###############################
    def _init_order_book(self):
        '''
        Initiate the order book to contain order information for execution.
        market orders: record to number of shares to buy/sell in market
        limit orders: a pd.Series with key string format of limit price, value integer format of buy/sell shares.
        target orders: same as limit order, different execution rules.
        '''
        self.market_buy_book = 0
        self.market_sell_book = 0
        self.limit_buy_book = pd.Series()
        self.limit_sell_book = pd.Series()
        self.target_buy_book = pd.Series()
        self.target_sell_book = pd.Series()

    def market_buy(self, quantity):
        '''
        quatity: number of shares to buy at market price.
        '''
        self.market_buy_book += quantity
    
    def market_sell(self, quantity):
        '''
        quantiy: number of shares to sell at market price.
        '''
        self.market_sell_book += quantity

    def limit_buy(self, quantity, price):
        '''
        quantity: number of shares to buy.
        price: highest price to buy
        '''
        self.limit_buy_book[str(price)] = self.limit_buy_book.get(str(price), 0) + quantity

    def limit_sell(self, quantity, price):
        '''
        quantity: number of shares to sell.
        price: lowest price to sell.
        '''
        self.limit_sell_book[str(price)] = self.limit_sell_book.get(str(price), 0) + quantity

    def target_buy(self, quantity, price):
        '''
        quantity: number of shares to buy.
        price: exact price to buy. (No transaction for higher or lower price)
        '''
        self.target_buy_book[str(price)] = self.target_buy_book.get(str(price), 0) + quantity

    def target_sell(self, quantity, price):
        '''
        quantity: number of shares to sell.
        price: exact price to sell. (No transaction for lower or higher price)
        '''
        self.target_sell_book[str(price)] = self.target_buy_book.get(str(price), 0) + quantity

    

#####################   Execution    #########################
    def _init_account(self):
        '''
        On attribute 'account', use a dictionary to store holding information. The dictionary has keys ['cash', 'shares']
        '''
        self.account = {'cash': self.capital, 'share': 0}
        self.transaction = pd.DataFrame(columns=['Date', 'Side', 'Quantity', 'Price'])

    def _load_trading_data(self, date):
        '''
        On attribute 'current_trading_data', use a pd.Series to store OHLC price on given date.
        '''
        date = pd.to_datetime(date)
        self.current_trading_data = self.__price[date, :]
        self.current_trading_date = date

    # to do: handling shorting money/shares? how to deal with remaining orders, return?
    def  transaction(self, date, side, qty, price):
        if side == 'B':
            self.account['cash'] -= qty*price
            self.account['share'] += qty 
            self.transaction.append({'Date':date, 'Side': 'B', 'Quantity':qty, 'Price': price}, ignore_index = True)
        elif side  == 'S':
            self.account['cash'] += qty*price
            self.account['share'] -= qty 
            self.transaction.append({'Date':date, 'Side': 'S', 'Quantity':qty, 'Price': price}, ignore_index = True)
        else:
            raise 'Unknown Side'


    ####################     Market orders     #######################
    def execute_market_orders(self):
        open_price = self.current_trading_data['O']
        date = self.current_trading_date

        # Market trade are execute at open price:
        market_shares = self.market_buy_book -  self.market_sell_book
        if market_shares == 0:
            return
        elif market_shares>0:
            if self.account['cash'] >= market_order_value:
                self.transaction(date, 'B', market_shares, open_price)
                self.market_buy_book = 0
                self.market_sell_book = 0
                return
            else:
                # If not enough capital, then buy as much as possible, non-executed shares will remain:
                trading_share = int(capital/open_price)
                self.transaction(date, 'B', trading_share, open_price)
                self.market_buy_book = market_shares - trading_share
                self.market_sell_book = 0
                return
        else:
            if self.account['share'] >= -market_shares:
                self.transaction(date, 'S', -market_shares, open_price)
                self.market_buy_book = 0
                self.market_sell_book = 0
                return
            else:
                # If not enough shares to sell, then sell all, non-executed shares will remain:
                trading_share = self.account['share']
                self.transaction(date, 'S', trading_share, open_price)
                self.market_buy_book = 0
                self.market_sell_book = -market_shares - trading_share 
                return

    ####################     Target Orders    ####################
    def execute_target_orders(self):
        low_price = self.current_trading_data['L']    
        high_price = self.current_trading_data['H']
        date = self.current_trading_date

        # Target buy:
        numerical_index = pd.to_numeric(self.target_buy_book.index)
        execute_index = (numerical_index <= high_price) & (numerical_index >= low_price)
        for price in numerical_index[execute_index]:
            self.transaction(date, 'B', self.target_buy_book[str(price)], price)
        self.target_buy_book = self.target_buy_book[~execute_index] # Update book

        # Target sell:
        numerical_index = pd.to_numeric(self.target_sell_book.index)
        execute_index = (numerical_index <= high_price) & (numerical_index >= low_price)
        for price in numerical_index[execute_index]:
            self.transaction(date, 'S', self.target_sell_book[str(price)], price)
        self.target_buy_book = self.target_sell_book[~execute_index] # Update book
        
    ####################     Limit Orders   ####################
    def execute_limit_orders(self):
        open_price = self.current_trading_data['O']
        high_price = self.current_trading_data['H']
        low_price = self.current_trading_data['L']
        date = self.current_trading_date

        # open buy:
        numerical_index = pd.to_numeric(self.limit_buy_book.index)
        execute_index = (numerical_index >= open_price)
        for price in numerical_index[execute_index]:
            self.transaction(date, 'B', self.limit_buy_book[str(price)], open_price)
        self.limit_buy_book = self.limit_buy_book[~execute_index] # Update book

        # intraday buy:
        numerical_index = pd.to_numeric(self.limit_buy_book.index)
        execute_index = (numerical_index >= low_price)
        for price in numerical_index[execute_index]:
            self.transaction(date, 'B', self.limit_buy_book[str(price)], price)
        self.limit_buy_book = self.limit_buy_book[~execute_index] # Update book

        # open sell:
        numerical_index = pd.to_numeric(self.limit_sell_book.index)
        execute_index = (numerical_index <= open_price)
        for price in numerical_index[execute_index]:
            self.transaction(date, 'S', self.limit_sell_book[str(price)], open_price)
        self.limit_sell_book = self.limit_sell_book[~execute_index] # Update book

        # intraday sell:
        numerical_index = pd.to_numeric(self.limit_sell_book.index)
        execute_index = (numerical_index <= high_price)
        for price in numerical_index[execute_index]:
            self.transaction(date, 'S', self.limit_sell_book[str(price)], price)
        self.limit_sell_book = self.limit_sell_book[~execute_index] # Update book












class universe:
    def __init__(self, OHLC = dict()):
        '''
        OHLC is a dictionary with key as identity of security (Ticker, SEDOL, ISIN or any other id) and value as data frame with columns ['O', 'H', 'L', 'C'] and rows dates of corresponding data.
        '''
        self.OHLC = OHLC
    
    def append(self, id, OHLC_df):
        self.OHLC[id] = OHLC_df
    
    def merge(self, OHLC_dict):
        self.OHLC = {**self.OHLC, **OHLC_dict}

class orders:
    '''
    Each order should has id (Ticker/SEDOL or anything recognized in the universe), side (B/S), quantity, type (Market/Limit/Target) and price. It should be a pd.Series with corresponding indexes.
    Order record is a data frame of columns ['id', 'side', 'quantity', 'type', 'price']
    '''
    def __init__(self, order_record=None):
        if order_record:
            self.record = order_record
        else:
            self.record = pd.DataFrame(columns=['id', 'side', 'quantity', 'type', 'price'])

    def append(self, order):
        '''
        order is a pd.Series with index ['id', 'side', 'quantity', 'type', 'price'].
        '''
        self.record = self.record.append(order, ignore_index = True)
    
    def merge(self, new_order_record):
        '''
        new_order_record is pd.DataFrame with columns ['id', 'side', 'quantity', 'type', 'price'].
        '''
        self.record = pd.concat([self.record, new_order_record], axis=1, join='inner')

    def delete(self, order=None, order_record=None):
        pass


class account:
    '''
    Account contains information of holding. It is a data frame of columns ['quantity', 'price', 'value'], index as id of security or 'cash'.
    Transactions will record quantity change in account.
    '''
    def __init__(self, capital = 10**6):
        self.holding = pd.DataFrame(columns=['quantity', 'price', 'value'])
        self.holding.loc['Cash', :] = [capital, 1, capital]
    


class signal:

    '''
    signal is a model for evaluating features like lagged OHLC data, fundamental data, related security information, macro data (like GDP growth, interest rate change, FX movement) to output trading recomendation. 
    The API here is much like sk-learn models if it needs to be updated or trained with new data. The model can also be hard-code rules for certain logic. The output can be a direct trade label (Buy, Sell, Hold or numerical values 1, 0, -1) or a rating for each trade label, or a probability distribution of each label.
    '''
    def __init__(self):
        pass




