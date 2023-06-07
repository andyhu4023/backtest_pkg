import unittest
from pandas.testing import assert_frame_equal, assert_series_equal

import backtest_pkg as bt 
import pandas as pd 
import numpy as np

class TestTrading(unittest.TestCase):
    def setUp(self):
        def construct_price_data(data):
            index = pd.date_range('2020-01-01', periods=len(data), freq='D')
            price_data = pd.DataFrame(dict(
                open = data, 
                high = [i*1.2 for i in data],
                low =[i*0.8 for i in data], 
                close = data,
                adj_close = data,
            ), index = index)
            return price_data
        
        ticker1 = 'ticker1'
        data1 = [1., 3., 2., 4., 3., 5.]
        ticker2 = 'ticker2'
        data2 = [5., 3., 4., 2., 3., 1.]
        self.universe = {ticker1, ticker2}
        self.price1 = construct_price_data(data1)
        self.price2 = construct_price_data(data2)
        self.market = bt.market()
        self.market.add_stock(ticker1, self.price1)
        self.market.add_stock(ticker2, self.price2)
        self.trading_system = bt.trading_system()

    def test_set_up(self):
        # Check initial state of market:
        self.assertEqual(self.market.universe, self.universe)
        assert_frame_equal(self.market.price['ticker1'], self.price1)
        assert_frame_equal(self.market.price['ticker2'], self.price2)

        # Checking initial state of trading system:
        transaction_df = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction_df = transaction_df.astype({'Ticker': str, 'Quantity': float})
        assert_frame_equal(self.trading_system.transaction, transaction_df)
        self.assertEqual(self.trading_system.account, (None, dict()))
        self.assertEqual(self.trading_system.order_book, list())
        

    def test_market_buy_order(self):
        # Create market buy order:
        share, price = 1.0, 3.0
        order = bt.Order('market', 'ticker1', share, None)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)

        # Execute buy order at date '2020-01-02':
        date = pd.to_datetime('2020-01-02')
        self.market.execute_orders(self.trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['ticker1', 'Cash'],
            Quantity = [share, -share*price],
        ))
        account = bt.Account(date, {'ticker1':share, 'Cash':-share*price})
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_market_sell_order(self):
        # Create market sell order:
        share, price = 2., 4.
        order = bt.Order('market', 'ticker2', -share, None)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute a sell order at '2020-01-03'
        date = pd.to_datetime('2020-01-03')
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = ['ticker2', 'Cash'],
            Quantity = [-share, share*price],
        )) 
        account = bt.Account(date, {
            'ticker2':-share,
            'Cash':share*price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_target_buy_order(self):
        # Create target buy order:
        share, price = 1.0, 3.2
        order = bt.Order('target', 'ticker1', share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)

        # Execute at '2020-01-02' and succeed:
        date = pd.to_datetime('2020-01-02')
        self.market.execute_orders(self.trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['ticker1', 'Cash'],
            Quantity = [share, -share*price],
        ))
        account = bt.Account(date, {'ticker1':share, 'Cash':-share*price})
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_target_sell_order(self):
        # Create target sell order:
        share, price = 2., 4.3
        order = bt.Order('target', 'ticker2', -share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-03' and succeed:
        date = pd.to_datetime('2020-01-03')
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = ['ticker2', 'Cash'],
            Quantity = [-share, share*price],
        ))
        account = bt.Account(date, {
            'ticker2':-share,
            'Cash':share*price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_target_fail_order(self):
        order_book = []
        # Create target buy order:
        share, price = 3., 4.3
        order = bt.Order('target', 'ticker1', share, price)
        self.trading_system.create_order(order)
        order_book.append(order)
        # Execute at '2020-01-01' and fail:
        date = pd.to_datetime('2020-01-01')
        self.market.execute_orders(self.trading_system, date)
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = (date, dict())

        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

        # Create target sell order:
        share, price = 2., 1.5
        order = bt.Order('target', 'ticker2', -share, price)
        self.trading_system.create_order(order)
        order_book.append(order)
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-03' and fail:
        date = pd.to_datetime('2020-01-03')
        self.market.execute_orders(self.trading_system, date)
        account = (date, dict())
        
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_up_buy_open_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_up', 'ticker1', 2.0, 3.5
        date = pd.to_datetime('2020-01-02')
        execute_price = 3.0
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at open:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_up_sell_open_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_up', 'ticker1', -2.0, 3.5
        date = pd.to_datetime('2020-01-02')
        execute_price = 3.0
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at open:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_up_buy_intra_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_up', 'ticker1', 2.0, 2.9
        date = pd.to_datetime('2020-01-02')
        execute_price = 2.9
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at intraday target:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_up_sell_intra_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_up', 'ticker1', -2.0, 2.9
        date = pd.to_datetime('2020-01-02')
        execute_price = 2.9
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at intraday target:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_up_buy_fail_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_up', 'ticker1', 2.0, 2.1
        date = pd.to_datetime('2020-01-02')
        execute_price = None
        # Create limit up buy order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and fail:
        self.market.execute_orders(self.trading_system, date)
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = (date, dict())

        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_up_sell_fail_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_up', 'ticker1', -2.0, 2.1
        date = pd.to_datetime('2020-01-02')
        execute_price = None
        # Create limit up buy order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and fail:
        self.market.execute_orders(self.trading_system, date)
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = (date, dict())
        
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_down_buy_open_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_down', 'ticker2', 2.0, 2.5
        date = pd.to_datetime('2020-01-02')
        execute_price = 3.0
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at open:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_limit_down_sell_open_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_down', 'ticker2', -2.0, 2.5
        date = pd.to_datetime('2020-01-02')
        execute_price = 3.0
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at open:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)
        
    def test_limit_down_buy_intra_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_down', 'ticker2', 2.0, 3.5
        date = pd.to_datetime('2020-01-02')
        execute_price = 3.5
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at open:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)
        
    def test_limit_down_sell_intra_order(self):
        order_type, ticker, share, price = 'limit_down', 'ticker2', -2.0, 3.5
        date = pd.to_datetime('2020-01-02')
        execute_price = 3.5
        # Create limit up order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed at open:
        self.market.execute_orders(self.trading_system, date)
        transaction =pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = [ticker, 'Cash'],
            Quantity = [share, -share*execute_price],
        ))
        account = bt.Account(date, {
            ticker:share,
            'Cash':-share*execute_price,
        })
        order_book = list()
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)
        
    def test_limit_down_buy_fail_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_down', 'ticker1', 2.0, 4.1
        date = pd.to_datetime('2020-01-02')
        execute_price = None
        # Create limit up buy order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and fail:
        self.market.execute_orders(self.trading_system, date)
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = (date, dict())

        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)
        
    def test_limit_down_sell_fail_order(self):
        # Setting for test:
        order_type, ticker, share, price = 'limit_down', 'ticker1', -2.0, 4.1
        date = pd.to_datetime('2020-01-02')
        execute_price = None
        # Create limit up buy order:
        order = bt.Order(order_type, ticker, share, price)
        self.trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(self.trading_system.order_book, order_book)
        # Execute at '2020-01-02' and fail:
        self.market.execute_orders(self.trading_system, date)
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = (date, dict())

        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)

    def test_multi_orders(self):
        # Settings: (Use list to contain data)
        date_range = pd.date_range('2020-01-01', periods=3, freq='D')
        order_info = [
            ('market', 'ticker1', 3.0, None),
            ('target', 'ticker1', 2.0, 1.1),
            ('target', 'ticker2', 2.0, 0.2),
            ('limit_up', 'ticker1', 3.0, 2.0),
            ('limit_up', 'ticker2', -4.0, 1.8),
            ('limit_down', 'ticker2', 3.0, 4.2)
        ]
        execute_price = [1.0, 1.1, None, 2.0, None, 4.2]

        # Backtest on the period:
        for i in range(len(date_range)):
            date = date_range[i]
            self.trading_system.create_order(bt.Order(*order_info[i*2]))
            self.trading_system.create_order(bt.Order(*order_info[i*2+1]))
            self.market.execute_orders(self.trading_system, date)

        # Expected final result:
        transaction = pd.DataFrame(dict(
            Date = pd.to_datetime(['2020-01-01']*4+['2020-01-03']*4),
            Ticker = ['ticker1', 'Cash']*3+['ticker2', 'Cash'],
            Quantity = [3., -3., 2., -2.2, 3., -6., 3., -12.6], 
        ))
        account = bt.Account(date_range[-1], {
            'ticker1': 8.,
            'ticker2': 3.,
            'Cash':-23.8,
        })
        order_book = [bt.Order(*order_info[2]), bt.Order(*order_info[4])]
        
        assert_frame_equal(self.trading_system.transaction, transaction)
        self.assertEqual(self.trading_system.account, account)
        self.assertEqual(self.trading_system.order_book, order_book)
        
class TestDataUtil(unittest.TestCase):
    def setUp(self):
        def construct_price_data(data):
            index = pd.date_range('2020-01-01', periods=len(data), freq='D')
            price_data = pd.DataFrame(dict(
                open = data, 
                high = [i*1.2 for i in data],
                low =[i*0.8 for i in data], 
                close = data,
                adj_close = data,
            ), index = index)
            return price_data
        # Setting up:
        ticker1 = 'ticker1'
        data1 = [1., 3., 2., 4., 3., 5.]
        ticker2 = 'ticker2'
        data2 = [5., 3., 4., 2., 3., 1.]
        self.universe = {ticker1, ticker2}
        self.price1 = construct_price_data(data1)
        self.price2 = construct_price_data(data2)
        # Market initiation:
        self.market = bt.market()
        self.market.add_stock(ticker1, self.price1)
        self.market.add_stock(ticker2, self.price2)
        # Period data:
        self.start_str = '2020-01-02'
        self.end_str = '2020-01-04'
        self.period = 3
        self.expect_price_data = self.price1.loc[self.start_str:self.end_str, :]

    def test_price_whole_period(self):
        assert_frame_equal(self.market.get_price('ticker1'), self.price1)
        assert_frame_equal(self.market.get_price('ticker2'), self.price2)

    def test_price_start_end(self):
        start_date = pd.to_datetime(self.start_str)
        end_date = pd.to_datetime(self.end_str)

        # Both inputs date format:
        price_data = self.market.get_price('ticker1', start_date, end_date)
        assert_frame_equal(price_data, self.expect_price_data)
        # Both inputs str format:
        price_data = self.market.get_price('ticker1', self.start_str, self.end_str)
        assert_frame_equal(price_data, self.expect_price_data)
        # One date format and one str format:
        price_data = self.market.get_price('ticker1', start_date, self.end_str)
        assert_frame_equal(price_data, self.expect_price_data)

    def test_price_start_period(self):
        start_date = pd.to_datetime(self.start_str)

        # Date format:
        price_data = self.market.get_price('ticker1', start_date, period=self.period)
        assert_frame_equal(price_data, self.expect_price_data)
        # Str format:
        price_data = self.market.get_price('ticker1', self.start_str, period=self.period)
        assert_frame_equal(price_data, self.expect_price_data)

    def test_price_end_period(self):
        end_date= pd.to_datetime(self.end_str)

        # Date format:
        price_data = self.market.get_price('ticker1', end_date=end_date, period=self.period)
        assert_frame_equal(price_data, self.expect_price_data)
        # Str format:
        price_data = self.market.get_price('ticker1', end_date=self.end_str, period=self.period)
        assert_frame_equal(price_data, self.expect_price_data)
        












