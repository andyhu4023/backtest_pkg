import unittest
from pandas.util.testing import assert_frame_equal, assert_series_equal

import backtest_pkg as bt 
import pandas as pd 
import numpy as np

class TestTrading(unittest.TestCase):
    def setUp(self):
        def construct_market(data):
            ticker = ['Test Ticker']
            index = pd.date_range('2020-01-01', periods=len(data), freq='D')
            data_dict = dict(
                adj_close_price = pd.DataFrame(data, index=index, columns=ticker),
                open_price = pd.DataFrame(data, index=index, columns=ticker),
                high_price = pd.DataFrame([i*1.2 for i in data], index=index, columns=ticker),
                low_price = pd.DataFrame([i*0.8 for i in data], index=index, columns=ticker),
                close_price = pd.DataFrame(data, index=index, columns=ticker),
            )
            return data_dict
        data = [1, 3, 2, 4, 3, 5]
        data_dict = construct_market(data)
        self.index = data_dict['adj_close_price'].index
        self.ticker= data_dict['adj_close_price'].columns
        self.market = bt.market(**data_dict)

    def test_initiation(self):
        trading_system = bt.trading_system()
        transaction_df = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction_df = transaction_df.astype({'Ticker': str, 'Quantity': float})
        assert_frame_equal(trading_system.transaction, transaction_df)
        self.assertEqual(trading_system.account, (None, dict()))
        self.assertEqual(trading_system.order_book, list())
        

    def test_market_order(self):
        # Initiate:
        trading_system = bt.trading_system()

        buy_share, price = 1.0, 3.0
        # Create market buy order:
        order = bt.Order('market', 'Test Ticker', buy_share, None)
        trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(trading_system.order_book, order_book)

        # Execute buy order at date '2020-01-02':
        date = pd.to_datetime('2020-01-02')
        self.market.execute_orders(trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [buy_share, -buy_share*price],
        ))
        account = bt.Account(date, {'Test Ticker':buy_share, 'Cash':-buy_share*price})
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        sell_share, sell_price = 2., 4.
        # Create market sell order:
        order = bt.Order('market', 'Test Ticker', -sell_share, None)
        trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(trading_system.order_book, order_book)
        # Execute a sell order at '2020-01-04'
        date = pd.to_datetime('2020-01-04')
        self.market.execute_orders(trading_system, date)
        transaction = transaction.append(pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [-sell_share, sell_share*sell_price],
        )), ignore_index=True)
        account = bt.Account(date, {
            'Test Ticker':buy_share-sell_share,
            'Cash':sell_share*sell_price-buy_share*price,
        })
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)


    def test_target_order(self):
        # Initiate:
        trading_system = bt.trading_system()

        buy_share, buy_price = 1.0, 3.1
        # Create target buy order:
        order = bt.Order('target', 'Test Ticker', buy_share, buy_price)
        trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(trading_system.order_book, order_book)

        # Execute at '2020-01-01' and fail:
        date = pd.to_datetime('2020-01-01')
        self.market.execute_orders(trading_system, date)
        self.assertEqual(trading_system.order_book, order_book)
        # Execute at '2020-01-02' and succeed:
        date = pd.to_datetime('2020-01-02')
        self.market.execute_orders(trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [buy_share, -buy_share*buy_price],
        ))
        account = bt.Account(date, {'Test Ticker':buy_share, 'Cash':-buy_share*buy_price})
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        sell_share, sell_price = 2., 4.1
        # Create target sell order:
        order = bt.Order('target', 'Test Ticker', -sell_share, sell_price)
        trading_system.create_order(order)
        order_book = [order]
        self.assertEqual(trading_system.order_book, order_book)
        # Execute at '2020-01-03' and fail:
        date = pd.to_datetime('2020-01-03')
        self.market.execute_orders(trading_system, date)
        self.assertEqual(trading_system.order_book, order_book)
        # Execute at '2020-01-04' and succeed:
        date = pd.to_datetime('2020-01-04')
        self.market.execute_orders(trading_system, date)
        transaction = transaction.append(pd.DataFrame(dict(
            Date = [date]*2,
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [-sell_share, sell_share*sell_price],
        )), ignore_index=True)
        account = bt.Account(date, {
            'Test Ticker':buy_share-sell_share,
            'Cash':sell_share*sell_price-buy_share*buy_price,
        })
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

    def test_limit_order(self):
        #### Test fail limit buy:
        date = pd.to_datetime('2020-01-01')
        buy_share, buy_price = 1., 0.5
        trading_system = bt.trading_system()
        order = bt.Order('limit', 'Test Ticker', buy_share, buy_price)
        trading_system.create_order(order)
        # Before execution:
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)
        # After execution:
        self.market.execute_orders(trading_system, date)
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        #### Test open limit buy:
        date = pd.to_datetime('2020-01-01')
        buy_share, buy_price, execute_price = 1., 2.2, 1.
        trading_system = bt.trading_system()
        order = bt.Order('limit', 'Test Ticker', buy_share, buy_price)
        trading_system.create_order(order)
        # Before execution:
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)
        # After execution:
        self.market.execute_orders(trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [buy_share, -buy_share*execute_price],
        ))
        account = bt.Account(date, {'Test Ticker':buy_share, 'Cash':-buy_share*execute_price})
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        #### Test intraday limit buy:
        date = pd.to_datetime('2020-01-01')
        buy_share, buy_price, execute_price = 1., 0.9, 0.9
        trading_system = bt.trading_system()
        order = bt.Order('limit', 'Test Ticker', buy_share, buy_price)
        trading_system.create_order(order)
        # Before execution:
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)
        # After execution:
        self.market.execute_orders(trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [buy_share, -buy_share*execute_price],
        ))
        account = bt.Account(date, {'Test Ticker':buy_share, 'Cash':-buy_share*execute_price})
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        #### Test fail limit sell:
        date = pd.to_datetime('2020-01-01')
        sell_share, sell_price = 2., 2.
        trading_system = bt.trading_system()
        order = bt.Order('limit', 'Test Ticker', -sell_share, sell_price)
        trading_system.create_order(order)
        # Before execution:
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)
        # After execution:
        self.market.execute_orders(trading_system, date)
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        #### Test open limit sell:
        date = pd.to_datetime('2020-01-01')
        sell_share, sell_price, execute_price = 1., 0.5, 1.
        trading_system = bt.trading_system()
        order = bt.Order('limit', 'Test Ticker', -sell_share, sell_price)
        trading_system.create_order(order)
        # Before execution:
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)
        # After execution:
        self.market.execute_orders(trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [-sell_share, sell_share*execute_price],
        ))
        account = bt.Account(date, {'Test Ticker':-sell_share, 'Cash':sell_share*execute_price})
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        #### Test intraday limit sell:
        date = pd.to_datetime('2020-01-01')
        sell_share, sell_price, execute_price = 1., 1.1, 1.1
        trading_system = bt.trading_system()
        order = bt.Order('limit', 'Test Ticker', -sell_share, sell_price)
        trading_system.create_order(order)
        # Before execution:
        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)
        # After execution:
        self.market.execute_orders(trading_system, date)
        transaction = pd.DataFrame(dict(
            Date = [date, date],
            Ticker = ['Test Ticker', 'Cash'],
            Quantity = [-sell_share, sell_share*execute_price],
        ))
        account = bt.Account(date, {'Test Ticker':-sell_share, 'Cash':sell_share*execute_price})
        order_book = list()
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

    def test_multi_orders(self):
        # Initiate and settings:
        trading_system = bt.trading_system()
        buy_share = 1.0
        target_price, limit_price = 2.1, 2.1
        expect_transaction = pd.DataFrame(dict(
            Date = pd.to_datetime(['2020-01-02']*2+['2020-01-03']*4),
            Ticker = ['Test Ticker', 'Cash']*3,
            Quantity = [buy_share, -buy_share*3., buy_share, -buy_share*target_price, buy_share, -buy_share*2.]
        ))

        # Create buy orders of three kinds:
        market_order = bt.Order('market', 'Test Ticker', buy_share, None)
        target_order = bt.Order('target', 'Test Ticker', buy_share, target_price)
        limit_order = bt.Order('limit', 'Test Ticker', buy_share, limit_price)
        trading_system.create_order(market_order)
        trading_system.create_order(target_order)
        trading_system.create_order(limit_order)

        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [market_order, target_order, limit_order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        # Execute at '2020-01-02':
        date = pd.to_datetime('2020-01-02')
        self.market.execute_orders(trading_system, date)
        account = bt.Account(date, {'Test Ticker':buy_share, 'Cash':-buy_share*3.})
        assert_frame_equal(trading_system.transaction, expect_transaction.iloc[:2, :])
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book[1:])
        # Execute at '2020-01-03':
        date = pd.to_datetime('2020-01-03')
        self.market.execute_orders(trading_system, date)
        account = bt.Account(date, {'Test Ticker':buy_share*3., 'Cash':-buy_share*(3.+2.1+2.)})
        assert_frame_equal(trading_system.transaction, expect_transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, list())

        # Initiate and settings:
        trading_system = bt.trading_system()
        sell_share = -1.0
        target_price, limit_price = 2.9, 2.9
        expect_transaction = pd.DataFrame(dict(
            Date = pd.to_datetime(['2020-01-01']*2+['2020-01-02']*4),
            Ticker = ['Test Ticker', 'Cash']*3,
            Quantity = [sell_share, -sell_share*1., sell_share, -sell_share*target_price, sell_share, -sell_share*3.]
        ))

        # Create sell orders of three kinds:
        market_order = bt.Order('market', 'Test Ticker', sell_share, None)
        target_order = bt.Order('target', 'Test Ticker', sell_share, target_price)
        limit_order = bt.Order('limit', 'Test Ticker', sell_share, limit_price)
        trading_system.create_order(market_order)
        trading_system.create_order(target_order)
        trading_system.create_order(limit_order)

        transaction = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        transaction = transaction.astype({'Ticker': str, 'Quantity': float})
        account = bt.Account(None, dict())
        order_book = [market_order, target_order, limit_order]
        assert_frame_equal(trading_system.transaction, transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book)

        # Execute at '2020-01-01':
        date = pd.to_datetime('2020-01-01')
        self.market.execute_orders(trading_system, date)
        account = bt.Account(date, {'Test Ticker':sell_share, 'Cash':-sell_share*1.})
        assert_frame_equal(trading_system.transaction, expect_transaction.iloc[:2, :])
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, order_book[1:])
        # Execute at '2020-01-02':
        date = pd.to_datetime('2020-01-02')
        self.market.execute_orders(trading_system, date)
        account = bt.Account(date, {'Test Ticker':sell_share*3., 'Cash':-sell_share*(1.+2.9+3.)})
        assert_frame_equal(trading_system.transaction, expect_transaction)
        self.assertEqual(trading_system.account, account)
        self.assertEqual(trading_system.order_book, list())













#%%%%%%%%%%%%%%%

# %%
