import unittest
from pandas.util.testing import assert_frame_equal

import backtest_pkg as bt 
import pandas as pd 
import numpy as np
from math import sqrt, log, sin, pi

def cal_std(data):
    if len(data)<=1:
        return np.nan
    data_mean = sum(data)/len(data)
    data_var = sum((i-data_mean)**2 for i in data)/(len(data)-1)
    return sqrt(data_var)
def cal_mean(data):
    return sum(data)/len(data)

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        n = 10   # length of the period
        price_dict = {
            'Up trend': list(range(1, n+1)),
            'Down trend': list(range(n, 0, -1)),
            'Convex': list(1+(n/2)**2+ i*(i-n+1) for i in range(n)),
            'Concave': list(1+i*(n-1-i) for i in range(n)),
            'Sin': list(1+n*(1+sin(i/(n-1)*2*pi)) for i in range(n)),
        }
        adj_price_df = pd.DataFrame(price_dict, index=pd.date_range('2020-01-01', periods=n,freq='D'))
        self.ticker = adj_price_df.columns 
        self.index = adj_price_df.index
        self.price = adj_price_df

        self.trading_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        self.trading_status.iloc[:3, 0]=False
        self.trading_status.iloc[3:6, 1]=False
        self.trading_status.iloc[6:, 2]=False

        self.weight = pd.DataFrame(1, index=self.index[[0, 5]], columns=self.ticker)
        self.share = pd.DataFrame(1, index=self.index[[0, 5]], columns=self.ticker)
        ''' Price in values:
        'Up trend': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Down trend': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'Convex': [26, 18, 12, 8, 6, 6, 8, 12, 18, 26],
        'Concave': [1, 9, 15, 19, 21, 21, 19, 15, 9, 1],
        'Sin': [11, 17.42, 20.84, 19.66, 14.42, 7.57, 2.33, 1.15, 4.57, 11]
        '''

    def test_portfolio_set_price(self):
        # Normal setting:
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=self.price)
        expect_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, expect_status)

        # Setting at initiation:
        port = bt.portfolio(weight=self.weight, price=self.price)
        expect_status = pd.DataFrame(True, index=self.index, columns=self.ticker)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, expect_status)

        # Price and trading status cannot be set directly
        with self.assertRaises(AttributeError):
            port.price = self.price  
        with self.assertRaises(AttributeError):
            port.trading_status = self.trading_status  

        # Try masking out untradable prices:
        price = self.price.where(self.trading_status, other=np.nan)
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=price)
        assert_frame_equal(port.price, price)
        assert_frame_equal(port.trading_status, self.trading_status)
    def test_portfolio_set_price_and_trading_status(self):
        # Normal setting price and trading status:
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=self.price, trading_status=self.trading_status)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, self.trading_status)

        # Setting at initiation:
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status)
        assert_frame_equal(port.price, self.price)
        assert_frame_equal(port.trading_status, self.trading_status)

        # Independent NA prices and trading status:
        price = self.price.copy()
        price.iloc[:5, 4] = np.nan
        expect_status = self.trading_status.copy()
        expect_status.iloc[:5, 4]=False
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=price, trading_status=self.trading_status)
        assert_frame_equal(port.price, price)
        assert_frame_equal(port.trading_status, expect_status)

        # Out range trading status:
        out_range_status = self.trading_status.copy()
        out_range_status['Extra Ticker'] = True
        out_range_status.loc[pd.to_datetime('2020-01-20'), :]=False
        expect_status = self.trading_status
        port = bt.portfolio(weight=self.weight)
        port.set_price(price=self.price, trading_status=out_range_status)
        assert_frame_equal(port.trading_status, expect_status)

    def test_portfolio_weight(self):
        # Noraml equal weigt:
        port = bt.portfolio(weight=self.weight, price=self.price)
        expect = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weight, expect) 

        # Weights of row sum==zeros:
        weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[1, :] = 0
        port = bt.portfolio(weight=weight, price=self.price)
        expect = weight.iloc[[0], :]
        assert_frame_equal(port.weight, expect)
        weight = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[0, :] = 0
        port = bt.portfolio(weight=weight, price=self.price)
        expect = weight.iloc[[1], :]
        assert_frame_equal(port.weight, expect)

        # Out range weight:
        out_range_weight=self.weight.copy()
        out_range_weight['Extra Ticker']=1
        out_range_weight.loc[pd.to_datetime('2020-01-20'), :]=1
        port = bt.portfolio(weight=out_range_weight, price=self.price)
        expect = pd.DataFrame(0.2, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weight, expect) 
    def test_portfolio_weight_with_trading_status(self):
        # Weights on untradables:
        port = bt.portfolio(weight=self.weight, price=self.price, trading_status=self.trading_status)
        expect = pd.DataFrame(0.25, index=self.index[[0, 5]], columns=self.ticker)
        expect.iloc[0, 0]=0
        expect.iloc[1, 1]=0
        assert_frame_equal(port.weight, expect) 

        # Weights sum 0 from untradables:
        weight = pd.DataFrame(0.25, index=self.index[[0, 5]], columns=self.ticker)
        weight.iloc[0, 0]=0
        weight.iloc[1, :] = [0, 1, 0, 0, 0]
        port = bt.portfolio(weight=weight, price=self.price, trading_status=self.trading_status)
        expect = weight.iloc[[0], :]
        assert_frame_equal(port.weight, expect) 
        

    def test_portfolio_from_share(self):
        # No trading status:
        price_1 = [1, 10, 26, 1, 11]
        price_2 = [6, 5, 6, 21, 1+10*(1+sin(5/9*2*pi))]
        weight_value=[[i/sum(price_1) for i in price_1], [j/sum(price_2) for j in price_2]]
        expect = pd.DataFrame(weight_value, index=self.index[[0, 5]], columns=self.ticker)
        port = bt.portfolio(share=self.share, price=self.price)
        assert_frame_equal(port.weight,expect) 

        # With trading status:
        price_1 = [0, 10, 26, 1, 11]
        price_2 = [6, 0, 6, 21, 1+10*(1+sin(5/9*2*pi))]
        weight_value=[[i/sum(price_1) for i in price_1], [j/sum(price_2) for j in price_2]]
        port = bt.portfolio(share=self.share, price=self.price, trading_status=self.trading_status)
        expect = pd.DataFrame(weight_value, index=self.index[[0, 5]], columns=self.ticker)
        assert_frame_equal(port.weight,expect) 
    
    def test_portfolio_end_date(self):
        port = bt.portfolio(weight=self.weight, price=self.price)
        self.assertEqual(port.end_date, self.index[-1])
        end_date = pd.to_datetime('2020-01-08')
        port = bt.portfolio(weight=self.weight, price=self.price, end_date=end_date)
        self.assertEqual(port.end_date, end_date)
        
    def test_portfolio_daily_ret(self):
        price = pd.DataFrame(index=self.index)
        price['Normal'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Normal case
        price['Suspension'] = [1, 2, 3, np.nan, np.nan, na.nan, 4, 5, 6, 7]  # Temporary suspension
        price['Delisting'] = [1, 2, 3, 4, 5] + [np.nan]*5  # Delisting
        price['Late'] = [np.nan]*5 + [1, 2, 3, 4, 5]# Late listing:
        
        expect = pd.DataFrame(index=self.index)
        expect['Normal'] = [np.nan]+[log((i+1/i)) for i in range(1,10)]  
        expect['Suspension'] = [np.nan]+[log(2/1), log(3/2)] + [0.]*3 + [log(4/3), log(5/4), log(6/5), log(7/6)] 
        expect['Delisting'] =[np.nan]+[log((i+1/i)) for i in range(1,5)] + [0.]*5
        expect['Late'] = [np.nan]*6+[log((i+1/i)) for i in range(1,5)]

        port = bt.portfolio(weight=self.weight, price=price)
        assert_frame_equal(port.daily_ret, expect)


    def test_portfolio_drift_weight(self):
        # NO rebalance:

        # 1 rebalance:
        pass
    def test_portfolio_drift_weight_with_trading_status(self):
        # No rebalance
        # 1 rebalance:
        pass

    def test_portfolio_performance_return(self):
        pass

    def test_portfolio_performance_volatility(self):
        pass

    def test_portfolio_with_benchmark(self):
        pass 

    
    def test_portfolio_performance_active_return(self):
        pass

    def test_portfolio_performance_tracking_error(self):





#%%%%%%%%%%%%%%%%%%%


# %%
