import pandas as pd 

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




