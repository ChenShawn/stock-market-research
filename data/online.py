import tensorflow as tf
import pandas as pd
import numpy as np
import tushare as ts
import functools
import datetime
import json

try:
    import data.global_variables as G
except ModuleNotFoundError:
    import global_variables as G


def get_representation(df, colname):
    value_set = list(set(df[colname].values.tolist()))
    # value_set = list(map(lambda x: str(x).zfill(6), value_set))
    value_hash = dict(zip(value_set, list(range(len(value_set)))))
    return value_hash


def get_today(diff=0):
    today = datetime.datetime.now()
    month = str(today.month).zfill(2)
    day = str(today.day).zfill(2)
    return f'{today.year}-{month}-{day}'


def normalize_gaussian_data(colname, meanvars, x, epsilon=1e-9):
    numerator = x - meanvars[colname]['mean']
    denominator = meanvars[colname]['std'] + epsilon
    return numerator / denominator


def get_weekday_dict():
    weekday_list = ['Mon', 'Tue', 'Wedn', 'Thur', 'Fri']
    return dict(zip(weekday_list, list(range(len(weekday_list)))))


def get_weekday(daystr):
    weekday_list = ['Mon', 'Tue', 'Wedn', 'Thur', 'Fri', 'Sat', 'Sun']
    times = daystr.split('-')
    times = list(map(int, times))
    times = datetime.date(*times)
    return weekday_list[times.weekday()]


def filter_valid_stocks(stock_basics, codestr):
    codeline = stock_basics[stock_basics['codenum'] == codestr]
    codestr = str(codeline['timeToMarket'].values.tolist()[0])
    # assert len(codeline) == 1, f'{len(codeline)} duplicated codenum={codestr} exists'
    try:
        on_market_date = datetime.datetime(
            year=int(codestr[: 4]),
            month=int(codestr[4: 6]),
            day=int(codestr[6: ]),
        )
    except ValueError:
        # some csv has invalid on-market data, which can cause ValueError
        # stocks without on-market data should be identified as abnormal
        return False
    timediff = datetime.datetime.now() - on_market_date
    if timediff.days <= 90:
        return False
    return True


class OnlineGenerator(object):
    def __init__(self, start_date='20200101', lookback=14):
        self.lookback = lookback
        self.stock_basics = pd.read_csv('./data/stock_basics.csv')
        self.codenum_to_onehot = get_representation(self.stock_basics, 'codenum')
        self.weekday_to_onehot = get_weekday_dict()
        self.area_to_onehot = get_representation(self.stock_basics, 'area')
        self.industry_to_onehot = get_representation(self.stock_basics, 'industry')
        with open('./data/feature.json', 'r') as fd:
            self.feature_info = json.load(fd)
        
        valid_index = self.stock_basics['timeToMarket'].apply(lambda x: str(x) < start_date)
        self.valid_subdf = self.stock_basics[valid_index]
        assert len(self.valid_subdf) > 0, \
            f'Invalid start_date definition {start_date}'


    def __iter__(self):
        for codenum in self.valid_subdf['codenum']:
            codestr = str(codenum).zfill(6)
            codedf = self.download_nearest_data(codestr)
            if codedf is None:
                continue
            basic_num, basic_cat = self.merge_basic_stocks(codenum)
            yield (codedf[None, :], basic_num, basic_cat[None, :]), codestr
        return 0


    def download_nearest_data(self, codestr):
        # step 1: download data and check validity
        online_df = ts.get_hist_data(codestr, start='2020-03-01')
        if online_df is None:
            return None
        today = get_today()

        online_df['date'] = online_df.index.values
        online_df = online_df.sort_values(by='date')
        try:
            latest_date = online_df['date'][-1]
        except:
            return None
        if len(online_df) < 14 or latest_date != today:
            print(f'len(online_df)={len(online_df)} get_today={today} latest={latest_date}')
            return None
        online_df = online_df.iloc[-self.lookback: ]
        
        # step 2: feature processing
        online_df['weekday'] = online_df['date'].apply(get_weekday)
        online_df['madiff'] = online_df['ma10'] - online_df['ma20']
        online_df['madiff'] = online_df['ma10'] - online_df['ma20']
        madiff_mean = online_df['madiff'].rolling(9, center=True).mean()
        madiff_mean.iloc[: 4] = online_df['madiff'].iloc[: 4]
        madiff_mean.iloc[-4: ] = online_df['madiff'].iloc[-4: ]
        online_df['macd'] = 2.0 * (online_df['madiff'] - madiff_mean)
        
        # step 3: onehot encoding and standardize
        weekday = online_df['weekday'].apply(lambda x: self.weekday_to_onehot[x]).values.tolist()
        weekday_onehot = np.eye(5, dtype=np.float32)[weekday]
        for col in online_df.columns:
            if col in self.feature_info.keys():
                norm_fn = functools.partial(normalize_gaussian_data, col, self.feature_info)
                online_df[col] = online_df[col].apply(norm_fn)
        online_arr = online_df[G.STOCK_NUMERICALS].values.astype(np.float32)
        online_arr = np.concatenate([online_arr, weekday_onehot], axis=-1)
        return online_arr


    def merge_basic_stocks(self, codenum):
        basic_line = self.stock_basics[self.stock_basics['codenum'] == codenum]
        assert len(basic_line) == 1, \
            f'find {len(basic_line)} lines of {codenum} in stock_basics.csv'
        
        # numerical features
        for col in basic_line.columns:
            if col in self.feature_info.keys():
                norm_fn = functools.partial(normalize_gaussian_data, col, self.feature_info)
                basic_line[col] = basic_line[col].apply(norm_fn)
        basic_numerical = basic_line[G.BASIC_NUMERICALS].values.astype(np.float32)

        # categorical features
        industry = basic_line['industry'].apply(lambda x: self.industry_to_onehot[x])
        area = basic_line['area'].apply(lambda x: self.area_to_onehot[x])
        stockcode = basic_line['codenum'].apply(lambda x: self.codenum_to_onehot[x])
        basic_catogorical = np.concatenate([
            industry.values.astype(np.int32), 
            area.values.astype(np.int32), 
            stockcode.values.astype(np.int32)
        ], axis=-1)
        return basic_numerical, basic_catogorical


    def from_code_to_name(self, codenum, name='name'):
        basic_line = self.stock_basics[self.stock_basics['codenum'] == codenum]
        assert len(basic_line) == 1, \
            f'find {len(basic_line)} lines of {codenum} in stock_basics.csv'
        return basic_line[name].values.tolist()[0]

    


if __name__ == '__main__':
    generator = OnlineGenerator()
    xs, codestr = next(iter(generator))
    print(xs[0], xs[1], xs[2])