import tushare as ts
import pandas as pd
import numpy as np
import datetime
import functools
import tqdm
import json
import math
import os


def maybe_download_stock_list(basedir='./stocks/'):
    """maybe_download_stock_list
    return: type pd.DataFrame
               code,代码
               name,名称
               industry,细分行业
               area,地区
               pe,市盈率
               outstanding,流通股本
               totals,总股本(万)
               totalAssets,总资产(万)
               liquidAssets,流动资产
               fixedAssets,固定资产
               esp,每股收益
               bvps,每股净资
               pb,市净率
               timeToMarket,上市日期
    """
    def get_time_diff(timestr):
        timestr = str(timestr)
        try:
            on_market_date = datetime.datetime(
                year=int(timestr[: 4]),
                month=int(timestr[4: 6]),
                day=int(timestr[6: ]),
            )
        except ValueError:
            return -1.0
        now = datetime.datetime.now()
        timediff = now - on_market_date
        return float(timediff.days)
    
    # download stock basics
    all_path = os.path.join(basedir, 'stock_basics.csv')
    basic_df = ts.get_stock_basics()
    basic_df['codenum'] = basic_df.index.values
    basic_df['exist_time'] = basic_df['timeToMarket'].apply(get_time_diff)
    basic_df.to_csv(os.path.join(basedir, 'stock_basics.csv'))
    return basic_df


def check_and_download_latest_stock_data(codes, basedir='./stocks/', start='2008-01-01', 
                                         filtering_func=lambda x: x is not None,
                                         feature_prepropcessing_func=lambda x: x):
    """check_and_download_latest_stock_data
    ts.get_hist_data: return type pd.DataFrame
    属性: 日期，开盘价，最高价，收盘价，最低价，成交量，价格变动，涨跌幅，
        5日均价，10日均价，20日均价，5日均量，10日均量，20日均量，(换手率)
    """
    print(f' [*] Start downloading {len(codes)} stocks')
    num_download, num_failed = 0, 0
    valid_codes = list(filter(filtering_func, codes))
    for cod in tqdm.tqdm(valid_codes):
        csv_path = os.path.join(basedir, str(cod) + '.csv')
        if not os.path.exists(csv_path):
            stock_df = ts.get_hist_data(str(cod), start=start)
            if stock_df is not None:
                stock_df = feature_prepropcessing_func(stock_df)
                stock_df.to_csv(csv_path, index=False)
                num_download += 1
            else:
                num_failed += 1
    print(f' [*] Download {num_download} stock files as csv with {num_failed} failed cases')
    return num_download, num_failed


def filter_valid_stocks(stock_basics, codestr):
    """filter_valid_stocks
    Rule 1: don't use stocks launched to market in nearest 3 months
    Rule 2: not in the blacklist investigated by government
    NOTE: rule 1 will filter out about 100 among 3800+ stocks
    """
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


def feature_prepropcessing(df):
    """feature_prepropcessing
    1. add weekday as one column of feature
    2. add macd as features
    3. generate label column
    """
    weekday_list = ['Mon', 'Tue', 'Wedn', 'Thur', 'Fri', 'Sat', 'Sun']
    def get_weekday(daystr):
        times = daystr.split('-')
        times = list(map(int, times))
        times = datetime.date(*times)
        return weekday_list[times.weekday()]

    def get_policy_label(data):
        if data[0] < 0 and data[1] > 0:
            # negative MACD turning to positive: signal of buy in
            return -0.5
        elif data[0] > 0 and data[1] < 0:
            # positive MACD turning to negative: signal of sell out
            return 0.5
        else:
            return 0.0

    def gen_labels(buy_sell_list, period=30, tax=0.03):
        labels = []
        for idx, price in enumerate(buy_sell_list):
            buy, sell = 0.0, 0.0
            subdf = buy_sell_list[idx + 1: idx + period]
            for sub in subdf:
                if sub < 0 and buy == 0:
                    buy = sub
                elif sub > 0 and buy != 0:
                    sell = sub
                    break
            profit = buy + sell - tax
            if profit > 0:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    df['date'] = df.index.values
    df['weekday'] = df['date'].apply(get_weekday)
    # use 10 and 20 days ma curve to compute madiff
    # for convinience use uniform smoothing instead of ema
    df['madiff'] = df['ma10'] - df['ma20']
    madiff_mean = df['madiff'].rolling(9, center=True).mean()
    madiff_mean.iloc[: 4] = df['madiff'].iloc[: 4]
    madiff_mean.iloc[-4: ] = df['madiff'].iloc[-4: ]
    df_macd = 2.0 * (df['madiff'] - madiff_mean)
    df['macd'] = df_macd
    # label=1 if profit is positive when using MACD policy in the next trial
    buy_sell_ratio = df_macd.rolling(2).apply(get_policy_label)
    buy_sell_ratio[0] = 0.0
    # df['buy_sell_ratio'] = buy_sell_ratio
    buy_sell_price = buy_sell_ratio * (df['high'] + df['low'])
    # df['buy_sell_price'] = buy_sell_price
    df['label'] = gen_labels(buy_sell_price.values.tolist())
    return df.sort_values('date')


def download_financial_report(start_year=2008, basedir='./profit/'):
    """download_financial_report
    return type pd.DataFrame
        code,代码
        name,名称
        roe,净资产收益率(%)
        net_profit_ratio,净利率(%)
        gross_profit_rate,毛利率(%)
        net_profits,净利润(万元)
        eps,每股收益
        business_income,营业收入(百万元)
        bips,每股主营业务收入(元)
    """
    uptodate = 2019
    for year in range(start_year, uptodate + 1):
        for season in range(1, 5):
            profit = ts.get_profit_data(year, season)
            profit.to_csv(os.path.join(basedir, f'profit_{year}_{season}.csv'))
    print(' [*] OK')


def get_stats(basedir='./stocks/', other_df=None):
    # initialization
    statinfo = {}
    columns = [
        'open', 'high', 'close', 'low', 'volume', 'price_change', 'madiff',
        'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20', 'macd'
    ]
    for col in columns:
        statinfo[col] = {'mean': 0.0, 'std': 0.0, 'max': -1e+8, 'min': 1e+8}
    csv_files = os.listdir(basedir)
    if 'stock_basics.csv' in csv_files:
        csv_files.remove('stock_basics.csv')
    csv_files = [os.path.join(basedir, fn) for fn in csv_files]
    # iterate over all stock csv files
    for idx, csv in enumerate(csv_files):
        if idx % 100 == 0:
            print('\r', '>> Function get_stats iter={}'.format(idx), end='')
        ratio = 1.0 / (float(idx) + 1.0)
        csv_df = pd.read_csv(csv)
        for col in columns:
            colmean = csv_df[col].mean() if not math.isnan(csv_df[col].mean()) else statinfo[col]['mean']
            colstd = csv_df[col].std() if not math.isnan(csv_df[col].std()) else statinfo[col]['std']
            statinfo[col]['mean'] += ratio * (colmean - statinfo[col]['mean'])
            statinfo[col]['std'] += ratio * (colstd - statinfo[col]['std'])
            colmax = csv_df[col].max()
            colmin = csv_df[col].min()
            statinfo[col]['max'] = statinfo[col]['max'] if colmax < statinfo[col]['max'] else colmax
            statinfo[col]['min'] = statinfo[col]['min'] if colmin > statinfo[col]['min'] else colmin
    print('\n')
    if other_df is not None:
        for col in other_df.columns:
            if other_df[col].dtype == np.float64 or other_df[col].dtype == np.float32:
                statinfo[col] = {
                    'mean': other_df[col].mean(), 
                    'std': other_df[col].std(), 
                    'max': other_df[col].max(), 
                    'min': other_df[col].min()
                }
    return statinfo


if __name__ == '__main__':
    basic_df = maybe_download_stock_list()
    code_list = basic_df.index.values.tolist()
    filter_func = functools.partial(filter_valid_stocks, basic_df)
    check_and_download_latest_stock_data(code_list, 
        filtering_func=filter_func, 
        feature_prepropcessing_func=feature_prepropcessing)

    # Generate statistic json file using all csv files
    statinfo = get_stats(other_df=basic_df)
    with open('./feature.json', 'w+') as fd:
        json.dump(statinfo, fd, indent=2)
    print(' [*] Done!!')



    # df = pd.read_csv('./stocks/300800.csv')
    # result = feature_prepropcessing(df)
    # result.to_csv('./tmp.csv', index=False)
    # print(' [*] {} out of {} are positive'.format(result['label'].sum(), len(result)))