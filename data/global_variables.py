import tensorflow as tf
import pandas as pd
import datetime
import os
import json

"""
GLOBAL VARIABLES DEFINITIONS
"""
# Global stock_basics.csv DataFrame stored in memory permanently
STOCK_BASICS = pd.read_csv('./stock_basics.csv')
CSV_BASEDIRS = os.listdir('./stocks/')
CSV_STOCK_FILES = [os.path.join('./stocks/', x) for x in CSV_BASEDIRS]

PROFIT_BASEDIRS = os.listdir('./profit/')
CSV_PROFIT_FILES = [os.path.join('./profit/', x) for x in PROFIT_BASEDIRS]

# for numerical feature columns
CATEGORIES = {
    'industry': list(set(STOCK_BASICS['industry'].values.tolist())),
    'area': list(set(STOCK_BASICS['area'].values.tolist())),
    #'codenum': list(set(STOCK_BASICS['codenum'].values.tolist())), 
    'weekday': ['Mon', 'Tue', 'Wedn', 'Thur', 'Fri', 'Sat', 'Sun']
}
# for categarical feature columns
with open('./feature.json', 'r') as fd:
    MEANVARS = json.load(fd)

COLUMNS = [
    'open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 
    'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20', 'weekday', 'madiff', 'macd'
]
BASIC_COLUMNS = [
    'industry', 'area', 'pe', 'outstanding', 'totals',
    'totalAssets', 'liquidAssets', 'fixedAssets', 'reserved',
    'reservedPerShare', 'esp', 'bvps', 'pb', 'undp', 'codenum',
    'perundp', 'rev', 'profit', 'gpr', 'npr', 'holders', 'exist_time'
]
STOCK_CATEGORICALS = ['weekday']
BASIC_CATEGORICALS = ['industry', 'area', 'codenum']
STOCK_NUMERICALS = [col for col in COLUMNS if col not in STOCK_CATEGORICALS]
BASIC_NUMERICALS = [col for col in BASIC_COLUMNS if col not in BASIC_CATEGORICALS]

# RECORD_DEFAULTS_ = [
#     tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, 
#     tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, 
#     tf.float32, tf.float32, tf.float32, tf.float32, tf.string, 
#     tf.float32, tf.int32
# ]

# RECORD_DEFAULTS = [[0.0] for _ in range(14)] + [[0], [0.0], [0]]


del STOCK_BASICS