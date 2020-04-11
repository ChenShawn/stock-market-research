import pandas as pd
import os, sys


split_time = '2020-03-01'
basedir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'stocks')

lookback = 14
csv_files = os.listdir(basedir)
csv_files = [os.path.join(basedir, csv) for csv in csv_files]


def filter_invalid_files(csvname):
    df = pd.read_csv(csvname)
    df = df[df['date'] > split_time]
    return len(df) >= lookback


valid_files = list(filter(filter_invalid_files, csv_files))
with open('./validation_files.txt', 'w+') as fd:
    for fn in valid_files:
        fd.write(f'{fn}\n')
