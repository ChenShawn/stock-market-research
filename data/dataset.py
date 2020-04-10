import tensorflow as tf
import pandas as pd
import numpy as np
import random
import datetime
import functools
import collections
import logging
import os

import data.global_variables as G

logfmt = '[%(levelname)s][%(asctime)s][%(funcName)s][%(lineno)d] %(message)s'
logging.basicConfig(filename='./data/logs/dataset.log', level=logging.INFO, format=logfmt)


class CSVSequentialDataset(object):
    """CSVSequentialDataset
    NOTE: Acoording to tf official documentation link
    https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn#from_generator

    1. You should not use this method if you need to serialize 
       your model and restore it in a different environment with different devices.
    2. Mutating global variables or external state can cause undefined behavior, 
       and we recommend that you explicitly cache any external state in generator 
       before calling `Dataset.from_generator()`.

    TODO:
    1. random access to csv: done
    2. terminal condition: done 
    3. Split training set and validation set.
    4. Add logging: done
    """
    def __init__(self, csv_path, lookback=14, batch_size=32):
        self.csv_path = csv_path.copy()
        random.shuffle(self.csv_path)
        self.lookback = lookback
        self.batch_size = batch_size
        self.num_reload = 0
        self.logger = logging.getLogger(__name__)

        # Explicitly cache all external variables
        self.stock_numericals = G.STOCK_NUMERICALS.copy()
        self.basic_numericals = G.BASIC_NUMERICALS.copy()
        self.meanvars = G.MEANVARS.copy()
        self.stock_basics = self.init_numerical_csv('./data/stock_basics.csv')
        
        # Dynamicly maintain a candidate list of all csv files
        self.candidates = self.csv_path[: batch_size]
        self.iterators = [0 for _ in range(batch_size)]
        self.candidate_df = [self.init_numerical_csv(cand) for cand in self.candidates]
        self.logger.info('CSV datasets has been built')


    def __call__(self):
        while self.num_reload < len(self.csv_path) - self.batch_size:
            xs_num, basic_num, basic_cat, batch_ys = [], [], [], []
            for idx in range(len(self.candidate_df)):
                if self.iterators[idx] >= len(self.candidate_df[idx]) - self.lookback:
                    # reload another candidate if current csv iterated to the end
                    self.reload_another_candidate(idx)

                seq, bnum, bcat, label = self.read_next_batch(idx)
                self.iterators[idx] += 1
                xs_num.append(seq)
                basic_num.append(bnum)
                basic_cat.append(bcat)
                batch_ys.append(label)
            xs_num = tf.stack(xs_num, axis=0)
            basic_num = tf.concat(basic_num, axis=0)
            basic_cat = tf.stack(basic_cat, axis=0)
            batch_ys = tf.concat(batch_ys, axis=0)
            yield (xs_num, basic_num, basic_cat), batch_ys
        self.reset()
        return self.next_cand_index


    def read_next_batch(self, index):
        def get_int32_representation(valuecol, coldf, dtype=np.int32):
            value_set = list(set(valuecol.values.tolist()))
            value_hash = dict(zip(value_set, list(range(len(value_set)))))
            return coldf.apply(lambda x: value_hash[x]).values.astype(dtype)

        # Stock numerical features
        max_entry_size = len(self.candidate_df[index]) - self.lookback - 1
        assert max_entry_size > 0, \
            f'data length must be larger than lookback: ' \
            f'{len(self.candidate_df[index])} < {self.lookback}'
        entry = random.randint(0, max_entry_size)
        subdf = self.candidate_df[index].iloc[entry: entry + self.lookback]

        subdf_numerical = subdf[self.stock_numericals].values.astype(np.float32)
        # Stock catogorical features
        weekday = get_int32_representation(
            self.candidate_df[index]['weekday'], subdf['weekday'])
        weekday_onehot = np.eye(5, dtype=np.int32)[weekday]
        seqfeat = tf.concat([subdf_numerical, weekday_onehot], axis=-1)

        # Basic numerical features
        codenum = int(self.candidates[index].split('/')[-1].split('.')[0])
        basic_line = self.stock_basics[self.stock_basics['codenum'] == codenum]
        assert len(basic_line) == 1, f'find {len(basic_line)} lines of {codenum} in stock_basics.csv'
        basic_numerical = basic_line[self.basic_numericals].values.astype(np.float32)

        # Basic categorical features
        industry = get_int32_representation(self.stock_basics['industry'], basic_line['industry'])
        area = get_int32_representation(self.stock_basics['area'], basic_line['area'])
        stockcode = get_int32_representation(self.stock_basics['codenum'], basic_line['codenum'])
        basic_categorical = tf.concat([industry, area, stockcode], axis=-1)

        # processing labels
        label = [subdf['label'].iloc[-1]]
        return seqfeat, basic_numerical, basic_categorical, label


    def reload_another_candidate(self, index):
        newcand = self.csv_path[self.num_reload + self.batch_size]
        msg = f'Replace candidate {self.candidates[index]} with {newcand}'
        self.candidates[index] = newcand
        self.iterators[index] = 0
        self.candidate_df[index] = self.init_numerical_csv(newcand)
        self.num_reload += 1
        self.logger.info(msg)
        self.logger.info(f'num_reload={self.num_reload}')


    def init_numerical_csv(self, csvname):
        """init_numerical_csv
        Use either gaussian normalization or 0-1 normalization for numerical columns
        """
        df = pd.read_csv(csvname)
        for col in df.columns:
            if col in self.meanvars.keys():
                norm_fn_partialized = functools.partial(self.normalize_gaussian_data, col)
                df[col] = df[col].apply(norm_fn_partialized)
        return df

    def reset(self):
        random.shuffle(self.csv_path)
        self.num_reload = 0
        self.candidates = self.csv_path[: batch_size]
        self.iterators = [0 for _ in range(batch_size)]
        self.candidate_df = [self.init_numerical_csv(cand) for cand in self.candidates]
        self.logger.info('CSV dataset status has been reset')

    def normalize_gaussian_data(self, colname, x, epsilon=1e-9):
        numerator = x - self.meanvars[colname]['mean']
        denominator = self.meanvars[colname]['std'] + epsilon
        return numerator / denominator

    def normalize_numerical_data(self, colname, x, epsilon=1e-9):
        numerator = x - self.meanvars[colname]['min']
        denominator = self.meanvars[colname]['max'] - self.meanvars[colname]['min'] + epsilon
        return numerator / denominator


class CSVSequentialTrainingSet(CSVSequentialDataset):
    def __init__(self, validation_start='2020-03-11', *args, **kwargs):
        self.validation_start = validation_start
        super(CSVSequentialTrainingSet, self).__init__(*args, **kwargs)

    def init_numerical_csv(self, csvname):
        """init_numerical_csv
        Overriding base class method `init_numerical_csv`
        Split training set before `validation_start`
        """
        df = pd.read_csv(csvname)
        if not csvname.endswith('stock_basics.csv'):
            df = df[df['date'] < self.validation_start]
        assert len(df) >= self.lookback, \
            f'User should confirm `len(df)>lookback` is strictly satisfied in download.py' \
            f'len(df)={len(df)} lookback={self.lookback}'
        for col in df.columns:
            if col in self.meanvars.keys():
                norm_fn_partialized = functools.partial(self.normalize_gaussian_data, col)
                df[col] = df[col].apply(norm_fn_partialized)
        return df


class CSVSequentialValidationSet(CSVSequentialDataset):
    def __init__(self, validation_start='2020-03-11', *args, **kwargs):
        self.validation_start = validation_start
        super(CSVSequentialValidationSet, self).__init__(*args, **kwargs)

    def init_numerical_csv(self, csvname):
        df = pd.read_csv(csvname)
        if not csvname.endswith('stock_basics.csv'):
            df = df[df['date'] > self.validation_start]
        assert len(df) >= self.lookback, \
            f'User should confirm `len(df)>lookback` is strictly satisfied in download.py' \
            f'len(df)={len(df)} lookback={self.lookback}'
        for col in df.columns:
            if col in self.meanvars.keys():
                norm_fn_partialized = functools.partial(self.normalize_gaussian_data, col)
                df[col] = df[col].apply(norm_fn_partialized)
        return df


def build_dataset_from_generator(basedir, batch_size=32, lookback=14, 
                                 validation_start='2020-03-11'):
    csv_files = os.listdir(basedir)
    csv_files = [os.path.join(basedir, fn) for fn in csv_files]
    generator_train = CSVSequentialTrainingSet(
        validation_start=validation_start, 
        csv_path=csv_files, 
        lookback=lookback,
        batch_size=batch_size)
    generator_eval = CSVSequentialValidationSet(
        validation_start=validation_start, 
        csv_path=csv_files, 
        lookback=lookback,
        batch_size=batch_size)
    data_train = tf.data.Dataset.from_generator(generator_train, 
        output_types=((tf.float32, tf.float32, tf.int32), tf.int32))
    data_eval = tf.data.Dataset.from_generator(generator_eval, 
        output_types=((tf.float32, tf.float32, tf.int32), tf.int32))
    return data_train, data_eval


if __name__ == '__main__':
    generator = CSVSequentialDataset(G.CSV_STOCK_FILES, batch_size=32)
    dataset = tf.data.Dataset.from_generator(generator, 
        output_types=(tf.float32, tf.float32, tf.int32, tf.int32))
    
    a,b,c,d = next(iter(dataset))
    a,b,c,d = next(iter(dataset))
    a,b,c,d = next(iter(dataset))
    a,b,c,d = next(iter(dataset))
    a,b,c,d = next(iter(dataset))
    a,b,c,d = next(iter(dataset))
    print('\n\n\n\n\n\t')
    print(a, end='\n\n')
    print(b, end='\n\n')
    print(c, end='\n\n')
    print(d, end='\n\n')
    print(a.shape, b.shape, c.shape, d.shape)
