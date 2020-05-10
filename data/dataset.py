import tensorflow as tf
import pandas as pd
import numpy as np
import random
import datetime
import functools
import collections
import logging
import os

try:
    import data.global_variables as G
except ModuleNotFoundError:
    import global_variables as G

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
    3. Split training set and validation set: done
    4. Add logging: done
    5. performance optimization
    
    For performance tips refer to:
    https://tensorflow.google.cn/guide/data_performance?hl=zh-cn
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
        return 0


    def read_next_batch(self, index):
        def get_int32_representation(valuecol, coldf, dtype=np.int32):
            value_set = list(set(valuecol.values.tolist()))
            value_hash = dict(zip(value_set, list(range(len(value_set)))))
            return coldf.apply(lambda x: value_hash[x]).values.astype(dtype)

        # Stock numerical features
        entry = self.get_subdf_entry(index)
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

    def get_subdf_entry(self, index):
        max_entry_size = len(self.candidate_df[index]) - self.lookback
        assert max_entry_size >= 0, \
            f'data length must be larger than lookback: ' \
            f'{len(self.candidate_df[index])} < {self.lookback}'
        entry = random.randint(0, max_entry_size)
        return entry

    def reset(self):
        random.shuffle(self.csv_path)
        self.num_reload = 0
        self.candidates = self.csv_path[: self.batch_size]
        self.iterators = [0 for _ in range(self.batch_size)]
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
    def __init__(self, validation_start='2020-03-01', *args, **kwargs):
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
            f'User should confirm `len(df)>lookback` is strictly satisfied in download.py ' \
            f'file={csvname} len(df)={len(df)} lookback={self.lookback}'
        for col in df.columns:
            if col in self.meanvars.keys():
                norm_fn_partialized = functools.partial(self.normalize_gaussian_data, col)
                df[col] = df[col].apply(norm_fn_partialized)
        return df


class CSVSequentialValidationSet(CSVSequentialDataset):
    def __init__(self, validation_start='2020-03-01', *args, **kwargs):
        self.validation_start = validation_start
        super(CSVSequentialValidationSet, self).__init__(*args, **kwargs)

    def init_numerical_csv(self, csvname):
        df = pd.read_csv(csvname)
        if not csvname.endswith('stock_basics.csv'):
            df = df[df['date'] >= self.validation_start]
        assert len(df) >= self.lookback, \
            f'User should confirm `len(df)>lookback` is strictly satisfied: ' \
            f'file={csvname} len(df)={len(df)} lookback={self.lookback}'
        for col in df.columns:
            if col in self.meanvars.keys():
                norm_fn_partialized = functools.partial(self.normalize_gaussian_data, col)
                df[col] = df[col].apply(norm_fn_partialized)
        return df


class TextLineCSVGenerator(object):
    def __init__(self, csv_file, colspec, batch_size=64, lookback=14):
        """TextLineCSVGenerator.__init__
        An attempt to generator batched data from csv file
        csv_files: type list
        colspec: type OrderedDict, key: colname -> value: function
        """
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.lookback = lookback
        self.fd = open(csv_file, 'r')
        self.colspec = colspec
        self.iterator = 0
        self.fd.readline()

    
    def __call__(self):
        lines = [fd.readline() for _ in range(self.lookback)]
        pass


    def __del__(self):
        self.fd.close()


def build_dataset_from_generator(basedir, batch_size=32, lookback=14, num_epochs=20,
                                 validation_start='2020-03-11'):
    # for training set using `os.listdir`
    csv_files = os.listdir(basedir)
    csv_files = [os.path.join(basedir, fn) for fn in csv_files]
    generator_train = CSVSequentialTrainingSet(
        validation_start=validation_start, 
        csv_path=csv_files, 
        lookback=lookback,
        batch_size=batch_size)
    # for evaluating using local txt file
    with open('./data/validation_files.txt', 'r') as fd: 
        lines = fd.readlines()
        eval_files = [ln[: -1] for ln in lines]
    generator_eval = CSVSequentialValidationSet(
        validation_start=validation_start, 
        csv_path=eval_files, 
        lookback=lookback,
        batch_size=batch_size)
    data_train = tf.data.Dataset.from_generator(generator_train, 
        output_types=((tf.float32, tf.float32, tf.int32), tf.int32)
    ).repeat(num_epochs)
    data_eval = tf.data.Dataset.from_generator(generator_eval, 
        output_types=((tf.float32, tf.float32, tf.int32), tf.int32))
    return data_train, data_eval


def build_tfrecord_dataset(basedir, batch_size=128, num_epochs=20):
    def _parse_function(exam_proto):
        feature_description = {
            'seqfeat': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            'basic_num': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            'basic_cat': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            'label': tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }
        res = tf.io.parse_single_example(exam_proto, feature_description)
        seqfeat = tf.reshape(tf.io.decode_raw(res['seqfeat'], tf.float32), [14, 20])
        basic_num = tf.reshape(tf.io.decode_raw(res['basic_num'], tf.float32), [19])
        basic_cat = tf.reshape(tf.io.decode_raw(res['basic_cat'], tf.int32), [3])
        label = tf.reshape(tf.io.decode_raw(res['label'], tf.int32), [1])
        return (seqfeat, basic_num, basic_cat), label

    base_train = os.path.join(basedir, 'train')
    base_eval = os.path.join(basedir, 'eval')
    record_train = os.listdir(base_train)
    record_train = [os.path.join(base_train, fn) for fn in record_train]
    record_eval = os.listdir(base_eval)
    record_eval = [os.path.join(base_eval, fn) for fn in record_eval]
    data_train = tf.data.TFRecordDataset(record_train).map(_parse_function)
    data_eval = tf.data.TFRecordDataset(record_eval).map(_parse_function)

    data_train = data_train.batch(batch_size).shuffle(2048)
    data_eval = data_eval.batch(batch_size)
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
