import tensorflow as tf
import pandas as pd
import numpy as np
import random
import datetime
import functools
import collections
import os

import data.global_variables as G


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
    1. the sampling may become non-iid since the data window are moving in order.
       An alternative implmentation is to adopt random access to csv rows,
       and reload candidates when a fixed number of iteration is reached.
    2. `__call__` should have a terminal condition
    """
    def __init__(self, csv_path, lookback=14, batch_size=32, iter_method='random'):
        """CSVSequentialDataset.__init__
        iter_method: random | order
        """
        self.csv_path = csv_path
        self.lookback = lookback
        self.max_reload = len(csv_path)
        self.batch_size = batch_size
        self.num_reload = 0
        self.iter_method = iter_method

        # Explicitly cache all external variables
        self.stock_numericals = G.STOCK_NUMERICALS.copy()
        self.basic_numericals = G.BASIC_NUMERICALS.copy()
        self.meanvars = G.MEANVARS.copy()
        self.stock_basics = self.init_numerical_csv('./data/stock_basics.csv')
        
        # Dynamicly maintain a candidate list of all csv files
        self.candidates = np.random.choice(csv_path, size=[batch_size], replace=False).tolist()
        self.iterators = [0 for _ in range(batch_size)]
        self.diffset = [csv for csv in csv_path if csv not in self.candidates]
        self.candidate_df = [self.init_numerical_csv(cand) for cand in self.candidates]


    def __call__(self):
        while self.num_reload < self.max_reload:
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
        return -1


    def read_next_batch(self, index):
        def get_int32_representation(valuecol, coldf, dtype=np.int32):
            value_set = list(set(valuecol.values.tolist()))
            value_hash = dict(zip(value_set, list(range(len(value_set)))))
            return coldf.apply(lambda x: value_hash[x]).values.astype(dtype)

        # Stock numerical features
        subdf = self.candidate_df[index].iloc[
            self.iterators[index]: self.iterators[index] + self.lookback
        ]

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
        newcand = random.choice(self.diffset)
        diffidx = self.diffset.index(newcand)
        # print(f' [*] Replace candidate {self.candidates[index]} with {self.diffset[diffidx]}')
        self.diffset[diffidx] = self.candidates[index]
        self.candidates[index] = newcand
        self.iterators[index] = 0
        self.candidate_df[index] = self.init_numerical_csv(newcand)
        self.num_reload += 1


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


    def move_iterator(self, index):
        if self.iter_method == 'order':
            return self.iterators[index] + 1
        elif self.iter_method == 'random':
            maxiter = len(self.candidate_df[index]) - self.lookback
            return random.randint(0, maxiter - 1)
        else:
            raise NotImplementedError('iter_method should be either `order` or `random`')


    def normalize_gaussian_data(self, colname, x, epsilon=1e-9):
        numerator = x - self.meanvars[colname]['mean']
        denominator = self.meanvars[colname]['std'] + epsilon
        return numerator / denominator

    def normalize_numerical_data(self, colname, x, epsilon=1e-9):
        numerator = x - self.meanvars[colname]['min']
        denominator = self.meanvars[colname]['max'] - self.meanvars[colname]['min'] + epsilon
        return numerator / denominator


def create_dataset_from_file(csv_path, 
                             batch_size=64, 
                             num_epochs=20, 
                             shuffle=False, 
                             label_name='label'):
    """create_dataset_from_file
    Used when one single csv file are given
    NOTE: Fucking Google this API directly modify the list given to `select_columns`
    """
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_path,
        batch_size=batch_size,
        select_columns=G.COLUMNS.copy(),
        label_name=label_name,
        num_epochs=num_epochs,
        shuffle=shuffle)
    return dataset


def build_input_features():
    import functools
    # functor to preprocess continuous features
    def process_continuous_data(mean, var, data):
        data = (tf.cast(data, tf.float32) - mean) / var
        return tf.reshape(data, [-1, 1])
    
    # Compared with the codes given in tf2.0 documentary, 
    # this can guarantee the correctness of the feature order
    feature_columns = []
    for col in G.COLUMNS:
        if col in G.CATEGORIES.keys():
            print('[*] processing column key={} type=CATEGORICAL'.format(col))
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=col, vocabulary_list=G.CATEGORIES[col])
            feature = tf.feature_column.indicator_column(cat_col)
        elif col in G.MEANVARS.keys():
            print(' [*] processing column key={} type=NUMERICAL'.format(col))
            norm_fn = functools.partial(process_continuous_data, *G.MEANVARS[col])
            feature = tf.feature_column.numeric_column(key=col, normalizer_fn=norm_fn)
        else:
            continue
        feature_columns.append(feature)

    # preprocessing layer in keras
    preprocessing_layer = tf.keras.layers.DenseFeatures(feature_columns)
    return preprocessing_layer


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
