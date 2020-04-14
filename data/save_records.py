import tensorflow as tf
import pandas as pd
import numpy as np
import functools
import os

from data.dataset import CSVSequentialDataset


class CSVLinearGenerator(CSVSequentialDataset):
    """CSVLinearGenerator
    Override method `get_subdf_entry` to support linear access to csv
    """
    def __init__(self, *args, **kwargs):
        super(CSVLinearGenerator, self).__init__(*args, **kwargs)

    def get_subdf_entry(self, index):
        return self.iterators[index]


class CSVLinearGeneratorTraining(CSVLinearGenerator):
    def __init__(self, validation_start='2020-03-01', *args, **kwargs):
        self.validation_start = validation_start
        super(CSVLinearGeneratorTraining, self).__init__(*args, **kwargs)

    def init_numerical_csv(self, csvname):
        df = pd.read_csv(csvname)
        if not csvname.endswith('./data/stock_basics.csv'):
            df = df[df['date'] < self.validation_start]
        assert len(df) >= self.lookback, \
            f'User should confirm `len(df)>lookback` is strictly satisfied in download.py ' \
            f'len(df)={len(df)} lookback={self.lookback}'
        for col in df.columns:
            if col in self.meanvars.keys():
                norm_fn_partialized = functools.partial(self.normalize_gaussian_data, col)
                df[col] = df[col].apply(norm_fn_partialized)
        return df

    
class CSVLinearGeneratorValidation(CSVLinearGenerator):
    def __init__(self, validation_start='2020-03-01', *args, **kwargs):
        self.validation_start = validation_start
        super(CSVLinearGeneratorValidation, self).__init__(*args, **kwargs)

    def init_numerical_csv(self, csvname):
        df = pd.read_csv(csvname)
        if not csvname.endswith('./data/stock_basics.csv'):
            df = df[df['date'] >= self.validation_start]
        assert len(df) >= self.lookback, \
            f'User should confirm `len(df)>lookback` is strictly satisfied in download.py ' \
            f'file={csvname} len(df)={len(df)} lookback={self.lookback}'
        for col in df.columns:
            if col in self.meanvars.keys():
                norm_fn_partialized = functools.partial(self.normalize_gaussian_data, col)
                df[col] = df[col].apply(norm_fn_partialized)
        return df


def build_linear_dataset_from_generator(basedir, lookback=14, 
                                        validation_start='2020-03-01'):
    # for training set using `os.listdir`
    csv_files = os.listdir(basedir)
    csv_files = [os.path.join(basedir, fn) for fn in csv_files]
    generator_train = CSVLinearGeneratorTraining(
        validation_start=validation_start, 
        csv_path=csv_files, 
        lookback=lookback,
        batch_size=1)
    # for evaluating using local txt file
    with open('./data/validation_files.txt', 'r') as fd: 
        lines = fd.readlines()
        eval_files = [ln[: -1] for ln in lines]
    generator_eval = CSVLinearGeneratorValidation(
        validation_start=validation_start, 
        csv_path=eval_files, 
        lookback=lookback,
        batch_size=1)
    data_train = tf.data.Dataset.from_generator(generator_train, 
        output_types=((tf.float32, tf.float32, tf.int32), tf.int32))
    data_eval = tf.data.Dataset.from_generator(generator_eval, 
        output_types=((tf.float32, tf.float32, tf.int32), tf.int32))
    return data_train, data_eval


def process_dataset(dataset, savedir, num_writers=20):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        if not isinstance(value, np.ndarray):
            value = value.numpy()
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value.tolist()]))

    writerlist = [tf.io.TFRecordWriter(savedir.format(idx)) for idx in range(num_writers)]
    niter = 0
    try:
        for xs, ys in dataset:
            features = tf.train.Features(feature={
                'seqfeat': _bytes_feature(xs[0]),
                'basic_num': _bytes_feature(xs[1]),
                'basic_cat': _bytes_feature(xs[2]),
                'label': _bytes_feature(ys)
            })
            example_proto = tf.train.Example(features=features)
            writerlist[niter].write(example_proto.SerializeToString())
            niter = (niter + 1) % num_writers
    finally:
        for writer in writerlist:
            writer.close()


def main():
    data_train, data_eval = build_linear_dataset_from_generator('./data/stocks/')
    process_dataset(data_train, './data/records/train/train_{}.tfrecord', num_writers=20)
    process_dataset(data_eval, './data/records/eval/eval_{}.tfrecord', num_writers=1)
    print(' [*] Done!!')


if __name__ == '__main__':
    main()
