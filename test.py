import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os

from data import dataset, online
from models import burn_in_lstm

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu', default='0,1', type=str, help='which gpu to be used')
parser.add_argument('--ckpt-name', default='basic_lstm.v2.01', type=str, help='label name')
parser.add_argument('--model', default='simple', type=str, help='simple|burnin')
parser.add_argument('--csv-dir', default='./data/records/', type=str, help='csv path')
parser.add_argument('--save-dir', default='./train', type=str, help='csv path')
parser.add_argument('--batch-size', default=64, type=int, help='RESERVED')
parser.add_argument('--lr', default=1e-3, type=float, help='RESERVED')
parser.add_argument('--num-epochs', default=40, type=int, help='RESERVED')
parser.add_argument('--memory-growth', action="store_true", default=False)
args = parser.parse_args()


def load_model():
    if args.model == 'simple':
        model = burn_in_lstm.SimpleSequentialLSTM()
    elif args.model == 'burnin':
        model = burn_in_lstm.BurnInStateLSTM()
    else:
        raise NotImplementedError('model must be among simple|burnin')

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    adam = tf.keras.optimizers.Adam(lr=args.lr)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model.compile(loss=loss, optimizer=adam, metrics=metrics)
    model.build(input_shape=[(None, 14, 20), (None, 19), (None, 3)])

    if '{}.hdf5'.format(args.ckpt_name) in os.listdir(args.save_dir):
        try:
            model.load(os.path.join(args.save_dir, args.ckpt_name + '.hdf5'))
        except:
            model.load_weights(os.path.join(args.save_dir, args.ckpt_name + '.hdf5'))
        print(' [*] Loaded pretrained model {}.hdf5'.format(args.ckpt_name))
    return model


def evaluate():
    data_train, data_eval = dataset.build_tfrecord_dataset(
        basedir=args.csv_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
    model = load_model()
    results = model.evaluate(data_eval)
    print(results)


def online_filtering(model, threshold=0.8, logfd=None):
    positive_list = []
    generator = online.OnlineGenerator()
    for features, codestr in generator:
        logit = model.predict(features)
        prob = tf.sigmoid(logit).numpy()[0]
        if prob > threshold:
            codename = generator.from_code_to_name(int(codestr), 'name')
            time_to_market = generator.from_code_to_name(int(codestr), 'timeToMarket')
            positive_list.append((codestr, codename, time_to_market, prob))
            if logfd is not None:
                logline = f'{codestr} | {codename} | {time_to_market} | {prob}\n'
                logfd.write(logline)
                logfd.flush()
            print(f' [*] Stock {codestr} is positive')
    print(f' [*] Total number of positive stocks: {len(positive_list)}')
    return positive_list


if __name__ == '__main__':
    # evaluate()
    # data_train, data_eval = dataset.build_tfrecord_dataset('./data/records/')
    # print(data_train, data_eval)

    # xs, ys = next(iter(data_eval))
    # print(xs, '\n\n\n\n\n')
    # print(ys, '\n\n\n\n\n')

    if tf.test.is_built_with_cuda():
        if args.memory_growth:
            gpu_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        if len(args.gpu) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = load_model()
    with open('./docs/stock_selection.md', 'a+') as fd:
        positive_list = online_filtering(model, logfd=fd)
    print('[*] Done!!')
    