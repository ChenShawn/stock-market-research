import tensorflow as tf
import pandas as pd
import numpy as np
import tushare as ts
import datetime
import argparse
import logging
import math
import os

from data import online
from models import burn_in_lstm

logfmt = '[%(levelname)s][%(asctime)s][%(filename)s][%(funcName)s][%(lineno)d] %(message)s'
logging.basicConfig(filename='./data/logs/simulation.log', level=logging.INFO, format=logfmt)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu', default='0,1', type=str, help='which gpu to be used')
    parser.add_argument('--ckpt-name', default='basic_lstm.v2.01', type=str, help='label name')
    parser.add_argument('--model', default='simple', type=str, help='simple|burnin')
    parser.add_argument('--save-dir', default='./train', type=str, help='csv path')
    parser.add_argument('--batch-size', default=64, type=int, help='RESERVED')
    parser.add_argument('--lr', default=1e-3, type=float, help='RESERVED')
    parser.add_argument('--num-epochs', default=40, type=int, help='RESERVED')
    parser.add_argument('--memory-growth', action="store_true", default=False)

    parser.add_argument('--start-time', default='2017-12-01', type=str, help='start time')
    parser.add_argument('--end-time', default='2020-01-01', type=str, help='end time')
    parser.add_argument('--stock-size', default=5, type=int, help='maxnum of stock')
    parser.add_argument('--total-money', default=100000.0, type=float, help='total money')
    parser.add_argument('--tax', default=0.02, type=float, help='tax (coarse estimation)')
    return parser.parse_args()


def load_model(args):
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


def get_policy_label(data):
    if data[0] < 0 and data[1] > 0:
        # negative MACD turning to positive: signal of buy in
        return 1.0
    elif data[0] > 0 and data[1] < 0:
        # positive MACD turning to negative: signal of sell out
        return -1.0
    else:
        return 0.0


def compute_profit(pricelist, money):
    is_holding = False
    num_stocks = 0
    for price in pricelist[: -1]:
        if price < 0 and is_holding == False:
            is_holding = True
            num_stocks = int(math.floor(money / (-price)))
            money += num_stocks * price
        elif price > 0 and is_holding == True:
            is_holding = False
            money += num_stocks * price
            break
    if is_holding == True:
        money += num_stocks * pricelist[-1]
    if money < 20:
        print(pricelist)
        raise RuntimeError
    return money


class TradingSimulator(object):
    def __init__(self, start='2017-11-01', end='2020-01-01', money=1000.0, tax=0.05):
        self.now = start
        self.enddate = end

        assert self.enddate > self.now, \
            f'end must be later than start, end={self.end} start={self.start}'
        self.generator = online.OnlineGenerator(today=start)
        self.total_money = money
        self.tax = tax
        logger.info('TradingSimulator construction finished')


    def start(self, model):
        while self.now < self.enddate:
            selected_stocks = self._select_stocks(model)
            if selected_stocks:
                new_money = self._simulate_single_month(selected_stocks)
                logger.info(f'time={self.now} before={self.total_money} after={new_money}')
            else:
                new_money = self.total_money
            self._next_month()
            self.generator.time_tracking = self.now
            self.total_money = new_money
        return self.total_money


    def _simulate_single_month(self, selected_stocks):
        each_stock_money = self.total_money / float(len(selected_stocks))
        each_stock_money = [each_stock_money for _ in range(len(selected_stocks))]
        for index, stock in enumerate(selected_stocks):
            transactions = self._simulate_single_stock(stock)
            if transactions is None:
                continue
            money = compute_profit(transactions, each_stock_money[index])
            money -= self.tax
            loginfo = 'Date={} stock={} before_transaction={} after_transaction={}'
            logging.info(loginfo.format(self.now, stock, each_stock_money[index], money))
            each_stock_money[index] = money
        return sum(each_stock_money)


    def _simulate_single_stock(self, codestr):
        startstr = self.now
        endstr = '-'.join(self.now.split('-')[: -1]) + '-30'

        stock_fn = os.path.join('./data/stocks/', f'{codestr}.csv')
        try:
            df = pd.read_csv(stock_fn)
        except:
            return None
        df = df[df['date'] <= endstr]
        df = df[df['date'] > startstr]
        # need at least 12 days each month to guarantee the stock has valid data
        if len(df) <= 12:
            info = 'Stock {} doesn\'t have sufficient data from {} to {}, len(df)={}'
            logger.error(info.format(codestr, startstr, endstr, len(df)))
            return None
        
        buy_sell_ratio = df['macd'].rolling(2).apply(get_policy_label)
        # don't do any transaction in the first day of the month
        buy_sell_ratio.iloc[0] = 0.0
        # Any stocks unsold in the last day will be mandatorily sold
        buy_sell_ratio.iloc[-1] = 1.0
        # Tip given by ddhou: use close market price as the transaction price
        buy_sell_price = buy_sell_ratio * df['close']
        return buy_sell_price.values.tolist()


    def _select_stocks(self, model, nstock=5, max_retry=1000, threshold=0.8):
        stock_list = []
        num_trial = 0
        logger.info('Executing stock selection...')
        iterator = iter(self.generator)
        while len(stock_list) < nstock and num_trial < max_retry:
            try:
                batch_xs, codestr = next(iterator)
            except StopIteration:
                break
            probs_op = tf.sigmoid(model.predict(batch_xs))
            probs = probs_op.numpy().tolist()[0][0]
            if probs > threshold:
                stock_list.append(codestr)
                logger.info(f'Selected stock {codestr} in date {self.now}')
            num_trial += 1
        return stock_list

    
    def _next_month(self):
        timestr = self.now.split('-')
        timenums = list(map(int, timestr))
        assert len(timenums) == 3, \
            f'len(timenums)={len(timenums)}, which is supposed to be 3'
        timenums[1] += 1
        if timenums[1] > 12:
            timenums[1] = 1
            timenums[0] += 1
        timenums[-1] = 1
        timestr = '{}-{}-{}'.format(
            str(timenums[0]).zfill(4), 
            str(timenums[1]).zfill(2), 
            str(timenums[2]).zfill(2)
        )
        self.now = timestr


if __name__ == '__main__':
    args = parse_arguments()
    model = load_model(args)
    simulator = TradingSimulator(
        start=args.start_time,
        end=args.end_time,
        money=args.total_money,
        tax=args.tax
    )
    final_money = simulator.start(model)

