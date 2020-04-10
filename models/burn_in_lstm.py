import tensorflow as tf
import numpy as np
from collections import namedtuple

from . import seq2seq


def build_lstm_by_device(units, input_shape):
    if tf.test.is_built_with_cuda():
        lstm = tf.keras.layers.LSTM(
            units=units, input_shape=(None, input_shape), 
            return_sequence=True, return_state=True)
    else:
        lstm = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(units), 
            input_shape=(None, input_shape), 
            return_sequences=True, return_state=True)
    return lstm


class SimpleSequentialLSTM(tf.keras.Model):
    """SimpleSequentialLSTM
    A simpler version of BurnInStateLSTM,
    implemented for debugging and baseline comparison
    """
    def __init__(self):
        super(SimpleSequentialLSTM, self).__init__()
        self.code_embedding = tf.keras.layers.Embedding(3825, 64)
        self.area_embedding = tf.keras.layers.Embedding(32, 6)
        self.industry_embedding = tf.keras.layers.Embedding(110, 12)
        self.lstm = build_lstm_by_device(128, 20)
        self.shared_dense = tf.keras.layers.Dense(64, 'relu')
        self.basic_dense = tf.keras.layers.Dense(64, 'relu')
        self.global_dense = tf.keras.layers.Dense(1)


    def call(self, input_ops):
        input_seq, input_basic_num, input_basic_cat = input_ops

        # processing sequential features
        seq_list = tf.unstack(input_seq, axis=1)
        seq_embedding = [self.shared_dense(seq) for seq in seq_list]
        lstm_output, state_h, state_c = self.lstm(seq_embedding)
        
        # processing basic features
        industry, area, codenum = tf.unstack(input_basic_cat, axis=-1)
        industry_embedded = self.industry_embedding(industry)
        area_embedded = self.area_embedding(area)
        codenum_embedded = self.code_embedding(codenum)

        basic_features = tf.concat([
            input_basic_num, 
            industry_embedded,
            area_embedded, 
            codenum_embedded
        ], axis=-1)
        basic_embedding = self.basic_dense(basic_features)

        # processing global features
        global_features = tf.concat([state_h, basic_embedding], axis=-1)
        logits = self.global_dense(global_features)
        return logits


class BurnInStateLSTM(tf.keras.Model):
    def __init__(self, burn_in_length=7):
        super(BurnInStateLSTM, self).__init__()
        self.burn_in_length = burn_in_length

        self.code_embedding = tf.keras.layers.Embedding(3825, 64)
        self.area_embedding = tf.keras.layers.Embedding(32, 6)
        self.industry_embedding = tf.keras.layers.Embedding(110, 12)

        self.lstm = build_lstm_by_device(128, 20)
        self.luong_attention = seq2seq.LuongAttention(128)
        self.flatten = tf.keras.layers.Flatten()

        self.shared_dense = tf.keras.layers.Dense(64, 'relu')
        self.basic_dense = tf.keras.layers.Dense(64, 'relu')
        self.global_dense = tf.keras.layers.Dense(1)


    def call(self, input_ops):
        input_seq, input_basic_num, input_basic_cat = input_ops

        # processing sequential features
        seq_list = tf.unstack(input_seq, axis=1)
        assert len(seq_list) > self.burn_in_length, \
            f'Input sequences length must be larger than burn-in size: ' \
            f'burn_in_size: {self.burn_in_length} > input_sequence_size: {len(seq_list)}'

        seq_embedding = [self.shared_dense(seq) for seq in seq_list]
        burn_in_input = tf.stack(seq_embedding[: self.burn_in_length], axis=1)
        updated_seq = tf.stack(seq_embedding[self.burn_in_length: ], axis=1)
        encoder_output, state_h, state_c = self.lstm(burn_in_input)
        
        encoder_output = tf.stop_gradient(encoder_output)
        state_h = tf.stop_gradient(state_h)
        state_c = tf.stop_gradient(state_c)

        decoder_output, _, __ = self.lstm(updated_seq, initial_state=(state_h, state_c))
        context, _ = self.luong_attention(encoder_output, decoder_output)
        context_flatten = self.flatten(context)
        
        # processing basic features
        industry, area, codenum = tf.unstack(input_basic_cat, axis=-1)
        industry_embedded = self.industry_embedding(industry)
        area_embedded = self.area_embedding(area)
        codenum_embedded = self.code_embedding(codenum)

        basic_numerical = self.basic_dense(input_basic_num)

        # processing global features
        global_features = tf.concat([
            context_flatten, 
            basic_numerical, 
            industry_embedded, 
            area_embedded, 
            codenum_embedded
        ], axis=-1)
        logits = self.global_dense(global_features)
        return logits


    def stock_cosine_similarity(self, x, y):
        """stock_cosine_similarity
        For all three functions below must conform:
        Both input x and y must be type int
        """
        emb_xs = self.code_embedding(x)
        emb_ys = self.code_embedding(y)
        norm_product = tf.norm(emb_xs, axis=-1) * tf.norm(emb_ys, axis=-1)
        return tf.reduce_sum(emb_xs * emb_ys, axis=-1) / norm_product

    def industry_cosine_similarity(self, x, y):
        emb_xs = self.industry_embedding(x)
        emb_ys = self.industry_embedding(y)
        norm_product = tf.norm(emb_xs, axis=-1) * tf.norm(emb_ys, axis=-1)
        return tf.reduce_sum(emb_xs * emb_ys, axis=-1) / norm_product

    def area_cosine_similarity(self, x, y):
        emb_xs = self.area_embedding(x)
        emb_ys = self.area_embedding(y)
        norm_product = tf.norm(emb_xs, axis=-1) * tf.norm(emb_ys, axis=-1)
        return tf.reduce_sum(emb_xs * emb_ys, axis=-1) / norm_product


    def get_state_and_context(self, input_seq):
        """get_state_and_context
        I don't know when would this function be useful for me,
        probably it's gonna be interesting to check whether context 
        or LSTM final hidden are good representations for sequential features,
        which might be useful to analyze sequential stock data.
        """
        seq_list = tf.unstack(input_seq, axis=1)
        assert len(seq_list) > self.burn_in_length, \
            f'Input sequences length must be larger than burn-in size: ' \
            f'burn_in_size: {self.burn_in_length} > input_sequence_size: {len(seq_list)}'

        seq_embedding = [self.shared_dense(seq) for seq in seq_list]
        burn_in_input = tf.stack(seq_embedding[: self.burn_in_length], axis=1)
        updated_seq = tf.stack(seq_embedding[self.burn_in_length: ], axis=1)
        encoder_output, state_h, state_c = self.lstm(burn_in_input)
        
        encoder_output = tf.stop_gradient(encoder_output)
        state_h = tf.stop_gradient(state_h)
        state_c = tf.stop_gradient(state_c)
        decoder_output, _, __ = self.lstm(updated_seq, initial_state=(state_h, state_c))
        context, _ = self.luong_attention(encoder_output, decoder_output)

        states_tuple = namedtuple('HiddenState', ['h', 'c'])
        return HiddenState(h=state_h, c=state_c), context

        
if __name__ == '__main__':
    model = BurnInStateLSTM()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    adam = tf.keras.optimizers.Adam(1e-3)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model.compile(loss=loss, optimizer=adam, metrics=metrics)

    a = tf.convert_to_tensor([[4]], dtype=tf.int32)
    b = tf.convert_to_tensor([[19]], dtype=tf.int32)
    cossim = model.stock_cosine_similarity(a, b)
    print(f' [*] Cosine similarity: {cossim}  with shape {cossim.shape}')
