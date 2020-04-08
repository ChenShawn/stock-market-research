import tensorflow as tf
import numpy as np

from data.dataset import CSVSequentialDataset


class BurnInStateLSTM(tf.keras.Model):
    def __init__(self):
        super(BurnInStateLSTM, self).__init__()

        self.weekday_embedding = tf.keras.layers.Embedding(5, )
