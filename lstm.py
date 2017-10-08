import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import _init_


class lstm:
    def __init__(self):
        self.time_step_size = _init_.frame
        self.hidden_size = _init_.hidden_size
        self.layer_num = _init_.lstm_layer_num
        self.reuse = False

    def Multi_lstm(self, x, reuse):
        # 输入shape = (batch_size, timestep_size, input_size)    # [_ini.frame, batch_size, 1000]
        # x = tf.reshape(x, [-1, _init_.frame, _init_.feature_size])
        # 1-layer LSTM with n_hidden units.
        rnn_cell = rnn.BasicLSTMCell(self.hidden_size, reuse=reuse)
        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        return outputs[-1]

