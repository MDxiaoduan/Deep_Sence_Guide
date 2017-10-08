import vgg
import lstm

import _init_
import tensorflow as tf

from cosine_distance import cd
cnn = vgg.vgg16()
BL = lstm.lstm()


def BP(softmax_a):
    # BP
    pre = tf.squeeze(tf.matmul(tf.expand_dims(softmax_a, 1), _init_.y_memory))
    print(tf.shape(pre), tf.shape(tf.squeeze(_init_.y_classify_ind)), _init_.y_memory,
          _init_.y_classify)  # tf.squeeze(y_classify_ind)去掉tensor中大小为1的纬度
    top_k = tf.nn.in_top_k(pre, tf.squeeze(_init_.y_classify_ind), 1)  # predictions, targets
    acc = tf.reduce_mean(tf.to_float(top_k))
    correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(pre, 1e-10, 1.0)) * _init_.y_classify, 1)
    loss = tf.reduce_mean(-correct_prob, 0)
    print("********************************************")
    optim = tf.train.AdamOptimizer(_init_.learning_rate)
    print("0")
    grads = optim.compute_gradients(loss)
    print("1")
    train_step = optim.apply_gradients(grads)
    return train_step, loss, acc


class CBL:
    def __index__(self):
        self.batch_size = _init_.batch_size
        self.reuse = False

    def CBL_cnn(self, memory_data, classify_data):
        data_target = []
        data_memory = []
        with tf.variable_scope("cnn") as scope_name:
            for ii in range(_init_.frame):
                if ii > 0:
                    scope_name.reuse_variables()
                data_target.append(cnn(classify_data[:, ii, :, :, :]))   # [_ini.frame, batch_size, 1000]

            for jj in range(_init_.memory_size):  # memory
                data_support = []
                for kk in range(_init_.frame):  # frame
                    data_support.append(cnn(memory_data[:, jj, kk, :, :, :]))  # memory*frame
                data_memory.append(data_support)

        with tf.variable_scope("lstm") as lstm_name:
            # data_target = tf.stack(data_target)      # 拼接矩阵 good function
            target_feature = BL.Multi_lstm(data_target, reuse=False)  # [batch_size, _init_.hidden_size]
            lstm_name.reuse_variables()
            support_feature = []
            for data in data_memory:
                print(tf.shape(data))
                support_feature.append(BL.Multi_lstm(data, reuse=True))
            # support_feature = tf.stack(support_feature)   # [memory_size, batch_size, _init_hidden_size]
        print(target_feature, support_feature)
        softmax_a, similarities = cd(support_feature, target_feature)
        print("softmax_a", softmax_a)
        return softmax_a



