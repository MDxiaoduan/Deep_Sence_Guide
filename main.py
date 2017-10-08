from __future__ import division
import tqdm
from _init_ import *

import CBL
import cv2
import numpy as np
import matplotlib.pyplot as plt
from read_video import read_video
from CBL import BP
from DeepLearning.python import list_save

data = read_video()
cnn = CBL.CBL()


softmax_a = cnn.CBL_cnn(x_memory, x_classify)
print(softmax_a)

train_step, loss, acc = BP(softmax_a)

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    ACC = []
    with tqdm.tqdm(total=int(iteration_numbers/dis_play)) as pbar:
        while step < iteration_numbers:
            memory_data, memory_label, classify_data, classify_label, begin, classify_data_show = data.get_mini_batch(name="train")
            _, loss_, acc_, softmax_a_ = sess.run([train_step, loss, acc, softmax_a],
                                                       feed_dict={x_memory: memory_data, y_memory_ind: memory_label,
                                                                  x_classify: classify_data, y_classify_ind: classify_label})
            step += 1
            if step % dis_play == 0:
                memory_data, memory_label, classify_data, classify_label, begin, classify_data_show = data.get_mini_batch(
                    name="test")
                acc_, softmax_a_ = sess.run([acc, softmax_a],
                                                      feed_dict={x_memory: memory_data, y_memory_ind: memory_label,
                                                                 x_classify: classify_data,y_classify_ind: classify_label})
                ACC.append(acc_)
                pbar.set_description("train_loss:{}, train_accuracy: {}".format(loss_, acc_))
                pbar.update(1)

                if step == 490:
                    for ii in range(batch_size):
                        plt.figure(ii)
                        plt.plot((np.arange(softmax_a_.shape[1])), softmax_a_[ii, :])
                        plt.title("location is " + str(begin[ii]))
                        plt.savefig(image_save + "Scene" + str(ii) + "_location_" + str(begin[ii]) + ".jpg")
                    for ii in range(batch_size):
                        for jj in range(frame):
                            cv2.imwrite(image_save + "test" + str(ii) + "_" + str(jj) + "_" + str(begin[ii]) + ".jpg",
                                        classify_data_show[ii, jj, :, :, :])
        list_save(ACC, "data\\acc.txt")
        plt.plot((np.arange(len(ACC))), ACC)
        plt.show()
