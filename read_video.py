import _init_
import cv2
import os
import random
import numpy as np
from DeepLearning.Image import plot_images


def read_video_to_img(path, img_path):
    for train_name in os.listdir(path):
        if os.path.exists(img_path + train_name):
            pass
        else:
            os.mkdir(img_path + train_name)
        cap = cv2.VideoCapture(path + train_name)
        ret, frame = cap.read()
        fgbg = cv2.createBackgroundSubtractorMOG2()
        count = 0
        while ret is True:  # 如果视频没有结束 ret就是True
            if count % 5 == 0 and count < 800:
                fname = "_{:03d}.jpg".format(count)
                frame = np.rot90(frame, axes=(1, 0))
                cv2.imwrite(img_path + train_name + "\\" + train_name + fname, frame)
            ret, frame = cap.read()
            count += 1


class read_video:
    def __init__(self):
        self.memory_data = np.zeros((_init_.batch_size, _init_.memory_size, _init_.frame, _init_.image_size[0], _init_.image_size[1], _init_.image_size[2]))
        self.memory_label = np.zeros((_init_.batch_size, _init_.memory_size))
        self.classify_data = np.zeros((_init_.batch_size, _init_.frame, _init_.image_size[0], _init_.image_size[1], _init_.image_size[2]))
        self.classify_data_show = np.zeros(
            (_init_.batch_size, _init_.frame, 400, 400, 3))
        self.classify_label = np.zeros(_init_.batch_size)

    def get_mini_batch(self, name="train"):
        if name == "test":
            img_path = _init_.test_img_path
        else:
            img_path = _init_.train_img_path
        path_list = []
        for file in os.listdir(img_path):
            path_list.append(file)
        begin_list = []
        for ii in range(_init_.batch_size):
            train_name = path_list[np.random.randint(0, int(len(path_list)))]
            file_list = []
            count_memory = 0
            count_frame = 0
            for file in os.listdir(img_path + train_name):
                file_list.append(file)
                if count_frame == _init_.frame:
                    count_memory += 1
                    self.memory_label[ii, count_memory] = int(count_memory)
                    count_frame = 0
                frame = cv2.imread(img_path + train_name + "\\" + file)
                self.memory_data[ii, count_memory, count_frame, :, :, :] = cv2.resize(frame, (_init_.image_size[0], _init_.image_size[1]))
                count_frame += 1

            hat_class = np.random.randint(_init_.memory_size)  # 随机取一组作为测试分类0-15上取一个
            self.classify_label[ii] = hat_class
            if hat_class == 0:
                begin = np.random.randint(0, 2)
            elif hat_class == _init_.memory_size - 1:
                begin = np.random.randint(148, 150)
            else:
                begin = np.random.randint(hat_class*_init_.frame - 2, hat_class*_init_.frame + 2)
            begin_list.append(begin)
            for kk in range(begin, begin + _init_.frame):
                frame = cv2.imread(img_path + train_name + "\\" + file_list[kk])
                self.classify_data[ii, int(kk - begin), :, :, :] = cv2.resize(frame, (_init_.image_size[0], _init_.image_size[1]))
                self.classify_data_show[ii, int(kk - begin), :, :, :] = cv2.resize(frame, (400, 400))
        return self.memory_data, self.memory_label, self.classify_data, self.classify_label, begin_list, self.classify_data_show

# read_video_to_img(_init_.test_video_path, _init_.test_img_path)
# if __name__ == '__main__':
#     memory_data, memory_label, classify_data, classify_label = read_video().get_mini_batch()
#     print(classify_label[0])
#     plot_images(classify_data[0, :, :, :, :], classify_label[0])
#     plot_images(memory_data[0, :, 0, :, :, :], memory_label[0])
#
#     plot_images(classify_data[1, :, :, :, :], classify_label[0])
#     plot_images(memory_data[1, :, 0, :, :, :], memory_label[0])

