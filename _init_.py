import tensorflow as tf

video_path = "G:\\Scene\\video\\train\\"
train_img_path = "G:\\Scene\\video\\train_image\\"
image_test_path = "G:\\Scene\\video\\image\\V71002-142546.mp4\\"
image_save = "G:\\Scene\\video\\test\\"
test_video_path = "G:\\Scene\\video\\test_video\\"
test_img_path = "G:\\Scene\\video\\test_image\\"

batch_size = 16
learning_rate = 0.01
image_size = [28, 28, 3]
classes_number = 3
feature_size = 64
iteration_numbers = 500
dis_play = 5
hidden_size = 64
lstm_layer_num = 2

frame = 10
memory_size = int(160/frame)   # 16

ways = memory_size
shot = 1

# placeholder
x_memory = tf.placeholder(tf.float32, shape=[None, memory_size, frame, image_size[0], image_size[1], image_size[2]])     # memory_x
y_memory_ind = tf.placeholder(tf.int32, shape=[None, memory_size])
y_memory = tf.one_hot(y_memory_ind, ways)                                      # memory_label

x_classify = tf.placeholder(tf.float32, shape=[None, frame, image_size[0], image_size[1], image_size[2]])     # classify_x
y_classify_ind = tf.placeholder(tf.int32, shape=[None])
y_classify = tf.one_hot(y_classify_ind, ways)              # classify_label  转为one_hot

