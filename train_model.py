import tensorflow as tf
from dataset import Data, Dataset
import random
import numpy as np
import sys


def divide_dataset(feature_filename, label_filename, ratio, append_file=None):
    with open(feature_filename, "r+") as fi:
        feature_data_list = fi.readlines()
    with open(label_filename, "r+") as fi:
        label_data_list = fi.readlines()
    num = feature_data_list.__len__()
    train_feature_list = []
    test_feature_list = []
    train_label_list = []
    test_label_list = []
    if not append_file:
        for i in range(num):
            tmprdm = random.uniform(0, 1)
            if tmprdm < ratio:
                train_feature_list.append(feature_data_list[i])
                train_label_list.append(label_data_list[i])
            else:
                test_feature_list.append(feature_data_list[i])
                test_label_list.append(label_data_list[i])
    else:
        with open(append_file, "r") as fi:
            append_list = fi.readlines()
        for i in range(num):
            tmprdm = random.uniform(0, 1)
            tmp = feature_data_list[i][:-1] + append_list[i]
            if tmprdm < ratio:
                train_feature_list.append(tmp)
                train_label_list.append(label_data_list[i])
            else:
                test_feature_list.append(tmp)
                test_label_list.append(label_data_list[i])

    with open("./data/train/train_feature.data", "w+") as fi:
        fi.write("".join(train_feature_list))
    with open("./data/train/train_label.data", "w+") as fi:
        fi.write("".join(train_label_list))

    with open("./data/test/test_feature.data", "w+") as fi:
        fi.write("".join(test_feature_list))
    with open("./data/test/test_label.data", "w+") as fi:
        fi.write("".join(test_label_list))


def data_import(feature_data, label_data, bit):
    with open(feature_data, "r") as fi:
        feature_list = fi.readlines()
    with open(label_data, "r") as fi:
        label_list = fi.readlines()
    data_set = []
    for i in range(feature_list.__len__()):
        data = Data(list(feature_list[i]), [list(label_list[i])[bit]])  # list(label_list[i])[0:1] 代表的是输出数据第0位构成的列表
        data_set.append(data)
    return Dataset(data_set)


def integrate_data(file1, file2):
    return True


# 数据集相关常数
INPUT_NODE = 64     # 输入数据的位数
OUTPUT_NODE = 1     # 输出数据的位数

# 神经网络的参数
LAYER1_NODE = 640

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.9999
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 500000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if not avg_class:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def inference2(input_tensor, avg_class, weights1, biases1):
    if not avg_class:
        return tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    else:
        return tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))


def train(data_set, test_set):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(tf.random_normal([INPUT_NODE, LAYER1_NODE]))
    biases1 = tf.Variable(tf.zeros([LAYER1_NODE]))

    weights2 = tf.Variable(tf.random_normal([LAYER1_NODE, OUTPUT_NODE]))
    biases2 = tf.Variable(tf.zeros([OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)

    loss = tf.reduce_mean(tf.square(y - y_))

    learning_rate = 0.01

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    flag = False

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: data_set.features, y_: data_set.labels}
        test_feed = {x: test_set.features, y_: test_set.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(loss, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, 1 - validate_acc))
            xs, ys = data_set.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: xs, y_: ys})

            if (1 - validate_acc) > 0.991:
                flag = True
                break

        test_acc = sess.run(loss, feed_dict=test_feed)
        print("After %d training step(s), test accuracy "
              "using average model is %g " % (TRAINING_STEPS, 1 - test_acc))
        # print(sess.run(average_y, feed_dict={x: test_data, y_: np.array([[1]], dtype=np.float32)}))
        return flag, [sess.run(weights1), sess.run(biases1), sess.run(weights2), sess.run(biases2)]


def main(argv=None):
    divide_dataset("./data/TestData_crc/CRC/Rb.txt", "./data/TestData_crc/CRC/2.txt", 0.99,
                   "./data/TestData_crc/CRC/Ra.txt")
    # 读文件，第一个是输入，第二个是输出，第三个是训练集数据占数据总量的比
    # 如果训练精度始终达不到（对于奇偶校验和海明校验）,大概率是训练集太小,需要增加训练集数据量

    result = []
    bit = 0
    while True:
        data_set = data_import("./data/train/train_feature.data", "./data/train/train_label.data", bit)
        test_set = data_import("./data/test/test_feature.data", "./data/test/test_label.data", bit)

        flag, tmp_result = train(data_set, test_set)

        if flag:
            np.savetxt("./parameter/crc/2/data"+str(bit)+"w1.txt", tmp_result[0])
            np.savetxt("./parameter/crc/2/data"+str(bit)+"b1.txt", tmp_result[1])
            np.savetxt("./parameter/crc/2/data"+str(bit)+"w2.txt", tmp_result[2])
            np.savetxt("./parameter/crc/2/data"+str(bit)+"b2.txt", tmp_result[3])
            print(bit)
            bit += 1
        else:
            continue
        if bit >= 16:
            break


if __name__ == "__main__":

    tf.app.run()


