import tensorflow as tf
from dataset import Data, Dataset
import random
import numpy as np


def divide_dataset(feature_filename, label_filename, ratio):
    with open(feature_filename, "r+") as fi:
        feature_data_list = fi.readlines()
    with open(label_filename, "r+") as fi:
        label_data_list = fi.readlines()
    num = feature_data_list.__len__()
    train_feature_list = []
    test_feature_list = []
    train_label_list = []
    test_label_list = []
    for i in range(num):
        tmprdm = random.uniform(0, 1)
        if tmprdm < ratio:
            train_feature_list.append(feature_data_list[i])
            train_label_list.append(label_data_list[i])
        else:
            test_feature_list.append(feature_data_list[i])
            test_label_list.append(label_data_list[i])

    with open("./data/train/train_feature.data", "w+") as fi:
        fi.write("".join(train_feature_list))
    with open("./data/train/train_label.data", "w+") as fi:
        fi.write("".join(train_label_list))

    with open("./data/test/test_feature.data", "w+") as fi:
        fi.write("".join(test_feature_list))
    with open("./data/test/test_label.data", "w+") as fi:
        fi.write("".join(test_label_list))


def data_import(feature_data, label_data):
    with open(feature_data, "r") as fi:
        feature_list = fi.readlines()
    with open(label_data, "r") as fi:
        label_list = fi.readlines()
    data_set = []
    for i in range(feature_list.__len__()):
        data = Data(list(feature_list[i])[0:24], list(label_list[i])[0:1])
        data_set.append(data)
    return Dataset(data_set)


# 数据集相关常数
INPUT_NODE = 24
OUTPUT_NODE = 1

# 神经网络的参数
LAYER1_NODE = 320

BATCH_SIZE = 50

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 200000
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


def train(data_set, test_set, test_data):
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

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: data_set.features, y_: data_set.labels}
        test_feed = {x: test_set.features, y_: test_set.labels}
        tmp = 0
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(loss, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, 1 - validate_acc))
                print(sess.run(y, feed_dict={x: [data_set.features[tmp]]}))
                tmp += 1
            xs, ys = data_set.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(loss, feed_dict=test_feed)
        print("After %d training step(s), test accuracy "
              "using average model is %g " % (TRAINING_STEPS, 1 - test_acc))
        # print(sess.run(average_y, feed_dict={x: test_data, y_: np.array([[1]], dtype=np.float32)}))


def main(argv=None):
    divide_dataset("./data/TestData/TestData/Ra.txt", "./data/TestData/TestData/leak0.txt", 0.6)
    data_set = data_import("./data/train/train_feature.data", "./data/train/train_label.data")
    test_set = data_import("./data/test/test_feature.data", "./data/test/test_label.data")

    string1 = "11110010000001100000000010010100"
    string2 = "10000100100001101010010011111011"
    test_data = np.array([list(string2)], dtype=np.float32)
    train(data_set, test_set, test_data)


if __name__ == "__main__":
    tf.app.run()


