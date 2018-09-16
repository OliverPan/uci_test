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
        data = Data(list(feature_list[i]), list(label_list[i]))
        data_set.append(data)
    return Dataset(data_set)


# 数据集相关常数
INPUT_NODE = 32
OUTPUT_NODE = 4

# 神经网络的参数
LAYER1_NODE = 20
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 300
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if not avg_class:
        layer1 = tf.sigmoid(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.sigmoid(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(data_set):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        data_set.num / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict={x: data_set.features, y_: data_set.labels})
                print("After %d training step(s), validation accuracy "
                      "using average model is %g " % (i, validate_acc))

            xs, ys = data_set.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})


def main(argv=None):
    divide_dataset("./data/TestData/TestData/Ra.txt", "./data/TestData/TestData/leak0.txt", 0.4)
    data_set = data_import("./data/train/train_feature.data", "./data/train/train_label.data")
    train(data_set)


if __name__ == "__main__":
    tf.app.run()


