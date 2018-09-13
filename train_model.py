import tensorflow as tf
import numpy as np
from dataset import Data
import random


def divide_dataset(filename, ratio):
    with open(filename, "r+") as fi:
        data_list = fi.readlines()
    train_list = []
    test_list = []
    for line in data_list:
        tmp_rnd = random.Random(0, 1)
        if tmp_rnd > ratio:
            test_list.append(line)
        else:
            train_list.append(line)

    with open("./data/train.data", "w+") as fi:
        fi.write("".join(train_list))
    with open("./data/test.data", "w+") as fi:
        fi.write("".join(test_list))


def data_import(data_file):
    with open(data_file, "r") as fi:
        data_list = fi.readlines()
    data_set = []
    for _ in data_list:
        line = _.split(",")
        data = Data(line[:-1], line[-1])
        data_set.append(data)
    return data_set


# 数据集相关常数
INPUT_NODE = 9
OUTPUT_NODE = 3

# 神经网络的参数
LAYER1_NODE = 5
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
    return True


if __name__ == "__main__":
    print("")



