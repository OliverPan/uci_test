import numpy as np


class Data(object):
    def __init__(self, feature, label):
        if feature[-1] == "\n":
            self.feature = np.array(feature[:-1], dtype=np.str_)
        else:
            self.feature = np.array(feature, dtype=np.str_)
        if label[-1] == "\n":
            self.label = label[:-1]
        elif not label:
            self.label = None
        else:
            self.label = label


class Dataset(object):
    def __init__(self, dataset):
        feature_lis = []
        label_lis = []
        num = 0
        for data in dataset:
            feature_lis.append(data.feature)
            label_lis.append(data.label)
            num += 1
        self.features = np.array(feature_lis, dtype=np.float32)
        self.labels = np.array(label_lis, dtype=np.float32)
        self.num = num
        self.index = 0

    def next_batch(self, batch_size):
        if self.index + batch_size < self.num:
            return_set = (self.features[self.index:self.index+batch_size],
                          self.labels[self.index:self.index+batch_size])
        else:
            return_set = \
                (np.append(self.features[self.index:], self.features[:(self.index + batch_size) % self.num], axis=0),
                 np.append(self.labels[self.index:], self.labels[:(self.index + batch_size) % self.num], axis=0))
        self.index = (self.index + batch_size) % self.num
        return return_set


