import numpy as np


class Data(object):
    def __init__(self, feature, label):
        self.feature = np.array(feature, dtype=np.str_)
        if label.endswith("/n"):
            self.label = label[:-1]
        else:
            self.label = label

