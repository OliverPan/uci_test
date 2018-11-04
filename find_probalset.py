import numpy as np


def import_rc(filename):
    with open(filename, "r+") as fi:
        rc_set = fi.readlines()
