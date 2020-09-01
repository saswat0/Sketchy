import time
import random
import os
import tensorflow as tf
import numpy as np
import time
from skimage import io, transform, exposure, color
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
time.sleep(2)


np.set_printoptions(precision=2)
import argparse

from tensorflow.python.tools import freeze_graph

# from skimage import io, transform, exposure, color

parser = argparse.ArgumentParser()
parser.add_argument("--ngf", type=int, default=48, help="number of generator filters in first conv layer")
a = parser.parse_args()


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
