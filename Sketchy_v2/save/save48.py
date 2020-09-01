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

def batchnorm(input,channels):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

       # channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

