import tensorflow as tf

def mean_squared_error(x, y):
    return tf.reduce_mean(tf.square(x - y))
