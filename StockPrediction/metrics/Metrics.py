import tensorflow as tf
from keras.src.utils import metrics_utils
from keras.src import backend

def accuracy(y_true, y_pred):
    y_true_r = tf.round(y_true * 100) / 100 #arrotonda alla 2 cifra
    y_pred_r = tf.round(y_pred * 100) / 100
    [
        y_pred_r,
        y_true_r,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred_r, y_true_r]
    )
    y_true_r.shape.assert_is_compatible_with(y_pred_r.shape)
    if y_true_r.dtype != y_pred_r.dtype:
        y_pred_r = tf.cast(y_pred_r, y_true_r.dtype)
    return tf.cast(tf.equal(y_true_r, y_pred_r), backend.floatx())



