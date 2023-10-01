import tensorflow as tf
from keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.W_q = self.add_weight("W_q", shape=(input_shape[-1], self.embed_dim))
        self.W_k = self.add_weight("W_k", shape=(input_shape[-1], self.embed_dim))
        self.W_v = self.add_weight("W_v", shape=(input_shape[-1], self.embed_dim))

    def call(self, inputs):
        q = tf.matmul(inputs, self.W_q)
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_probs, v)
        return output