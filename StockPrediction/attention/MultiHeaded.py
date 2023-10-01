from keras.layers import Layer
import tensorflow as tf

# Definizione dello strato di Self-Attention
class MultiHeadedAttention(Layer):
    def __init__(self, num_heads, head_dim, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def build(self, input_shape):
        self.W_q = self.add_weight("W_q", shape=(input_shape[-1], self.num_heads * self.head_dim))
        self.W_k = self.add_weight("W_k", shape=(input_shape[-1], self.num_heads * self.head_dim))
        self.W_v = self.add_weight("W_v", shape=(input_shape[-1], self.num_heads * self.head_dim))

    def call(self, inputs):
        q = tf.matmul(inputs, self.W_q)
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)

        q = tf.reshape(q, (-1, tf.shape(q)[1], self.num_heads, self.head_dim))
        k = tf.reshape(k, (-1, tf.shape(k)[1], self.num_heads, self.head_dim))
        v = tf.reshape(v, (-1, tf.shape(v)[1], self.num_heads, self.head_dim))

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_scores, v)
        output = tf.reshape(output, (-1, tf.shape(output)[1], self.num_heads * self.head_dim))

        return output