import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Interleaver(Layer):

    def __init__(self, **kwargs):
        super(Interleaver, self).__init__(**kwargs)

        
    # reuse: if True, reuse the same permutation as before
    def interleave(self, x, reuse=False):
        
        x_shape = tf.shape(x)
        batch_size = x_shape[0]
        x = tf.reshape(x, [batch_size, -1])
        x_len = tf.shape(x)[1]
        
        if reuse:
            perm = self.perm
        else:
            perm = tf.random.shuffle(tf.range(x_len))
            self.perm = perm # Store it for deinterleaving
        
        x_perm = tf.gather(x, perm, axis=1)
        x_perm = tf.reshape(x_perm, x_shape)
        
        return x_perm
    
    def de_interleave(self, y_perm):
        
        y_shape = tf.shape(y_perm)
        batch_size = y_shape[0]
        y_perm = tf.reshape(y_perm, [batch_size, -1])
        y_len = tf.shape(y_perm)[1]
        
        ind0 = tf.range(batch_size)
        ind0 = tf.reshape(tf.tile(tf.expand_dims(ind0, axis=1), [1, y_len]), [batch_size*y_len, 1])
        ind1 = self.perm
        ind1 = tf.reshape(tf.tile(tf.expand_dims(ind1, axis=1), [batch_size, 1]), [batch_size*y_len, 1])
        ind = tf.concat([ind0, ind1], axis=1)
        ind = tf.reshape(ind, [batch_size, y_len, 2])
        
        y = tf.scatter_nd(ind, y_perm, tf.shape(y_perm))
        y = tf.reshape(y, y_shape)
        
        return y