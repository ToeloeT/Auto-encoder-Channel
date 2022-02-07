import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from libphy.modulation.QAM_tools import gen_const


class GS(Layer):
    
    def __init__(self, bit_per_cu, init_qam=False, **kwargs):
        super(GS, self).__init__(**kwargs)
        
        self.bit_per_cu = bit_per_cu
        self.init_qam = init_qam
        
    def build(self, input_shape):
        
        if self.init_qam:
            C = gen_const(self.bit_per_cu.numpy())
            en_point = np.mean(np.sum(np.square(C), axis=1))
            C = C / np.sqrt(en_point)
            C = tf.constant(C, tf.float32)
            self.Cr = tf.Variable(C[:,0], dtype=tf.float32, trainable=True)
            self.Ci = tf.Variable(C[:,1], dtype=tf.float32, trainable=True)
        else:
            const_size = 2**self.bit_per_cu
            self.Cr = tf.Variable(tf.random_normal_initializer()(shape=[const_size], dtype=tf.float32), trainable=True)
            self.Ci = tf.Variable(tf.random_normal_initializer()(shape=[const_size], dtype=tf.float32), trainable=True)
        
    def get_const(self):
        C = tf.complex(self.Cr, self.Ci)
        
        # Center the constellation
        C = C - tf.reduce_mean(C)
        
        # Normalization
        en_point = tf.reduce_mean(tf.square(tf.math.real(C)) + tf.square(tf.math.imag(C)))
        norm_fac = tf.complex(tf.sqrt(en_point), 0.0)
        C = tf.math.divide_no_nan(C, norm_fac)
        
        return C
        

    # s is of shape [batch, block length] and is the index of constellation points
    def call(self, s):
        
        batch_size = tf.shape(s)[0]
        C = self.get_const()
        C = tf.tile(tf.expand_dims(C, axis=0), [batch_size, 1])
                
        # Mapping
        x = tf.gather(C, s, batch_dims=1)
        
        return x