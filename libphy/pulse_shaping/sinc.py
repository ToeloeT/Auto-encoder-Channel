import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


# Should be normalized such that the PSD, defined by P(f) = W |H(f)|^2, H(f) is the Fourier transform of the filter, integrates to one.
# For sinc, there is nothing to do.

class Sinc(Layer):
    
    # bandwidth [real valued, (1,)] : Bandwidth [Hz] of the filter.
    #  Typically, W = 1/T, where T is the sampling period.
    def __init__(self, bandwidth, block_size, **kwargs):
        super(Sinc, self).__init__(**kwargs)
        
        self.bandwidth = bandwidth
        self.block_size = block_size
    
    def sinc(self, x):
        pi = tf.constant(np.pi, tf.float32)
        v = tf.math.divide_no_nan(tf.math.sin(pi*x), pi*x)
        v = tf.where(tf.equal(x, 0.0), 1.0, v)
        return v
         
    # L [integer, ()]: desired number of taps. Assumed to be odd
    @tf.function
    def call(self, L):
        #return self.sinc(self.bandwidth*t)
        L_side = (L-1)//2
        a = tf.concat([tf.zeros(L_side, tf.complex64), tf.ones([1], tf.complex64), tf.zeros(L_side, tf.complex64)], axis=0)
        return a
    
    def total_energy(self):
        return tf.constant(1.0, tf.float32)
    
    def aclr(self):
        return tf.constant(0.0, tf.float32)
    
    def gen_cov_cholesky(self):
        return tf.eye(self.block_size, dtype=tf.complex64)
    