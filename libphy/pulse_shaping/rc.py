import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

# Should be normalized such that the PSD, defined by P(f) = W |H(f)|^2, H(f) is the Fourier transform of the filter, integrates to one.

class RC(Layer):
    
    PI = tf.constant(np.pi, tf.float32)
    
    # bandwidth [real valued, (1,)] : Bandwidth (not including excess bandwidth) [Hz]
    #  Typically, W = 1/T, where T is the sampling period.
    # beta [real valued in (0,1), (1,)] : roll-off factor
    def __init__(self, bandwidth, beta, **kwargs):
        super(RC, self).__init__(**kwargs)
        
        self.bandwidth = bandwidth
        self.T = 1.0/bandwidth
        
        tf.debugging.Assert(tf.greater_equal(beta, 0.0) and tf.less_equal(beta, 1.0), beta, 'Rased-cosine: invalid roll-off factor value')
        self.beta = tf.Variable(beta, trainable=False)
        
    def sinc(self, x):
        v = tf.math.divide_no_nan(tf.math.sin(RC.PI*x), RC.PI*x)
        v = tf.where(tf.equal(x, 0.0), 1.0, v)
        return v

    def rc(self, t):
        v = (1.0/self.T) * self.sinc(t/self.T) * tf.math.divide_no_nan(tf.cos(RC.PI*self.beta*t/self.T), (1.0 - tf.square(2*self.beta*t/self.T)))
        if tf.not_equal(self.beta, 0.0):
            v = tf.where(tf.equal(t, self.T/(2.*self.beta)), (RC.PI/(4.*self.T))*self.sinc(1./(2*self.beta)), v)
            v = tf.where(tf.equal(t, -self.T/(2.*self.beta)), (RC.PI/(4.*self.T))*self.sinc(1./(2*self.beta)), v)
            
        v = v/self.bandwidth
        
        return v
         
    # L [integer, ()]: desired number of taps. Assumed to be odd
    @tf.function
    def call(self, L):
        #return self.rc(t)
        L_side = (L-1)//2
        a = tf.concat([tf.zeros(L_side, tf.complex64), tf.ones([1], tf.complex64), tf.zeros(L_side, tf.complex64)], axis=0)
        return a

    @tf.function
    def total_energy(self):
        return tf.constant(1.0, tf.float32)
    
    @tf.function
    def aclr(self):
        in_band = 1. + self.beta*(1./RC.PI - 0.5)
        oo_band = self.beta*(0.5 - 1./RC.PI)
        v = oo_band/in_band
        return v
    
    def set_beta(self, beta):
        self.beta.assign(beta)
        
    def noise_covariance(self, size):
        return tf.eye(size, dtype=tf.complex64)