import tensorflow as tf
from tensorflow.keras.layers import Layer


class MultiPathChannel(Layer):
    
    def __init__(self, **kwargs):
        super(MultiPathChannel, self).__init__(kwargs)
         
    # x: time domain baseband channel input symbols, with shape [batch_size, number of time steps]
    # h: channel response in time domain, with shape [batch_size, number of time steps, number of taps]
    # noise_power_db: Noise power in dB. Scalar
    @tf.function
    def call(self, x, h, noise_power_db):
        
        batch_size = tf.shape(x)[0]
        time_steps_no = tf.shape(x)[1]
        taps_no = tf.shape(h)[2]
        
        # If extra time steps are available for the channel, we just ignore them
        h = h[:,:time_steps_no,:]
        
        # Reshaping the channel input symbols
        x = tf.expand_dims(x, axis=2)
        x = tf.tile(x, [1, 1, taps_no])
        padding = tf.zeros([batch_size, taps_no, taps_no], tf.complex64)
        x = tf.concat([x, padding], axis=1)
        x_ = tf.transpose(tf.zeros_like(x), [2, 0, 1])
        for i in tf.range(taps_no):
            z = x[:,:,i]
            z = tf.roll(z, i, axis=1)
            x_ = tf.tensor_scatter_nd_update(x_, [[i]], [z])
        x_ = tf.transpose(x_, [1, 2, 0])
        x = x_[:,:time_steps_no,:] # [batch size, number of timer steps, number of taps]
        
        # Channel transfer function
        y = tf.reduce_sum(h*x, axis=2)
        noise_std = tf.sqrt(0.5*(tf.pow(10.0, noise_power_db/10.0)))
        noise_std = tf.expand_dims(noise_std, axis=1)
        noise = (tf.random.normal(tf.shape(y), mean=0.0, stddev=noise_std), tf.random.normal(tf.shape(y), mean=0.0, stddev=noise_std))
        noise = tf.complex(noise[0], noise[1])
        y = y + noise
        
        return y