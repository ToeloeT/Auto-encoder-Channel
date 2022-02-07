import tensorflow as tf
from tensorflow.keras.layers import Layer


# Simulate an additive Gaussian noise channel with arbitrary transmit and receive filters.
# ISI can be introduced due to the arbitrary filters, as well as correlated noise.

class AdditiveNoiseChannel(Layer):
    
    # pulse_shape: A pulse shape filter, to generate the channel taps
    # number_of_taps: total number of taps including both time directions. Must be an odd integer
    def __init__(self, pulse_shape, number_of_taps, **kwargs):
        super(AdditiveNoiseChannel, self).__init__(kwargs)
        
        tf.debugging.Assert(tf.equal(tf.math.mod(number_of_taps, 2), 1), number_of_taps, "Number of steps must be odd")
        
        self.pulse_shape = pulse_shape
        self.number_of_taps = number_of_taps
         
    # x [complex, (batch_size, block size)]: time domain baseband channel input symbols, with shape
    # N0 [real, (batch_size,)]: Noise power spectral density [dB]
    @tf.function
    def call(self, x, N0):
        
        N0 = tf.cast(tf.pow(10.0, N0/10.0), tf.complex64)
        batch_size = tf.shape(x)[0]
        block_size = tf.shape(x)[1]
        number_taps_one_side = (self.number_of_taps.numpy()-1)//2 # Number of taps on each side (pos or neg)
                
        # Compute the channe taps
        a = self.pulse_shape(self.number_of_taps) #(number of taps,)
        a = tf.tile(tf.expand_dims(tf.expand_dims(a, axis=0), axis=1), [batch_size, block_size, 1]) # (batch size, block size, number of taps)
        
        # Reshaping the channel input symbols
        x = tf.expand_dims(x, axis=2) # (batch size, block size, 1)
        x = tf.tile(x, [1, 1, number_taps_one_side+1]) # (batch size, block size, number of taps on each side)
        padding = tf.zeros([batch_size, number_taps_one_side+1, number_taps_one_side+1], tf.complex64) # (batch size, block size, number of taps on each side)
        x_1 = tf.concat([x,padding], axis=1) # (batch size, block size + number of taps on each size, number of taps on each side)
        x_2 = tf.concat([padding,x], axis=1) # (batch size, block size + number of taps on each size, number of taps on each side)
        x__1 = tf.zeros([number_taps_one_side+1, batch_size, block_size+number_taps_one_side+1], tf.complex64) # (number of taps on each side, batch size, block size + number of taps on each side)
        x__2 = tf.zeros([number_taps_one_side+1, batch_size, block_size+number_taps_one_side+1], tf.complex64) # (number of taps on each side, batch size, block size + number of taps on each side)
        for i in range(number_taps_one_side+1):
            z_1 = x_1[:,:,i] # (block size, block size + number of taps on each size)
            z_1 = tf.roll(z_1, i, axis=1) # (block size, block size + number of taps on each size)
            x__1 = tf.tensor_scatter_nd_update(x__1, [[i]], [z_1]) # (block size, batch size, block size + number of taps on each size)
            #
            z_2 = x_2[:,:,i] # (batch size, block size + number of taps on each size)
            z_2 = tf.roll(z_2, -i, axis=1) # (batch size, block size + number of taps on each size)
            x__2 = tf.tensor_scatter_nd_update(x__2, [[i]], [z_2]) # (block size, batch size, block size + number of taps on each size)
        x__1 = tf.transpose(x__1, [1, 2, 0]) # (batch size, block size + number of taps on each side, number of taps on each side)
        x__2 = tf.transpose(x__2, [1, 2, 0]) # (batch size, block size + number of taps on each side, number of taps on each side)
        x__2 = tf.reverse(x__2[:,:,1:], axis=[2])  # (batch size, block size + number of taps on each side, number of taps on each side - 1)
        x__1 = x__1[:,:-number_taps_one_side-1,:] # (batch size, block size, number of taps on each side)
        x__2 = x__2[:,number_taps_one_side+1:,:] # (batch size, block size, number of taps on each side - 1)
        x = tf.concat([x__2, x__1], axis=2) # (batch size, block size, number of taps)

        # Cholesky decomposition
        L = self.pulse_shape.gen_cov_cholesky()
        L = tf.tile(tf.expand_dims(L, axis=0), [batch_size, 1, 1])
        # Generating correlated noise
        noise = (tf.random.normal([batch_size, block_size], mean=0.0, stddev=1.0), tf.random.normal([batch_size, block_size], mean=0.0, stddev=1.0))
        noise = tf.complex(noise[0], noise[1])
        noise = tf.expand_dims(tf.sqrt(0.5*N0), axis=1)*noise
        noise = tf.squeeze(tf.matmul(L, tf.expand_dims(noise, axis=2)), axis=2)
        
        # Channel transfer function
        y = tf.reduce_sum(a*x, axis=2)
        y = y + noise
        
        return y