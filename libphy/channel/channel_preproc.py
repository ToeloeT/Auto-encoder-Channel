import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class ChannelPreProcess(Layer):
    
    LIGHT_SPEED = 299792458. # m/s
    
    # number_taps: desired number of taps
    # carr_freq: carrier frequency (Hz)
    # number_samples: desired number of samples for each channel realization
    # sampling_period: period at which samples at transmitted sampled (s)
    # pulse_shape: pulse shaping function (sinc, raised-cosine, ...)
    # spatial_oversampling: spatial oversampling factor
    def __init__(self, number_taps, carr_freq, num_samples, sampling_period, pulse_shape, spatial_oversampling, **kwargs):
        super(ChannelPreProcess, self).__init__(kwargs)
        
        self.number_taps = number_taps
        self.carr_freq = carr_freq
        self.num_samples = num_samples
        self.sampling_period = sampling_period
        self.spatial_oversampling = spatial_oversampling
        self.pulse_shape = pulse_shape
    
    def sinc(self, x):
        pi = tf.constant(np.pi, tf.float32)
        v = tf.math.divide_no_nan(tf.math.sin(pi*x), pi*x)
        v = tf.where(tf.equal(x, 0.0), 1.0, v)
        return v
        
    # Reconstruct the channel from samples, using sampling theorem
    # coeffs: paths coeff, with shape [batch_size, number of spatial samples, number of paths]
    # speed: user speed in m/s, with shape [batch_size]
    # RETURN channel coefficients, with shape [batch_size, self.num_samples, number of paths]
    def reconstruct_channel(self, coeffs, speed):
        # Channel signal bandwith (Twice the max Doppler shift)
        vm = self.spatial_oversampling*2.*self.carr_freq*speed/ChannelPreProcess.LIGHT_SPEED
        vm = tf.reshape(vm, [-1, 1, 1, 1])

        coeffs = tf.transpose(coeffs, [0, 2, 1])

        # Number of samples from which to reconstruct the signal
        num_spatial_samples = tf.shape(coeffs)[2]

        # Time steps at which to generate samples
        Ts = tf.range(start=1.0, limit=tf.cast(self.num_samples+1,tf.float32), dtype=tf.float32)
        Ts = Ts*self.sampling_period
        Ts = tf.reshape(Ts, [1, -1, 1, 1])

        # Applying the Sampling Theorem
        coeffs = tf.expand_dims(coeffs, axis=1)

        if tf.equal(tf.math.floormod(num_spatial_samples,2), 0):
            l = tf.math.floordiv(num_spatial_samples, 2)
            l = tf.cast(l, tf.float32)
            n = tf.range(-l, l, delta=1, dtype=tf.float32)
        else:
            l = tf.math.floordiv(num_spatial_samples, 2)
            l = tf.cast(l, tf.float32)
            n = tf.range(-l, l+1, delta=1, dtype=tf.float32)
        n = tf.reshape(n, [1, 1, 1, -1])

        x = tf.complex(self.sinc(vm*Ts - n), 0.0)
        coeffs = tf.reduce_sum(coeffs*x, axis=3)

        return coeffs

    # Compute taps coefficients from paths (or rays) coefficients and delays
    # delays: paths delays, with shape [batch_size, number of time_steps, number of paths]
    # coeffs: path coefficients, with shape [batch_size, number of time steps, number of paths]
    # RETURN the taps coefficients, with shape [batch size, number of time steps, number of taps]
    @tf.function
    def paths2taps(self, delays, coeffs):
        paths_delay = tf.expand_dims(delays, axis=2)
        paths_coeff = tf.expand_dims(coeffs, axis=2)
        k = tf.range(tf.cast(self.number_taps, tf.float32), dtype=tf.float32)
        k = tf.reshape(k, [1, 1, -1, 1])
        tap_coeffs = k*self.sampling_period - paths_delay
        tap_coeffs = tf.complex(self.pulse_shape(tap_coeffs), 0.0)*paths_coeff
        tap_coeffs = tf.reduce_sum(tap_coeffs, axis=3)
        return tap_coeffs
    
    # Normalize baseband amplitudes so that the total enery of all paths for each spatial sample is one
    def normalize_coeffs(self, coeffs):
        total_en = tf.reduce_sum(tf.square(tf.abs(coeffs)), axis=2, keepdims=True)
        coeffs = coeffs/tf.complex(tf.sqrt(total_en),0.0)

        return coeffs
        
         
    # ch_data: tensor with shape [batch size, 2, number of spatial samples, number of paths] of type complex64.
    #  The first slice along the first dimension is expected to carry the the channel coefficients
    #  The second slice along the first dimension is expected to carry the delays (and to have 0 on the imaginary parts)
    # speed: tensor with shape [batch size] carrying the speed in m/s for which to generate the taps coefficients
    @tf.function
    def call(self, ch_data, speed):
        
        coeffs = ch_data[:,0,:,:]
        delays = tf.math.real(ch_data[:,1,:,:])
        
        # Normalize
        coeffs = self.normalize_coeffs(coeffs)
        
        # Interpolate the channel coefficients of interest
        coeffs = self.reconstruct_channel(coeffs, speed)
        
        # Using only the delay from the middle of the track
        i = tf.math.floormod(tf.shape(delays)[1], 2)
        delays = tf.tile(tf.expand_dims(delays[:,i,:], axis=1), [1, tf.shape(coeffs)[1], 1])
        
        # Computing the channel tap coefficients
        tap_coeffs = self.paths2taps(delays, coeffs)
        
        return tap_coeffs