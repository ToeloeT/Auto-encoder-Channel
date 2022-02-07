import tensorflow as tf

class OFDM:
    
    # num_subc: number of subcarriers
    # cp_length: Cycle prefix duration [symbols]
    # use_cp: If True, uses a cycling-prefix, otherwise does not.
    def __init__(self, num_subc, cp_length, use_cp=True, **kwargs):
        
        self.num_subc = num_subc
        self.cp_length = cp_length
        self.use_cp = use_cp
    
    # x_f: Baseband symbols in the frequency domain [batch size, number of subcarriers, number of OFDM symbols]
    # The ordering of the subcarriers is assumed to be as follows: negatve frequencies (in order of decreasingly negative frequency),
    #  then center frequency, then positive frequencies.
    @tf.function
    def modulate(self, x_f):
        
        # Checking the number of subcarriers is large enough given the CP length
        tf.debugging.assert_positive(tf.shape(x_f)[1] - self.cp_length, "Number of subcarriers too small given the specified CP length")
        
        x_f = tf.transpose(x_f, [0, 2, 1])
        
        # FFT shift
        x_f = tf.signal.ifftshift(x_f, axes=2)
        
        # Moving to time domain
        x = tf.signal.ifft(x_f)*tf.complex(tf.sqrt(tf.cast(tf.shape(x_f)[-1], tf.float32)), 0.0)
        
        # Adding CP
        if self.use_cp:
            x = tf.concat([x[:,:,-self.cp_length:], x], axis=2)
        
        # Reshaping to get a single time domain signal
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        
        return x
        
    
    # y_t: Baseband symbols in the time domain [batch size, time steps]
    @tf.function
    def demodulate(self, y_t):
        
        if self.use_cp:
            symbol_length = self.num_subc + self.cp_length
        else:
            symbol_length = self.num_subc
        
        # Checking the signal dimension matches the specified CP length and number of subcarriers
        num_symbs = tf.math.floordiv(tf.shape(y_t)[1], symbol_length)
        rem_symbs = tf.math.floormod(tf.shape(y_t)[1], symbol_length)
        tf.debugging.assert_positive(num_symbs, "Signal duration is too short given specified OFDM configuration")
        tf.debugging.assert_equal(rem_symbs, 0, "Signal duration does not match specified OFDM configuration")
        
        # Reshaping
        y = tf.reshape(y_t, [tf.shape(y_t)[0], -1, symbol_length])
        
        # Removing CP
        if self.use_cp:
            y = y[:,:,self.cp_length:]
        
        # Moving to frequency domain
        y = tf.signal.fft(y)/tf.complex(tf.sqrt(tf.cast(tf.shape(y)[-1], tf.float32)), 0.0)
        
        # FFT shift
        y = tf.signal.fftshift(y, axes=2)
        
        y = tf.transpose(y, [0, 2, 1])
        
        return y