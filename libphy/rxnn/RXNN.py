import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, SeparableConv2D, BatchNormalization, Dense
import numpy as np


class ResNetBlock(Layer):
    
    def __init__(self, filters, kernel_size, dilation_rate, use_dense, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_dense = use_dense
                
    def build(self, input_shape):

        if self.use_dense:
            self.dense = Dense(self.filters, use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv1 = SeparableConv2D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate, padding='same')
        self.bn2 = BatchNormalization()
        self.conv2 = SeparableConv2D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate, padding='same')
    
    def call(self, x, training):
        
        z = x
        
        z = self.bn1(z, training=training)
        z = tf.nn.relu(z)
        z = self.conv1(z)
        
        z = self.bn2(z, training=training)
        z = tf.nn.relu(z)
        z = self.conv2(z)
    
        if self.use_dense:
            y = self.dense(x)
        else:
            y = x
        z = z + y
        
        return z

class RXNN(Layer):
    
    def __init__(self, bit_per_cu,  **kwargs):
        super(RXNN, self).__init__(**kwargs)
        
        self.bit_per_cu = bit_per_cu
        
    def build(self, input_shape):
        
        self.conv_in = Conv2D(256, (3,3), dilation_rate=(1,1), padding='same')
        self.res1 = ResNetBlock(256, (3,3), (3,1), False)
        self.res2 = ResNetBlock(256, (3,3), (6,2), False)
        self.res3 = ResNetBlock(256, (3,3), (6,2), False)
        self.res4 = ResNetBlock(256, (3,3), (3,1), False)
        self.conv_out = Conv2D(self.bit_per_cu.numpy(), (1,1), dilation_rate=(1,1), padding='same')
    
    # all inputs of shape [nm subframes, num subcarriers, block length]
    # y: received signal
    # snr_db: SNR [dB]
    def call(self, y, snr_db, training):
        
        # Positional encoding
        num_frames = tf.shape(y)[0]
        num_subc = tf.shape(y)[1]
        block_length = tf.shape(y)[2]
        spectral_pos_enc = tf.linspace(0.0, 1.0, num_subc)
        time_pos_enc = tf.linspace(0.0, 1.0, block_length)
        spectral_pos_enc = tf.expand_dims(spectral_pos_enc, axis=1)
        spectral_pos_enc = tf.tile(spectral_pos_enc, [1, block_length])
        time_pos_enc = tf.expand_dims(time_pos_enc, axis=0)
        time_pos_enc = tf.tile(time_pos_enc, [num_subc, 1])
        pos_enc = tf.stack([spectral_pos_enc, time_pos_enc], axis=2)
        pos_enc = tf.expand_dims(pos_enc, axis=0)
        pos_enc = tf.tile(pos_enc, [num_frames, 1, 1, 1])
        
        # Inputs
        z = tf.stack([tf.math.real(y), tf.math.imag(y),
                      snr_db], axis=3)
        z = tf.concat([z, pos_enc], axis=3)
        print('z: ',z.shape)
        # Call conv layers
        z = self.conv_in(z)
        z = self.res1(z, training)
        z = self.res2(z, training)
        z = self.res3(z, training)
        z = self.res4(z, training)
        z = self.conv_out(z)
        
        return z

"""
class RXNN(Layer):
    
    def __init__(self, bit_per_cu,  **kwargs):
        super(RXNN, self).__init__(**kwargs)
        
        self.bit_per_cu = bit_per_cu
        
    def build(self, input_shape):
        
        self.conv_in = Conv2D(128, (3,3), dilation_rate=(1,1), padding='same')
        self.res1 = ResNetBlock(128, (3,3), (3,2), False)
        self.res2 = ResNetBlock(128, (3,3), (3,2), False)
        self.res3 = ResNetBlock(128, (3,3), (6,4), False)
        self.res4 = ResNetBlock(128, (3,3), (6,4), False)
        self.res5 = ResNetBlock(128, (3,3), (3,2), False)
        self.res6 = ResNetBlock(128, (3,3), (3,2), False)
        self.conv_out = Conv2D(self.bit_per_cu.numpy(), (1,1), dilation_rate=(1,1), padding='same')
    
    # all inputs of shape [nm subframes, num subcarriers, block length]
    # y: received signal
    # snr_db: SNR [dB]
    def call(self, y, snr_db, training):
        
        # Positional encoding
        num_frames = tf.shape(y)[0]
        num_subc = tf.shape(y)[1]
        block_length = tf.shape(y)[2]
        spectral_pos_enc = tf.linspace(0.0, 1.0, num_subc)
        time_pos_enc = tf.linspace(0.0, 1.0, block_length)
        spectral_pos_enc = tf.expand_dims(spectral_pos_enc, axis=1)
        spectral_pos_enc = tf.tile(spectral_pos_enc, [1, block_length])
        time_pos_enc = tf.expand_dims(time_pos_enc, axis=0)
        time_pos_enc = tf.tile(time_pos_enc, [num_subc, 1])
        pos_enc = tf.stack([spectral_pos_enc, time_pos_enc], axis=2)
        pos_enc = tf.expand_dims(pos_enc, axis=0)
        pos_enc = tf.tile(pos_enc, [num_frames, 1, 1, 1])
        
        # Inputs
        z = tf.stack([tf.math.real(y), tf.math.imag(y),
                      snr_db], axis=3)
        z = tf.concat([z, pos_enc], axis=3)
        
        # Call conv layers
        z = self.conv_in(z)
        z = self.res1(z, training)
        z = self.res2(z, training)
        z = self.res3(z, training)
        z = self.res4(z, training)
        z = self.res5(z, training)
        z = self.res6(z, training)
        z = self.conv_out(z)
        
        return z
"""