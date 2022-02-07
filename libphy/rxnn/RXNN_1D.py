import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, SeparableConv1D, BatchNormalization, Dense
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
        self.conv1 = SeparableConv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate, padding='same')
        self.bn2 = BatchNormalization()
        self.conv2 = SeparableConv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate, padding='same')
    
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
        
        self.conv_in = Conv1D(128, 3, dilation_rate=1, padding='same')
        self.res1 = ResNetBlock(128, 3, 3, False)
        self.res2 = ResNetBlock(128, 3, 3, False)
        self.res3 = ResNetBlock(128, 3, 6, False)
        self.res4 = ResNetBlock(128, 3, 3, False)
        self.res5 = ResNetBlock(128, 3, 3, False)
        self.conv_out = Conv1D(self.bit_per_cu.numpy(), 1, dilation_rate=1, padding='same')
    
    # y [complex valued, (num blocks, block length)] : received baseband signal
    # Output: [real valued, (num blocks, block length, bit per cu)]
    def call(self, y, training):
        
        # Inputs
        z = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=2)
        
        # Call conv layers
        z = self.conv_in(z)
        z = self.res1(z, training)
        z = self.res2(z, training)
        z = self.res3(z, training)
        z = self.res4(z, training)
        z = self.res5(z, training)
        z = self.conv_out(z)
        
        return z