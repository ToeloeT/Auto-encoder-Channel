import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class DemapperFading(Layer):
    
    def __init__(self, bit_per_cu, L, **kwargs):
        super(DemapperFading, self).__init__(**kwargs)
        
        self.bit_per_cu = bit_per_cu
        self.const_size = 2**bit_per_cu
        
        L = np.argmax(L, axis=1)
        C = np.arange(self.const_size)
        C = [[int(j) for j in list(np.binary_repr(i, width=bit_per_cu.numpy()))] for i in C]
        C = np.array(C)
        self.B = []
        for i in range(self.bit_per_cu.numpy()):
            self.B.append([None, None])
            # 0
            a = np.where(C[:,i] == 0)[0]
            self.B[i][0] = np.take(L, a)
            # 1
            a = np.where(C[:,i] == 1)[0]
            self.B[i][1] = np.take(L, a)
    
    
    # y is of shape [num subframes, block length, num subc]
    # C is of shape [num subframes, const size]
    # noise power is a scalar or of shape [num subframes, 1]
    # h is of shape [num subframes, block length, num subc]
    # Ce is of shape [num subframes, num subc, block length]
    def call(self, y, C, noise_power_db, h, Ce):
        
        num_subf = tf.shape(y)[0]
        block_length = tf.shape(y)[2]
        num_subc = tf.shape(y)[1]
        
        # Vectorziing h and y
        y = tf.reshape(y, [num_subf, -1])
        h = tf.reshape(h, [num_subf, -1])
        Ce = tf.reshape(Ce, [num_subf, -1])
        
        noise_power = tf.pow(10.0, noise_power_db/10.0)
        noise_power = tf.expand_dims(noise_power, axis=1)
        v = Ce + noise_power
        
        v = tf.expand_dims(v, axis=2)
        y = tf.expand_dims(y, axis=2)
        C = tf.expand_dims(C, axis=1)
        h = tf.expand_dims(h, axis=2)
        
        ps_logits = -tf.square(tf.abs(y - h*C))/v
        LLR = []
        for i in range(self.bit_per_cu.numpy()):
            D = tf.constant(self.B[i][0], tf.int32)
            pyx0 = tf.gather(ps_logits, D, axis=2)
            
            D = tf.constant(self.B[i][1], tf.int32)
            pyx1 = tf.gather(ps_logits, D, axis=2)
            
            llr = tf.reduce_logsumexp(pyx1, axis=2) - tf.reduce_logsumexp(pyx0, axis=2)
            LLR.append(llr)
        LLR = tf.stack(LLR, axis=2)
        
        # Reverse vectorization
        LLR = tf.reshape(LLR, [num_subf, num_subc, block_length, self.bit_per_cu])
                
        return LLR