import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from libphy.modulation.QAM_tools import gen_const


class QAM(Layer):

    def __init__(self, bit_per_cu, **kwargs):
        super(QAM, self).__init__(**kwargs)
        
        self.bit_per_cu = bit_per_cu
        
    def build(self, input_shape):
        C = gen_const(self.bit_per_cu.numpy())
        en_point = np.mean(np.sum(np.square(C), axis=1))
        C = C / np.sqrt(en_point)
        C = tf.constant(C, tf.float32)
        self.C = tf.complex(C[:,0], C[:,1])

    # s is of shape [batch, block length] and is the index of constellation points
    def call(self, s):
        
        return tf.gather(self.C, s)