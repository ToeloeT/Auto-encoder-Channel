import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class EstimatorLMMSE(Layer):
    
    def __init__(self, **kwargs):
        super(EstimatorLMMSE, self).__init__(**kwargs)
    
    # Assumes:
    # y is of shape [num subframes, block length, num subc]
    # R is of shape [num subframes, block length*num subc, block length*num subc]
    # P is of shape [num subframes, block length, num subc], where 0 indicates no pilots
    def call(self, y, R, P, noise_power_db):
                
        num_subf = tf.shape(y)[0]
        block_length = tf.shape(y)[2]
        num_subc = tf.shape(y)[1]
        noise_var = tf.pow(10.0, noise_power_db/10.0)
        noise_var = tf.expand_dims(tf.expand_dims(noise_var, axis=1), axis=2)
        
        I = tf.eye(block_length*num_subc, batch_shape=[num_subf])
                
        # Vectorziing y and P
        y = tf.reshape(y, [num_subf, -1])
        P = tf.reshape(P, [num_subf, -1])
        
        # Projection matrix
        PI = tf.where(tf.not_equal(P, 0.0))[:,1]
        PI = tf.reshape(PI, [num_subf, -1])
        PI = tf.one_hot(PI, block_length*num_subc, dtype=tf.float32)
        PI = tf.complex(PI, tf.zeros(tf.shape(PI), tf.float32))
        PIh = tf.transpose(PI, perm=[0, 2, 1], conjugate=True)
        
        # Recevied pilots
        y = tf.expand_dims(y, axis=2)
        yp = tf.matmul(PI, y)
        
        P = tf.linalg.diag(P)
        Ph = tf.transpose(P, perm=[0, 2, 1], conjugate=True)
        RPh = tf.matmul(R, Ph)
        RPhPIh = tf.matmul(RPh, PIh)

        L = tf.matmul(P, RPh) + tf.complex(noise_var*I, 0.0)
        L = tf.matmul(PI, tf.matmul(L, PIh))
        L = tf.linalg.inv(L)
        L = tf.matmul(RPhPIh, L)
        
        # Compute an estimate of h
        h_hat = tf.matmul(L, yp)
        h_hat = tf.squeeze(h_hat, axis=2)
        
        # Compute the error covariance matrix
        Ce = R - tf.matmul(L, tf.transpose(RPhPIh, perm=[0, 2, 1], conjugate=True))
        Ce = tf.math.real(tf.linalg.diag_part(Ce))
        Ce = tf.reshape(Ce, [num_subf, num_subc, block_length])
        
        # 'Unvectorize' h_hat
        h_hat = tf.reshape(h_hat, [num_subf, num_subc, block_length])
        
        return h_hat, Ce