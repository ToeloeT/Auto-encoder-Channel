import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import numpy as np

class Parity_Bit(Layer):
    
    def __init__(self, H, H_std=None, **kwargs):
        super(Parity_Bit, self).__init__(**kwargs)
        
        m,n = H.shape
        k = n - m
        
        # Wite H in standard form
        if H_std is None:
            H_std,_ = self.gauss_jordan(H)
        self.H_std = H_std
        
        # Matrix tha generates the parity bits
        P = H_std[:,:k]
        P = P.T
                
        max_deg = np.max( [np.where(P[:,i] != 0)[0].shape[0] for i in range(P.shape[1])] )
        Ps = np.zeros((n-k, max_deg), dtype=int)
        Ms = np.zeros((n-k, max_deg), dtype=int)
        for i in range(n-k):
            u = np.where(P[:,i] != 0)[0]
            Ps[i,:u.shape[0]] = u
            Ms[i,:u.shape[0]] = np.ones((u.shape[0]), dtype=int)
        self.Ps = Ps
        self.Ms = Ms

    def gauss_jordan(self, H):
        m = H.shape[0] # Number of redunduncy bit
        n = H.shape[1] # code length
        k = n - m # number of information bit

        # To track changes
        I = np.eye(m).astype('int')

        H_std = np.copy(H)
        A = H_std[:,k:] # Matrix that must be diagonoal at the end of this process

        for l in range(m):

            # Find the pivot
            p = None
            for i in range(l,m):
                if A[i,l] == 1:
                    p = i
                    break
            if p is None:
                print("Error: no pivot found")
                return None

            # Switch rows if required
            if p > l:
                I[[l,p]] = I[[p,l]]
                H_std[[l,p]] = H_std[[p,l]]
                A = H_std[:,k:]

            # Substract the current row from the others
            for i in range(m):
                if i == l:
                    continue
                if A[i,l] == 0:
                    continue
                I[i] = (I[i] + I[l]) % 2
                H_std[i] = (H_std[i] + H_std[l]) % 2
                A = H_std[:,k:]

        return H_std, I
    
    def build(self, input_shapes):
        self.tf_Ps = tf.constant(self.Ps, dtype=tf.int32)
        self.tf_Ms = tf.constant(self.Ms, dtype=tf.int32)
        
    def call(self, b_info):
        
        #P = tf.constant(self.P, dtype=tf.int8)
        #b_info = tf.expand_dims(b_info, axis=1)
        #b_r = tf.math.floormod(tf.matmul(b_info, P), 2)
        #b_r = tf.squeeze(b_r, axis=1)
        

        b_r = tf.gather(b_info, self.tf_Ps, axis=1)
        b_r = b_r * self.tf_Ms
        b_r = tf.reduce_sum(b_r, axis=2)
        b_r = tf.math.floormod(b_r, 2)
        return b_r