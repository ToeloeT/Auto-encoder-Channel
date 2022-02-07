import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Subframe(Layer):

    def __init__(self, pilot_alloc_fn, **kwargs):
        super(Subframe, self).__init__(**kwargs)

        # In the file, each row is assumed to correspond to a subcarrier
        # The separators are columns ','
        # Each entry is a 1/0 depending on the symbol being a pilot/data
        # The number of entries per row should be the same for all rows (number of OFDM symbols)
        f = open(pilot_alloc_fn, 'r')
        lines = f.readlines()
        f.close()
        P = []
        i = 0
        for l in lines:
            i = i + 1
            P.append([int(i) for i in l.strip().split(',')])
        P = tf.constant(np.array(P), tf.float32)
        
        # Building pilots to add
        self.P = tf.complex(P*tf.sqrt(0.5), P*tf.sqrt(0.5))
        
        self.pil_loc = tf.cast(tf.where(tf.equal(P, 1.0)), tf.int32)
        self.dat_loc = tf.cast(tf.where(tf.equal(P, 0.0)), tf.int32)
        self.nb_data = tf.shape(self.dat_loc)[0]
        self.block_length = tf.shape(self.P)[1]
        self.nb_subc = tf.shape(self.P)[0]
    
    def bit_per_subf(self, bit_per_cu):
        return bit_per_cu*self.nb_data

    # x is of shape [batch size, data symbol per subframe]
    def call(self, x):
        
        tf.debugging.Assert(tf.equal(tf.shape(x)[1], self.nb_data), [x])
        
        batch_size = tf.shape(x)[0]
        s = tf.tile(tf.expand_dims(self.P, axis=0), [batch_size, 1, 1])
        
        ind_dat = tf.tile(tf.expand_dims(self.dat_loc, axis=0), [batch_size, 1, 1])
        batch_ind = tf.range(batch_size)
        batch_ind = tf.expand_dims(tf.tile(tf.expand_dims(batch_ind, axis=1), [1, self.nb_data]), axis=2)
        ind_dat = tf.concat([batch_ind, ind_dat], axis=2)

        s = tf.tensor_scatter_nd_update(s, ind_dat, x)
        return s
    
    # Like call, but with pilot squared norm instead of pilots
    # Usefull for IDD
    def call_abs2(self, x):
        
        tf.debugging.Assert(tf.equal(tf.shape(x)[1], self.nb_data), [x])
        
        batch_size = tf.shape(x)[0]
        s = tf.tile(tf.expand_dims(self.P, axis=0), [batch_size, 1, 1])
        s = tf.square(tf.abs(s))
        s = tf.complex(s, 0.0)
        
        ind_dat = tf.tile(tf.expand_dims(self.dat_loc, axis=0), [batch_size, 1, 1])
        batch_ind = tf.range(batch_size)
        batch_ind = tf.expand_dims(tf.tile(tf.expand_dims(batch_ind, axis=1), [1, self.nb_data]), axis=2)
        ind_dat = tf.concat([batch_ind, ind_dat], axis=2)

        s = tf.tensor_scatter_nd_update(s, ind_dat, x)
        return s

    # y is of shape [batch_size, nb_subcarrier, block length, bit_per_cu]
    def extract_data(self, y):
        
        tf.debugging.Assert(tf.equal(tf.shape(y)[2], self.block_length), [y])
        tf.debugging.Assert(tf.equal(tf.shape(y)[1], self.nb_subc), [y])
        
        batch_size = tf.shape(y)[0]
        
        ind_dat = tf.tile(tf.expand_dims(self.dat_loc, axis=0), [batch_size, 1, 1])
        batch_ind = tf.range(batch_size)
        batch_ind = tf.expand_dims(tf.tile(tf.expand_dims(batch_ind, axis=1), [1, self.nb_data]), axis=2)
        ind_dat = tf.concat([batch_ind, ind_dat], axis=2)
        
        sy = tf.gather_nd(y, ind_dat)
        
        return sy