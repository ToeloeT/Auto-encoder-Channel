import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import numpy as np

class BP_Decoder(Layer):
    
    def __init__(self, H, nb_iter, **kwargs):
        super(BP_Decoder, self).__init__(**kwargs)

        self.nb_iter = nb_iter
        self.atanh_clip_value = 1e-7
        self.H = H
        self.cn_count, self.code_length = H.shape
        
        self.edges = np.stack(np.where(H==1),axis=1) #edges are ordered by check nodes
        self.num_edges = self.edges.shape[0]
        self.max_cn_deg = int(np.max(np.sum(H, axis=1)))
        self.max_vn_deg = int(np.max(np.sum(H, axis=0)))
        self.cn_msg_ind = self.get_msg_ind('cn')
        self.vn_msg_ind = self.get_msg_ind('vn')
        self.vn2cn_ind = self.get_vn2cn_ind()
        self.cn_mask_ind = self.get_mask_ind('cn')
        self.vn_mask_ind = self.get_mask_ind('vn')

    def get_msg_ind(self, node_type):
        if node_type=='cn':
            num, max_deg, axis = self.cn_count, self.max_cn_deg, 0
        elif node_type=='vn':
            num, max_deg, axis = self.code_length, self.max_vn_deg, 1
        else: raise('Unknown node type')
        ind = np.zeros(shape=[num, max_deg])
        for i in range(num):
            tmp = np.nonzero(self.edges[:,axis]==i)[0]
            tmp = np.concatenate([tmp, self.num_edges*np.ones([max_deg-tmp.shape[0]])])
            ind[i,:]=tmp
        ind = np.reshape(ind, [-1])
        return ind.astype('int32')
    
    def get_vn2cn_ind(self):
        msg = np.arange(0,self.num_edges+1)
        msg = np.take(msg, self.vn_msg_ind)
        msg = msg[np.where(msg!=self.num_edges)]
        ind = np.argsort(msg)
        return ind
    
    def get_mask_ind(self, node_type):
        if node_type=='cn':
            num, max_deg, ind = self.cn_count, self.max_cn_deg, self.cn_msg_ind
        elif node_type=='vn':
            num, max_deg, ind = self.code_length, self.max_vn_deg, self.vn_msg_ind
        else: raise('Unknown node type')
        mask = np.arange(0,self.num_edges+1)
        mask = np.take(mask, ind, axis=0)
        mask_ind = np.nonzero(mask!=self.num_edges)[0]
        return mask_ind
    
    def msg2mat(self, msg, node_type):
        if node_type=='cn':
            ind, max_deg, num, nan_val = self.cn_msg_ind, self.max_cn_deg, self.cn_count, 1.0
        elif node_type=='vn':
            ind, max_deg, num, nan_val = self.vn_msg_ind, self.max_vn_deg, self.code_length, 0.0
        else: raise('Unknown node type')   
        msg = tf.concat([msg, nan_val*tf.ones([tf.shape(msg)[0], 1])], axis=1)
        mat = tf.gather(msg, ind, axis=1)
        mat = tf.reshape(mat, [-1, num, max_deg])
        return mat 
    
    def mat2msg(self, mat, node_type):
        if node_type=='cn':
            dims, mask_ind = self.cn_count*self.max_cn_deg, self.cn_mask_ind
        elif node_type=='vn':
            dims, mask_ind = self.code_length*self.max_vn_deg, self.vn_mask_ind
        else: raise('Unknown node type')  
        msg = tf.reshape(mat, shape=[-1, dims])
        msg = tf.gather(msg, mask_ind, axis=1)
        if node_type=='vn':
            msg = tf.gather(msg, self.vn2cn_ind, axis=1)
        return msg
    
    def check_to_var(self, vc):
        cv = tf.tanh(vc/2)
        cv = self.msg2mat(cv, 'cn')
        
        CVs = []
        for i in range(self.max_cn_deg):
            CVs.append(tf.concat([cv[:,:,:i], cv[:,:,i+1:]], axis=2))
        cv = tf.stack(CVs, axis=2)
        cv = tf.reduce_prod(cv, axis=3)
        #cv = tf.reduce_prod(cv, axis=2, keepdims=True)/(cv)
        
        cv_sign = tf.stop_gradient(tf.sign(cv))
        cv = cv - cv_sign*self.atanh_clip_value
        #cv = tf.clip_by_value(cv, clip_value_min=-self.atanh_clip_value, clip_value_max=self.atanh_clip_value)
        
        cv = 2*tf.atanh(cv)
        cv = self.mat2msg(cv, 'cn')
        return cv 
    
    def init_var_to_check(self, llr):
        return tf.gather(llr, self.edges[:,1], axis=1)

    def compute_llr_decoder(self, llr, cv):
        vc = self.msg2mat(cv, 'vn')
        vc = tf.reduce_sum(vc, axis=2)+llr
        return vc
        
    def var_to_check(self, cv, llr):
        vc = self.msg2mat(cv, 'vn')
        vc = (tf.reduce_sum(vc, axis=2, keepdims=True)+tf.expand_dims(llr, axis=2))-vc
        vc = self.mat2msg(vc, 'vn')
        return vc 

    def build(self, input_shape):
        pass

    def call(self, llr_demapper):
        
        llr_demapper = -llr_demapper
        
        if self.nb_iter == 0:
            return llr_demapper
        
        def stop_decode(llr_decoder, cv, i):
            return tf.less(i, self.nb_iter)
        
        """
        H = tf.cast(self.H, tf.int32)
        def stop_decode(llr_decoder, cv, i):
            bh = (-tf.sign(llr_decoder) + 1) / 2
            bh = tf.cast(bh, tf.int32)
            bh = tf.expand_dims(tf.squeeze(bh, axis=0), axis=1)
            c = tf.matmul(H, bh)
            c = tf.math.floormod(c, 2)
            c = tf.reduce_sum(c)
            return tf.math.logical_and(tf.less(i, self.nb_iter), tf.not_equal(c, 0))
        """
        
        def decode_it(llr_decoder, cv, i):            
            vc = self.var_to_check(cv, llr_demapper)
            cv = self.check_to_var(vc)
            llr_decoder = self.compute_llr_decoder(llr_demapper, cv)
            i = i + 1
            return llr_decoder, cv, i
        
        i = tf.constant(0)
        vc = self.init_var_to_check(llr_demapper)
        cv = self.check_to_var(vc)
        llr_decoder = self.compute_llr_decoder(llr_demapper, cv)
        
        llr_decoder, _, i = tf.while_loop(stop_decode, decode_it, [llr_decoder, cv, i])
        return -llr_decoder
    
    def get_valid_codewords(self, llr):
        
        H = tf.cast(self.H, tf.int32)
        H = tf.tile(tf.expand_dims(H, axis=0), [tf.shape(llr)[0], 1, 1])
        
        # Hard decision on bits
        bh = (tf.sign(llr) + 1) / 2
        bh = tf.cast(bh, tf.int32)
        
        # Parity check
        bh = tf.expand_dims(bh, axis=2)
        c = tf.matmul(H, bh)
        c = tf.squeeze(c, axis=2)
        c = tf.math.floormod(c, 2)
        c = tf.reduce_sum(c, axis=1)
        
        # Select valid codewords
        valid_cw = tf.where(tf.equal(c, 0))
        
        return valid_cw