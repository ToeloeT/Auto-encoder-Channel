import tensorflow as tf
from tensorflow.keras.layers import Layer
from scipy.integrate import quad
import numpy as np
import pickle


# Should be normalized such that the PSD, defined by P(f) = (1/W) |H(f/W)|^2, H(f) is the Fourier transform of the filter.
# This is equivalent to having h(t) that integrate to one when T. For sinc, this is already the case, so there is nothing to do.

class StepWiseTrainablePulseShaping(Layer):
        
    # bandwidth [real valued > 0, (1,)] : Bandwidth (not including excess bandwidth) [Hz]
    #  Typically, W = 1/T, where T is the sampling period.
    # duration [real valued > 0, (1,)]: fitler time duration [s]. The filter is time contained to (-duration/2, duration/2).
    # M [odd integer, (1,)]: number of steps
    #
    # To make calculations more efficient, it is assumed that (i) duration = dT, where d is a positive integer, an M = qd, where
    #  q is a positive integer.
    # block_size: size of a block of symbols that form a batch example. Needed to do some pre-computation for computing the noise
    #  correlation matrix Cholesky decomposition
    def __init__(self, bandwidth, duration, M, block_size, **kwargs):
        super(StepWiseTrainablePulseShaping, self).__init__(**kwargs)
        
        self.bandwidth = bandwidth
        self.duration = duration
        self.M = M
        self.block_size = block_size
        
        tf.debugging.Assert(tf.equal(tf.math.mod(M, 2), 1), M, "Number of steps must be odd")
                
        # Pre-compute some coefficients to speed-up ACLR calculation
        self.init_calc()
        
    def build(self, input_shape):
        
        # Transmit filter
        self.W_tx = tf.Variable(tf.random_normal_initializer()(shape=[self.M], dtype=tf.float32), trainable=True)
        # Receiver filter
        self.W_rx = tf.Variable(tf.random_normal_initializer()(shape=[self.M], dtype=tf.float32), trainable=True)
        
    def save_model(self, fn):
        W = self.get_weights()
        with open(fn, 'wb') as f:
            pickle.dump(W, f)

    def load_model(self, fn):
        with open(fn, 'rb') as f:
            W = pickle.load(f)
        self.set_weights(W)
    
    # Normalize a set of weights W such that the corresponding step funtion has unit energy
    def normalize_weights(self, W):
        c = self.duration/tf.cast(self.M, tf.float32)
        total_en = c * tf.reduce_sum(tf.square(W))
        W = W / tf.sqrt(total_en)
        return W
    
    def init_calc(self):
        
        M = self.M.numpy()
        D = self.duration.numpy()
        W = self.bandwidth.numpy()
        N = self.block_size.numpy()
        
        # Pre-compute the matrix required for the ACLR calculation
        
        def sinc_square(f):
            c = D/M
            return np.square(np.sinc(c*f))
        
        def A_l_real(f, l):
            c = D/M
            v = sinc_square(f)*np.cos(2*np.pi*c*l*f)
            return v
        
        def A_l_imag(f, l):
            c = D/M
            v = sinc_square(f)*np.sin(2*np.pi*c*l*f)
            return v
        
        Ls = np.arange(-M+1, M)
        a_r = np.flip(np.array([quad(A_l_real, 0.0, W*0.5, args=(l,))[0] for l in Ls]))
        a_i = np.flip(np.array([quad(A_l_imag, 0.0, W*0.5, args=(l,))[0] for l in Ls]))
        a = a_r + 1j*a_i
        A = []
        for i in range(M):
            A.append(np.roll(a, i))
        A = np.array(A)
        A = A[:,M-1:]
        self.A = tf.constant(A, tf.complex64)
        
        # Pre-compute the matrix required for noise correlation calculation
        # It is assumed that D is a multiple of 1/W, i.e., D = d/W for some integer d,
        #  and that M is a multiple of d, i.e., M = dq for some integer q.
        d = int(D*W)
        q = int(M/d)
        # Under these assumptions, matrices B(l) are sparse and coefficients equal 0 or 1.
        # Moreover, each row contains at most a single one, and for |l| > d, B(l) = 0
        # We construct structure for efficient computation of Bx, and therefore calculation of
        #  x^T B(l) x for some l.
        Bs_gather = []
        Bs_scatter = []
        Bs_size = []
        for l in np.arange(-d+1, d):
            Bg = []
            Bs = []
            for n in np.arange(M):
                k = n-q*l
                if (k < M) and (k >= 0):
                    Bg.append(k)
                    Bs.append(n)
            s = len(Bs)
            pad = M - s
            Bs = np.concatenate([Bs, np.zeros(pad)])
            Bg = np.concatenate([Bg, np.zeros(pad)])
            Bs_gather.append(Bg)
            Bs_scatter.append(Bs)
            Bs_size.append(s)
        self.Bs_gather = tf.constant(np.array(Bs_gather), tf.int32)
        self.Bs_scatter = tf.constant(np.array(Bs_scatter), tf.int32)
        self.Bs_size = tf.constant(np.array(Bs_size), tf.int32)
        
        # Fourier matrix
        w = np.exp(1j*2*np.pi/N)
        F = np.array([[np.power(w, k*j) for k in np.arange(N)] for j in np.arange(N)])
        self.F = tf.constant(F, tf.complex64)
    
    def get_Wtx(self):
        Wtx = self.W_tx
        Wtx = self.normalize_weights(Wtx)
        return Wtx
    
    def get_Wrx(self):
        Wrx = self.W_rx
        Wrx = self.normalize_weights(Wrx)
        return Wrx
    
    # Compute W1^T B(l) W2 for some indices Ls
    def compute_B_quad_product(self, W1, W2, l):
        c = self.duration/tf.cast(self.M, tf.float32)
        d = tf.cast(self.duration*self.bandwidth, tf.int32)
        v = tf.cast(0.0, tf.float32)
        if tf.less(tf.abs(l), d):            
            s = self.Bs_size[l+d-1]
            Bg = self.Bs_gather[l+d-1][:s]
            Bs = tf.expand_dims(self.Bs_scatter[l+d-1][:s], axis=1)
            g = tf.gather(W2, Bg)
            v = tf.scatter_nd(Bs, g, [self.M])
            v = tf.reduce_sum(W1*v)
        return v*c

    # It is assumed that size at least equals duration*bandwidth
    # Generate the Cholesky decomposition matrix of the noise covariance
    def gen_cov_cholesky(self):
                        
        # We use here the approximation that because size is large compared to d, the correlation matrix can be approximated
        #  by a circulant one.
        d = int(self.duration.numpy()*self.bandwidth.numpy())
        W_rx = self.get_Wrx()
        bs = tf.zeros([d], tf.float32)
        for l in range(d):
            v = self.compute_B_quad_product(W_rx, W_rx, l)
            bs = tf.tensor_scatter_nd_update(bs, [[l]], [v])
        bs = tf.concat([bs, tf.zeros(self.block_size - 2*d + 1), tf.reverse(bs[1:], axis=[0])], axis=0)
        bs = tf.complex(bs, 0.0)
        # Compute the eigenvalues
        F = self.F
        bs = tf.expand_dims(bs, axis=0)
        ev = tf.reduce_sum(bs*F, axis=1)
        # Square root of eigen values
        ev_sqrt = tf.complex(tf.sqrt(tf.math.real(ev)), 0.0)
        # Compute Cholesky decomposition
        L = tf.matmul(F, tf.matmul(tf.linalg.diag(ev_sqrt), tf.transpose(F, conjugate=True)))/tf.complex(tf.cast(self.block_size, tf.float32), 0.0)
        
        return L
    
    # L [integer, ()]: desired number of taps
    # It is assumed that L at least equals 2*d, where d = duration*bandwidth.
    # It is assumed that L-d is even (so L is odd)
    @tf.function
    def call(self, L):
        
        W_tx = self.get_Wtx()
        W_rx = self.get_Wrx()
                
        d =  int(self.duration.numpy()*self.bandwidth.numpy())
        a = tf.zeros([2*d-1], tf.float32)
        for l in range(-d+1, d, 1):
            v = self.compute_B_quad_product(W_tx, W_rx, l)
            a = tf.tensor_scatter_nd_update(a, [[l+d-1]], [v])
        pad = (L-2*d+1)//2
        a = tf.concat([tf.zeros(pad), a, tf.zeros(pad)], axis=0)
        a = tf.complex(a, 0.0)
        
        return a
    
    @tf.function
    def total_energy(self):
        return tf.constant(1.0, tf.float32)
    
    @tf.function
    def aclr(self):
        
        c = self.duration/tf.cast(self.M, tf.float32)
        Wtx = self.get_Wtx()
        Wtx = tf.expand_dims(tf.complex(Wtx, 0.0), 1)
        in_band_en = tf.math.real(tf.squeeze(tf.matmul(tf.matmul(tf.transpose(Wtx), self.A), Wtx), axis=1))[0]
        in_band_en = 2.*tf.square(c)*in_band_en
        
        aclr = self.total_energy()/in_band_en - self.total_energy()
        return aclr

    """
    # Just for debugging
    def ft_squared(self, f):
        c = self.duration/tf.cast(self.M, tf.float32)
        c = c.numpy()
        ws = self.normalize_weights(self.W_tx).numpy()
        
        v = c*np.sinc(c*f)
        
        M = self.M.numpy()
        Ms = np.arange(-(M-1)//2, (M-1)//2+1)
        es = np.exp(-1j*2*np.pi*c*Ms*f)
        v = v*np.sum(es*ws)
        
        v = np.square(np.abs(v))
        
        return v
    """