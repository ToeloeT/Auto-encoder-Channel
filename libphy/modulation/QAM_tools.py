import numpy as np

def gen_labeling(bit_per_cu):
    
        def genCode(n):
                if n == 0:
                    return ['']

                code1 = genCode(n-1)
                code2 = []
                for codeWord in code1:
                    code2 = [codeWord] + code2

                for i in range(len(code1)):
                    code1[i] += '0'
                for i in range(len(code2)):
                    code2[i] += '1'
                return code1 + code2  

        k = bit_per_cu // 2 # bits per dimension
        syms = 2**k
        m = syms**2
        # gray labeling
        gray_labeling = []
        for i in range(2**k):
            gray_labeling.append(np.array([int(e) for e in  genCode(k)[i]]))
        gray_labeling = np.array(gray_labeling)

        encoding_table_256_qam = gray_labeling[:,::-1]

        encoding_table_256_qam = np.hstack( [np.kron(encoding_table_256_qam,np.ones((syms,1))), np.tile(encoding_table_256_qam,(syms,1)) ])

        natural_number = np.sum(2**np.arange(2*k)[::-1] * encoding_table_256_qam,axis=1)
        table_gray = np.zeros((m,m))
        for bvec in range(m):
            table_gray[bvec,np.where(natural_number==bvec)[0]] =1

        return table_gray

def gen_const(bit_per_cu):
    
    points_per_dim = 2**(bit_per_cu // 2)
    
    A = np.arange(0, points_per_dim)
    A = A - np.mean(A)
    
    C = []
    for i in A:
        for j in A:
            C.append([i,j])
    C = np.array(C)
    
    L = gen_labeling(bit_per_cu)
    L = np.argmax(L, axis=1)
    C = C[L]
    
    return C