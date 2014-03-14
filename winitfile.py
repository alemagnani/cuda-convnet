import numpy as np
import scipy
import math
def uniform_tanh(name, idx, shape, params=None):
    print 'getting weights'
    rows, cols = shape
    out = scipy.random.random((rows,cols))
    r = math.sqrt(6.0/(rows+cols))
    out = out * (2* r) - r
    print 'weights out is shape {}'.format(out.shape)
    return np.array(out,dtype=np.single)