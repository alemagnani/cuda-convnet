from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from data import DataProvider
import numpy as np
import random as nr

class ScikitDataProvider(DataProvider):
    def __init__(self, data, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False, fraction_test=0.01):
        if batch_range == None:
            raise Exception('the range is empty')
        if init_batchnum is None or init_batchnum not in batch_range:
            init_batchnum = batch_range[0]

        self.data_dir = None
        self.batch_range = batch_range
        self.curr_epoch = init_epoch
        self.curr_batchnum = init_batchnum
        self.dp_params = dp_params
        self.batch_meta = None
        self.data_dic = None
        self.test = test
        self.batch_idx = batch_range.index(init_batchnum)


        self.X = data[0]
        self.y = data[1]
        self.fraction_test = fraction_test
        if self.y is not None:
            print 'data is: {}, X shape {}, y shape {}'.format(len(data), self.X.shape,self.y.shape)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.fraction_test,
                                                                                random_state=42)
        else:
            print 'data is: {}, X shape {}'.format(len(data), self.X.shape)
            self.X_train, self.X_test = train_test_split(self.X,
                                                                                test_size=self.fraction_test,
                                                                                random_state=42)
            self.y_train = np.array([0] * self.X_train.shape[0],dtype=np.float32)
            self.y_test = np.array([0] * self.X_test.shape[0],dtype=np.float32)





    def get_batch(self, batch_num):
        if batch_num == 1:

            out = [expand(self.X_train), adjust_labels(self.y_train)]
        else:
            out = [expand(self.X_test), adjust_labels(self.y_test)]
            #print 'bartch shape x: {} , y: {}'.format(out[0].shape, out[1].shape)
        return out

    def get_data_dims(self, idx):
        return self.X.shape[1] if idx == 0 else 1


    def get_num_classes(self):
        return max(self.y) + 1

def adjust_labels(labels):
    n = labels.shape[0]
    out = np.require(labels.reshape((1, n)), dtype=np.float32, requirements='C')
    #print 'shape of labels is {}'.format(out.shape)
    return out


def crop_flip_image_data(x, border_size=4, image_size=64, flip_image=True):
    inner_size = image_size - 2 * border_size
    #print 'image size: {}, border_size: {}, inner size: {}'.format(image_size, border_size, inner_size)
    #print 'image size: {}, border_size: {}, inner size: {}'.format(image_size, border_size, inner_size)

    y = x.reshape(3, image_size, image_size, x.shape[1])
    out = np.zeros(((3 * inner_size * inner_size), x.shape[1]))
    #print 'size of out for reshaping: {}'.format(out.shape)

    for c in xrange(x.shape[1]): # loop over cases
        startY, startX = nr.randint(0,border_size*2 ), nr.randint(0,border_size*2 )
        endY, endX = startY + inner_size, startX + inner_size
        pic = y[:,startY:endY,startX:endX, c]
        #print 'pic shape: {}'.format(pic.shape)
        if flip_image:
            if nr.randint(0,1) == 0: # also flip the image with 50% probability
                #print 'flipping image'
                pic = pic[:,:,::-1]
        out[:,c] = pic.reshape((3 * inner_size * inner_size,))
    return out

def expand(matrix, crop_image=False, border_size=4, image_size=64,flip_image=True):
    if isinstance(matrix, csr_matrix):
        check_correct_indeces(matrix)
        rows, cols = matrix.shape
        #print "matrix output has size {}, {}".format(cols,rows)

        return [np.require(matrix.data, dtype=np.float32, requirements='C'),np.require( matrix.indices, dtype=np.int32, requirements='C'), np.require(matrix.indptr, dtype=np.int32, requirements='C'), cols, rows]
    else:
        #print 'working with a dense matrix'
        x = matrix.T
        if crop_image:
            x = crop_flip_image_data(x,border_size=border_size,image_size=image_size,flip_image=flip_image)
        out =  np.require(x, dtype=np.float32, requirements='C')
        #print 'the shape is {}'.format(out.shape)
        return out

def slice(batch_data, idx):
    if  isinstance(batch_data,list):
        csrm = csr_matrix( (batch_data[0],batch_data[1],batch_data[2]), shape=(batch_data[4],batch_data[3]))
        csrm = csrm[idx,:]
        #print 'output size {}'.format(csrm.shape)
        return expand(csrm)
    else:
        print 'batch_data shape {}, max idx: {}'.format(batch_data.shape, np.max(idx))
        out =  np.require(batch_data[:,idx], requirements='C')
        #print 'output size {}'.format(out.shape)
        return out


def check_correct_indeces(X):

    ind = X.indices
    ptr = X.indptr
    data = X.data

    for k in xrange(len(ptr)-1):
        s = set()
        begin = ptr[k]
        end = ptr[k+1]

        idx   = np.argsort(ind[begin:end])
        ind[begin:end] = ind[begin:end][idx]
        data[begin:end] = data[begin:end][idx]


        #print 'k: {}, begin: {}, end: {}'.format(k, begin,end)
        if end < begin:
            print 'beging in wrong order with end ptr: {}'.format(ptr)
            exit(-1)

        previous_ni = -1
        for j in range(begin,end):
            ni = ind[j]
            if ni < previous_ni:
                print 'indeces ont in order ni: {}, previous: {}, j: {}'.format(ni, previous_ni, j)
                print ind[begin:end]
                exit(-1)
            else:
                previous_ni = ni
            if ni in s:
                print 'problem with repeating indeces'
                exit(-1)
            else:
                s.add(ni)