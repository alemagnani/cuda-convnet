# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#

from libc.limits cimport INT_MAX
from libc.math cimport sqrt
import numpy as np
cimport numpy as np

cdef extern from "cblas.h":
    void sscal "cblas_sscal"(int, float, float *, int) nogil
    void saxpy "cblas_saxpy"( int ,  float ,  float *, int , float *,  int ) nogil
    void scopy "cblas_scopy"( int ,  float *,  int , float *,  int ) nogil


np.import_array()


cdef class WeightVector_s(object):
    """Dense matrix represented by a scalar and a numpy array.
    """


    def __cinit__(self, np.ndarray[float, ndim=2, mode='c'] w):
        cdef float *wdata = <float *>w.data

        if w.shape[0] > INT_MAX:
            raise ValueError("More than %d features not supported; got %d."
                             % (INT_MAX, w.shape[0]))
        self.w = w
        self.w_data_ptr = wdata
        self.wscale = 1.0
        self.n_features = w.shape[0]
        self.n_classes = w.shape[1]
        self.stride_row = <int> w.strides[0] / w.itemsize
        self.stride_col = <int> w.strides[1] / w.itemsize

    # y <- alpha x * W + beta *y
    cdef void gemv(self,float alpha, float *x_data_ptr, int *x_ind_ptr,
                    int xnnz, float beta, float* y) nogil:
        cdef int j
        cdef int idx
        cdef float val
        cdef int n_classes = self.n_classes
        cdef int offset

        cdef float wscale = self.wscale
        cdef float* w_data_ptr = self.w_data_ptr
        cdef int stride_col = self.stride_col
        cdef int stride_row = self.stride_row

        sscal(<int>(self.n_classes), beta, y, 1)

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            offset = idx * stride_row
            saxpy(n_classes,alpha * val * wscale, w_data_ptr + offset, stride_col, y, 1 )



    # W <- alpha * x * y ^T + W
    cdef void ger(self,float alpha, float *x_data_ptr, int *x_ind_ptr,
                    int xnnz, float* y) nogil:
        cdef unsigned int j
        cdef int idx
        cdef float val
        cdef int n_classes = self.n_classes
        cdef int offset

        cdef float wscale = self.wscale
        cdef float* w_data_ptr = self.w_data_ptr
        cdef int stride_col = self.stride_col
        cdef int stride_row = self.stride_row

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            offset = idx * stride_row
            saxpy(n_classes,alpha * val/wscale, y,1, w_data_ptr + offset, stride_col )

    cdef void scale(self, float c) nogil:
        """Scales the weight vector by a constant ``c``.

        It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too
        small we call ``reset_swcale``."""
        self.wscale *= c
        if self.wscale < 1e-9:
            self.reset_wscale()

    cdef void reset_wscale(self) nogil:
        """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
        cdef float* w_data_ptr = self.w_data_ptr
        sscal(<int>(self.n_classes * self.n_features), self.wscale, w_data_ptr, 1)
        self.wscale = 1.0

    cdef void copy(self, float *d) nogil:
        cdef float* w_data_ptr = self.w_data_ptr
        scopy(<int>(self.n_classes * self.n_features), w_data_ptr,1, d, 1)
        sscal(<int>(self.n_classes * self.n_features), self.wscale, d, 1)

    cdef void copy_in(self, float *d) nogil:
        cdef float* w_data_ptr = self.w_data_ptr
        scopy(<int>(self.n_features * self.n_classes), d, 1 ,w_data_ptr,1)
        self.wscale = 1.0