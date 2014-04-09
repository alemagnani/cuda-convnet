cimport numpy as np


cdef extern from "math.h":
    cdef extern float sqrt(float x)


cdef class WeightVector_s(object):
    cdef np.ndarray w
    cdef float *w_data_ptr
    cdef float wscale
    cdef int n_classes
    cdef int n_features
    cdef int stride_row
    cdef int stride_col

    # y <- alpha * W *x + beta *y
    cdef void gemv(self,float alpha, float *x_data_ptr, int *x_ind_ptr,
                    int xnnz, float beta, float* y) nogil

    # W <- alpha * x * y ^T + W
    cdef void ger(self,float alpha, float *x_data_ptr, int *x_ind_ptr,
                    int xnnz, float* y) nogil

    # W <- alpha * W
    cdef void scale(self, float c) nogil
    cdef void reset_wscale(self) nogil

    cdef void copy(self, float *data) nogil
    cdef void copy_in(self, float *data) nogil