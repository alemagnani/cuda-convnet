# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
import sys
from time import time

cimport cython
from libc.math cimport exp, log, sqrt, pow, fabs
cimport numpy as np
cdef extern from "numpy/npy_math.h":
    bint isfinite "npy_isfinite"(double) nogil

from weight_vector cimport WeightVector_s

cdef extern from "cblas.h":
    void sscal "cblas_sscal"(int, float, float *, int) nogil
    void saxpy "cblas_saxpy"( int ,  float ,  float *, int , float *,  int ) nogil
    void scopy "cblas_scopy"( int ,  float *,  int , float *,  int ) nogil


cdef inline int int_max(int a, int b): return a if a >= b else b


np.import_array()

DEF OPTIMAL = 2


def entropy_max_sgd_s(np.ndarray[float, ndim=2, mode='c'] weights,
              np.ndarray[float, ndim=1, mode='c'] intercepts,
              np.ndarray[float, ndim=1, mode='c'] xd,
              np.ndarray[int, ndim=1, mode='c'] xind,
              np.ndarray[int, ndim=1, mode='c'] xindptr,
              np.ndarray[int, ndim=1, mode='c'] y,

              np.ndarray[float, ndim=1, mode='c'] x_weights,

              np.ndarray[float, ndim=1, mode='c'] xd_valid,
              np.ndarray[int, ndim=1, mode='c'] xind_valid,
              np.ndarray[int, ndim=1, mode='c'] xindptr_valid,
              np.ndarray[int, ndim=1, mode='c'] y_valid,

              np.ndarray[float, ndim=1, mode='c'] x_weights_valid,

              int n_features,
              int n_samples, int n_classes, int n_points_valid,
              float alpha,
              int n_iter,
              int verbose,
              int learning_rate, float eta0,
              float t=1.0,
              float intercept_decay=1.0,
              int early_stop=0,
              float improvement_threshold=0.95,
              float patience_increase=1.4,
              int patience=0):

    cdef WeightVector_s w = WeightVector_s(weights)

    # helper variable

    cdef float eta = 0.0
    cdef float p = 0.0
    cdef float update = 0.0

    cdef unsigned int count = 0
    cdef unsigned int epoch = 0
    cdef unsigned int row = 0

    cdef float * x_row_data_ptr
    cdef int * x_row_ind_ptr
    cdef int row_xnnz
    cdef unsigned int offset
    cdef int n_w_col = n_classes -1

    cdef float *x_data = <float *>xd.data
    cdef int *x_ind = <int *> xind.data
    cdef int *x_ind_ptr = <int *> xindptr.data

    cdef int * y_ptr = <int *> y.data

    cdef float * x_weights_ptr = NULL
    cdef int weigths_available = 0
    if x_weights is not None:
        x_weights_ptr = <float *> x_weights.data
        weigths_available = 1



    cdef np.ndarray[float, ndim = 1, mode = "c"] xW = np.zeros((n_w_col,), dtype=np.float32, order="c")
    cdef float * xW_data_ptr = <float * > xW.data


    #validation data ########################
    cdef unsigned int row_validation = 0
    cdef float *x_data_valid
    cdef int *x_ind_valid
    cdef int *x_ind_ptr_valid

    cdef int * y_ptr_valid

    cdef float * x_weights_ptr_valid = NULL

    cdef float m_valid
    cdef int guessed_class

    cdef float best_accuracy = -1
    cdef float correct_predictions
    cdef float total_predictions
    cdef float accuracy_validation

    cdef np.ndarray[float, ndim = 1, mode = "c"] xW_valid = None

    cdef np.ndarray[float, ndim = 2, mode = "c"] best_w =  None
    cdef np.ndarray[float, ndim = 1, mode = "c"] best_intercepts = None

    cdef float *best_w_ptr = NULL
    cdef float *best_intercepts_ptr = NULL


    cdef float * xW_data_ptr_valid = NULL

    cdef int gi

    if early_stop == 1:
        x_data_valid = <float *>xd_valid.data
        x_ind_valid = <int *> xind_valid.data
        x_ind_ptr_valid = <int *> xindptr_valid.data
        y_ptr_valid = <int *> y_valid.data

        xW_valid = np.zeros((n_w_col,), dtype=np.float32, order="c")
        xW_data_ptr_valid = <float * > xW_valid.data

        best_w = np.zeros((n_features,n_w_col), dtype=np.float32, order="c")
        best_intercepts = np.zeros((n_w_col,), dtype=np.float32, order="c")

        best_w_ptr = <float *>best_w.data
        best_intercepts_ptr = <float *>best_intercepts.data

        if x_weights_valid is not None:
            x_weights_ptr_valid = <float *> x_weights_valid.data


        print 'done creating variables for early stop'
    ##########################################

    cdef np.ndarray[float, ndim = 1, mode = "c"] derivative = np.zeros((n_classes,), dtype=np.float32, order="c")
    cdef float * derivative_data_ptr = <float * > derivative.data

    cdef float * intercepts_ptr = <float * > intercepts.data

    cdef float m = 0.0
    cdef float s = 0.0
    cdef float v = 0.0
    cdef unsigned int k
    cdef float weight
    cdef unsigned int re
    cdef float u = 0.0

    eta = eta0

    t_start = time()
    for epoch in range(n_iter):
        if verbose > 0 and epoch % 20 == 0 and early_stop != 1:
            sys.stdout.write("\repoch %i" % epoch)
            sys.stdout.flush()

        for row in range(n_samples):
            offset = x_ind_ptr[row]
            x_row_data_ptr = x_data + offset
            x_row_ind_ptr = x_ind + offset
            row_xnnz  = x_ind_ptr[row+1] - offset

            scopy(n_w_col, intercepts_ptr,1,xW_data_ptr,1)
            w.gemv(1.0, x_row_data_ptr, x_row_ind_ptr,
                    row_xnnz, 1, xW_data_ptr)

            if learning_rate == OPTIMAL:
                eta = 1.0 / (alpha * t)

            #L2 update
            w.scale(1.0 - eta * alpha)

            # computing  minus derivative
            m = 0.0
            for k in range(n_w_col):
                if xW_data_ptr[k] > m:
                    m = xW_data_ptr[k]
            s = exp(-m)
            for k in range(n_w_col):
                v = exp(xW_data_ptr[k] - m)
                xW_data_ptr[k] = v
                s += v

            sscal(n_w_col,-1.0/s,xW_data_ptr,1)

            re = y_ptr[row]
            if re < n_w_col:
                xW_data_ptr[re] += 1

            # done  minus derivative

            if weigths_available == 0:
                weight = 1.0
            else:
                weight = x_weights_ptr[row]

            saxpy(n_w_col,intercept_decay * eta * weight, xW_data_ptr, 1, intercepts_ptr,1 )

            w.ger(eta * weight, x_row_data_ptr, x_row_ind_ptr,
                    row_xnnz, xW_data_ptr)

            t += 1
            count += 1

        if epoch % 10 == 0 and early_stop == 1:
            total_predictions = 0.0
            correct_predictions = 0.0
            for row_validation in range(n_points_valid):
                offset = x_ind_ptr_valid[row_validation]
                x_row_data_ptr = x_data_valid + offset
                x_row_ind_ptr = x_ind_valid + offset
                row_xnnz  = x_ind_ptr_valid[row_validation+1] - offset

                scopy(n_w_col, intercepts_ptr,1, xW_data_ptr_valid,1)
                w.gemv(1.0, x_row_data_ptr, x_row_ind_ptr,
                        row_xnnz, 1, xW_data_ptr_valid)
                m_valid = 0.0
                guessed_class = n_w_col
                for gi in range(n_w_col):
                    if xW_data_ptr_valid[gi] > m_valid:
                        guessed_class = gi
                        m_valid = xW_data_ptr_valid[gi]

                if x_weights_ptr_valid == NULL:
                    total_predictions += 1.0
                else:
                    total_predictions += x_weights_ptr_valid[row_validation]

                if guessed_class == y_ptr_valid[row_validation]:
                    if x_weights_ptr_valid == NULL:
                        correct_predictions += 1.0
                    else:
                        correct_predictions += x_weights_ptr_valid[row_validation]
            accuracy_validation = correct_predictions  / total_predictions

            if verbose > 0:
                sys.stdout.write('\nepoch %i, sample processed %i, validation accuracy %f %%, patience %d, best accuracy %f %%\n' % (
                        epoch, (count + 1), accuracy_validation * 100., patience, best_accuracy * 100.))
                sys.stdout.flush()

            if  accuracy_validation * improvement_threshold >= best_accuracy:
                best_accuracy = accuracy_validation
                scopy(n_w_col,intercepts_ptr,1,best_intercepts_ptr,1)
                w.copy(best_w_ptr)
                patience = int_max(patience, <int>(count * patience_increase))
                if verbose > 0:
                    sys.stdout.write('\r\tpatience updated to %d, iteration %d' % (patience, count))
                    sys.stdout.flush()

            if patience <= count:
                if verbose > 0:
                    print '\ndone because patience %d, iteration %d' % (patience, count)
                    break



    w.reset_wscale()

    if early_stop == 1:
        scopy(n_w_col,best_intercepts_ptr,1, intercepts_ptr,1)
        w.copy_in(best_w_ptr)
        print '\nOptimization complete with best accuracy score of %f %%\n' % (best_accuracy * 100.)
    print 'The code run for %d epochs, with %f epochs/sec\n' % (
            epoch, 1. * epoch / (time() - t_start + 0.0001))

    return weights, intercepts


