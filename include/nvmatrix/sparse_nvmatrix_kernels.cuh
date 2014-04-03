/*
 * sparse_nvmatrix_kernels.cuh
 *
 *  Created on: Mar 1, 2014
 *      Author: alessandro
 */

#ifndef SPARSE_NVMATRIX_KERNELS_CUH_
#define SPARSE_NVMATRIX_KERNELS_CUH_

#include <cusparse_v2.h>

__global__  void slice_kernel(int start, int end, float * data, int* ind, int* ptr, float * dest, int stride_dest, bool isTrans);

__global__ void read_one_entry(int* array, int pos_to_read, int * read_value);

__global__ void check_matrix(float * data, int* ind, int* ptr, int nzz, int size, int size2);

__global__ void check_matrix_dense(float * data, int row, int col);

__global__  void sparse_mul_trans(float alpha, int m, int n, int k,const float * data,const int* ind,const int* ptr,const float* B, float* C);

void sparseMult(cusparseHandle_t handle,
        cusparseOperation_t transa,
        int m,
        int n,
        int k,
        int nnz,
        const float *alpha,
        const cusparseMatDescr_t descrA,
        const float *csrValA,
        const int *csrRowPtrA,
        const int *csrColIndA,
        const float *B,
        int ldb,
        const float *beta,
        float *C,
        int ldc);


#endif /* SPARSE_NVMATRIX_KERNELS_CUH_ */
