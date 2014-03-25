/*
 * sparse_nvmatrix_kernels.cuh
 *
 *  Created on: Mar 1, 2014
 *      Author: alessandro
 */

#ifndef SPARSE_NVMATRIX_KERNELS_CUH_
#define SPARSE_NVMATRIX_KERNELS_CUH_

__global__  void slice_kernel(int start, int end, float * data, int* ind, int* ptr, float * dest, int stride_dest, bool isTrans);

__global__ void read_one_entry(int* array, int pos_to_read, int * read_value);

__global__ void check_matrix(float * data, int* ind, int* ptr, int nzz, int size, int size2);

__global__ void check_matrix_dense(float * data, int row, int col);

#endif /* SPARSE_NVMATRIX_KERNELS_CUH_ */
