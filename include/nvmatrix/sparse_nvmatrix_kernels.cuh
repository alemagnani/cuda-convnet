/*
 * sparse_nvmatrix_kernels.cuh
 *
 *  Created on: Mar 1, 2014
 *      Author: alessandro
 */

#ifndef SPARSE_NVMATRIX_KERNELS_CUH_
#define SPARSE_NVMATRIX_KERNELS_CUH_

__global__  void slice_kernel(int start, int end, float * data, int* ind, int* ptr, float * dest, int stride_dest, bool isTrans);

#endif /* SPARSE_NVMATRIX_KERNELS_CUH_ */
