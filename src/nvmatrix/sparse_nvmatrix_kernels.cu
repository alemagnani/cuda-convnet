

#include "sparse_nvmatrix_kernels.cuh"

__global__  void slice_kernel(int start, int end, float * data, int* ind, int* ptr, float * dest, int stride_dest, bool isTrans){
	const int size = end - start;
	for (int row = blockIdx.x; row < size; row += gridDim.x){
		const int begin = ptr[row];
		const int num_entries = ptr[row+1] - begin;
		for (int pos = threadIdx.x; pos < num_entries; pos += blockDim.x){
			if (isTrans){
				dest[row + stride_dest * ind[pos]] = data[pos];
			}else{
				dest[row * stride_dest + ind[pos]] = data[pos];
			}
		}
	}
}
