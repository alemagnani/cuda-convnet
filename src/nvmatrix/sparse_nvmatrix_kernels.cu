

#include "sparse_nvmatrix_kernels.cuh"

__global__  void slice_kernel(int start, int end, float * data, int* ind, int* ptr, float * dest, int stride_dest, bool isTrans){
	const int size = end - start;


	for (int i = blockIdx.x; i < size; i += gridDim.x){
		const int begin = ptr[i+start];
		const int num_entries = ptr[i+1+start] - begin;
		for (int pos = threadIdx.x; pos < num_entries; pos += blockDim.x){
			const int k = pos+begin;
			if (isTrans){
				dest[i + stride_dest * ind[k]] = data[k];
			}else{
				dest[i * stride_dest + ind[k]] = data[k];
			}
		}
	}
}
