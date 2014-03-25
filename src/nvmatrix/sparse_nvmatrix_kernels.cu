

#include "sparse_nvmatrix_kernels.cuh"
#include <stdio.h>
#include <math.h>

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

__global__ void check_matrix_dense(float * data, int row, int col){
	printf("checking dense matrix!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	for (int r =0; r < row; r++){
		for(int c=0; c < col; c++){
			float b = data[r + c * row];
			if (isnan(b)){
				printf("foud nana element\n");
				return;
			}
		}
	}
}


__global__ void check_matrix(float * data, int* ind, int* ptr, int nzz, int size, int size2){
	printf("checking sparse matrix!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	int nzz2 = ptr[size] -ptr[0];
	if (nzz2 != nzz){
		printf("the nzz sizes don't match %d, %d\n",nzz, nzz2);
		return;
	}

	for( int i=0 ; i < size; i++){
		int begin = ptr[i];
		int end = ptr[i+1];

		if (end < begin){
			printf("end before begin\n");
			return;
		}
		if (begin < 0){
			printf("begin less than 0");
			return;
		}
		if (end > nzz){
			printf("end greater than nzz, %d, %d",end, nzz);
			return;
		}

		for (int k= begin; k < end; k++){
			float b = data[k];
			if (isnan(b)){
							printf("foud nan element\n");
							return;
			}
			int pos = ind[k];
			float c = 1.2* b;
			data[k] = c;

			if (pos < 0){
				printf("the position is less than 0");
				return;
			}
			if (pos >= size2){
				printf("pos is bigger or equalt than size2");
				return;
			}

		}


	}



}

__global__ void read_one_entry(int* array, int pos_to_read, int * read_value){
	read_value[0] = array[pos_to_read];
}
