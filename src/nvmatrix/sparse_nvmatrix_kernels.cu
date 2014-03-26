

#include "sparse_nvmatrix_kernels.cuh"
#include <stdio.h>
#include <math.h>
#include <cusparse_v2.h>
#include <helper_cuda.h>
#include "nvmatrix.cuh"

#include <iostream>

using namespace std;

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
	printf("checking sparse matrix!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
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

	printf("DONE checking sparse matrix\n");

}

__global__ void read_one_entry(int* array, int pos_to_read, int * read_value){
	read_value[0] = array[pos_to_read];
}

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
		int ldc){

	/*
	for(int i =0; i < n; i++){
		cout << "col " << i << "\n";
		cusparseStatus_t cusparseStatus = cusparseScsrmv( handle, transa,
				m,  k,  nnz,        alpha,
				descrA,
				csrValA,
				csrRowPtrA, csrColIndA,
				B + i * ldb , beta,
				C + i * ldc);
		checkCudaErrors(cusparseStatus);
	}
	 */

	float* csc_data;
	int* csc_ind;
	int* csc_ptr;

	cublasStatus status = cublasAlloc(nnz, sizeof(float),
			(void**) &csc_data);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device memory allocation error\n");
		exit(EXIT_FAILURE);
	}
	status = cublasAlloc(nnz, sizeof(int), (void**) &csc_ind);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device memory allocation error\n");
		exit(EXIT_FAILURE);
	}
	status = cublasAlloc(k+1, sizeof(int), (void**) &csc_ptr);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device memory allocation error\n");
		exit(EXIT_FAILURE);
	}

	cusparseStatus_t cusparseStatus = cusparseScsr2csc(handle, m,  k, nnz,
			csrValA, csrRowPtrA,
			csrColIndA, csc_data,
			csc_ind, csc_ptr,
			CUSPARSE_ACTION_NUMERIC,
			CUSPARSE_INDEX_BASE_ZERO);

	checkCudaErrors(cusparseStatus);




	cusparseStatus = cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE ,
			k, n, m , nnz,
			alpha,descrA,
			csc_data, csc_ptr, csc_ind,
			B,  ldb ,
			beta,
			C, ldc);
	checkCudaErrors(cusparseStatus);

	cudaThreadSynchronize();

	cublasStatus status2 = cublasFree(csc_data);
	if (status2 != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! memory free error on delete _sparsePtr\n");
		exit(EXIT_FAILURE);
	}
	status2 = cublasFree(csc_ind);
	if (status2 != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! memory free error on delete _sparsePtr\n");
		exit(EXIT_FAILURE);
	}
	status2 = cublasFree(csc_ptr);
	if (status2 != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! memory free error on delete _sparsePtr\n");
		exit(EXIT_FAILURE);
	}



}

