/*
 * csr_nvmatrix.cuh
 *
 *  Created on: Feb 21, 2014
 *      Author: alessandro
 */

#ifndef CSR_NVMATRIX_CUH_
#define CSR_NVMATRIX_CUH_

#include <nvmatrix.cuh>
#include <csr_matrix.h>
#include "sparse_nvmatrix_kernels.cuh"
#include "nvmatrix_kernels.cuh"

#define ERRORCHECK() cErrorCheck(__FILE__, __LINE__)

inline void cErrorCheck(const char *file, int line) {
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    printf(" @ %s: %d\n", file, line);
    exit(-1);
  }
}

class CsrNVMatrix : public NVMatrix {
protected:
	 int* _csrColIndA;
	 int* _csrRowPtrA;

	 int _nzz;

	 bool _ownsDataColInd;
	 bool _ownsDataRowInd;

public:
	 CsrNVMatrix();

    ~CsrNVMatrix();

    bool resize(const CsrMatrix &like);

    void copyFromHost(const CsrMatrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix);

    void sliceRows(int startRow, int endRow, NVMatrix& target) const {
    	target.scale(0.0);
    	dim3 blocks(std::min(NUM_BLOCKS_MAX, endRow-startRow));
    	dim3 threads(ELTWISE_THREADS_X);
    	slice_kernel<<<blocks,threads>>>(startRow,endRow, _devData, _csrColIndA, _csrRowPtrA, target.getDevData(), target.getStride(), target.isTrans() );
    }

    inline Matrix::MATRIX_TYPE get_type() const {
               	return Matrix::CSR;
        }

};


#endif /* CSR_NVMATRIX_CUH_ */
