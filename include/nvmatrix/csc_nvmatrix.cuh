/*
 * csc_nvmatrix.cuh
 *
 *  Created on: Mar 3, 2014
 *      Author: alessandro
 */

#ifndef CSC_NVMATRIX_CUH_
#define CSC_NVMATRIX_CUH_

#include <nvmatrix.cuh>
#include <csc_matrix.h>
#include "sparse_nvmatrix_kernels.cuh"
#include "nvmatrix_kernels.cuh"

#include <iostream>

using namespace std;


class CscNVMatrix : public NVMatrix {
protected:
	 int* _cscRowInd;
	 int* _cscColPtr;

	 int _nzz;

	 bool _ownsDataRowInd;
	 bool _ownsDataColPtr;

public:
	 CscNVMatrix();

    ~CscNVMatrix();

    bool resize(const CscMatrix &like);

    void copyFromHost(const CscMatrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix);

    void sliceCols(int startCol, int endCol, NVMatrix& target) const {
    	cout << "performing slice cols on a CSC nvmatrix, target rows: "<< target.getNumRows() <<" target cols:  "<< target.getNumCols()<<" target type: "<< target.get_type() << "\n";
    	cout << "performing slice cols on a CSC nvmatrix, this rows: "<< getNumRows() <<"  cols:  "<< getNumCols()<<"  type: "<< get_type() << ""\n";
    	target.scale(0.0);
    	cout << "done scaling of target\n";
    	ERRORCHECK();
    	dim3 blocks(std::min(NUM_BLOCKS_MAX, endCol-startCol));
    	dim3 threads(ELTWISE_THREADS_X);
    	cout << "starting slice kernel\n";
    	slice_kernel<<<blocks,threads>>>(startCol, endCol, _devData, _cscRowInd, _cscColPtr, target.getDevData(), target.getStride(), !target.isTrans() );
    	cout << "done slice kernel\n";
    }

    inline Matrix::MATRIX_TYPE get_type() const {
           	return Matrix::CSC;
    }

    void sliceRows(int startRow, int endRow, NVMatrix& target) const {
    	cout << "performing slice rows on a CSC nvmatrix";
    }


};




#endif /* CSC_NVMATRIX_CUH_ */
