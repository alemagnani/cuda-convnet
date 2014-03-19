/*
 * sparse_nvmatrix.cuh
 *
 *  Created on: Mar 16, 2014
 *      Author: alessandro
 */

#ifndef SPARSE_NVMATRIX_CUH_
#define SPARSE_NVMATRIX_CUH_



#include <nvmatrix.cuh>
#include <csc_matrix.h>
#include "sparse_nvmatrix_kernels.cuh"
#include "nvmatrix_kernels.cuh"


#include <iostream>

using namespace std;


class SparseNVMatrix : public NVMatrix {
protected:
	 int* _sparseInd;
	 int* _sparsePtr;

	 int _nzz;

	 bool _ownsDataInd;
	 bool _ownsDataPtr;

	  Matrix::SPARSE_TYPE _sparse_type;

public:



	 SparseNVMatrix();

    ~SparseNVMatrix();

    SparseNVMatrix(float* devData,int* sparseInd, int* sparsePtr,  int numRows, int numCols, int nzz, Matrix::SPARSE_TYPE type);

    bool resize(const SparseNVMatrix &like);

    void copyFromHost(const SparseMatrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix);

    inline SPARSE_TYPE get_sparse_type() const {
     	return _sparse_type;
     }


    /*
     * Does SOFT transpose and returns result, leaving this matrix unchanged
     */
    virtual NVMatrix& getTranspose();

    /*
     * Does HARD transpose and puts result in target
     */
    virtual void transpose(NVMatrix& target);

    /*
     * Does SOFT transpose
     */
    virtual void transpose();
    virtual bool transpose(bool trans);


    virtual void rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const;

    virtual void rightMult(const NVMatrix &b, float scaleAB);

    virtual void addProductChanged( const NVMatrix &b, float scaleTarget, float scaleAB, NVMatrix &target) const;


    virtual NVMatrix& sliceCols(int startCol, int endCol) const;

    void sliceCols(int startCol, int endCol, NVMatrix& target) const {

    	if (_sparse_type == Matrix::CSR){
    		throw string("CSR is not supported for column slicing");
    	}

    	endCol = endCol < 0 ? this->_numCols : endCol;
    	_checkBounds(0, getNumRows(), startCol, endCol);

    	 if (target.getNumRows() != getNumRows() || target.getNumCols() != (endCol - startCol)) {
    	        target.resize(getNumRows(), (endCol - startCol));
    	 }

    	target.apply(NVMatrixOps::Zero());


    	dim3 blocks(std::min(NUM_BLOCKS_MAX, endCol-startCol));
    	dim3 threads(ELTWISE_THREADS_X);

    	slice_kernel<<<blocks,threads>>>(startCol, endCol, _devData, _sparseInd, _sparsePtr, target.getDevData(), target.getStride(), !target.isTrans() );
    }

    inline Matrix::MATRIX_TYPE get_type() const {
           	return Matrix::SPARSE;
    }

    void sliceRows(int startRow, int endRow, NVMatrix& target) const {
    	cout << "performing slice rows on a CSC nvmatrix";
    	exit(-1);
    }


};







#endif /* SPARSE_NVMATRIX_CUH_ */
