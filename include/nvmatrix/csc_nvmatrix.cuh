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
#include <cusparse_v2.h>


#include <iostream>

using namespace std;


class CscNVMatrix : public NVMatrix {
protected:
	 int* _cscRowInd;
	 int* _cscColPtr;

	 int _nzz;

	 bool _ownsDataRowInd;
	 bool _ownsDataColPtr;
	 static cusparseMatDescr_t _descr;

public:
	 CscNVMatrix();

    ~CscNVMatrix();

    CscNVMatrix(float* devData,int* cscRowInd, int* cscColPtr,  int numRows, int numCols, int nzz);

    bool resize(const CscMatrix &like);

    void copyFromHost(const CscMatrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix);

    static cusparseMatDescr_t getDescription(){
    	if (_descr == NULL){
    			cusparseStatus_t cusparseStatus = cusparseCreateMatDescr(&_descr);
    			checkCudaErrors(cusparseStatus);
    			cusparseSetMatType(_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    			cusparseSetMatIndexBase(_descr,CUSPARSE_INDEX_BASE_ZERO);
    		}
    		return _descr;
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
    	//cout << "performing slice cols on a CSC nvmatrix, target rows: "<< target.getNumRows() <<" target cols:  "<< target.getNumCols()<<" target type: "<< target.get_type() << "\n";
    	//cout << "performing slice cols on a CSC nvmatrix, this rows: "<< getNumRows() <<"  cols:  "<< getNumCols()<<"  type: "<< get_type() << "\n";

    	endCol = endCol < 0 ? this->_numCols : endCol;
    	_checkBounds(0, getNumRows(), startCol, endCol);

    	 if (target.getNumRows() != getNumRows() || target.getNumCols() != (endCol - startCol)) {
    	        target.resize(getNumRows(), (endCol - startCol));
    	 }

    	target.apply(NVMatrixOps::Zero());


    	dim3 blocks(std::min(NUM_BLOCKS_MAX, endCol-startCol));
    	dim3 threads(ELTWISE_THREADS_X);

    	slice_kernel<<<blocks,threads>>>(startCol, endCol, _devData, _cscRowInd, _cscColPtr, target.getDevData(), target.getStride(), !target.isTrans() );
    }

    inline Matrix::MATRIX_TYPE get_type() const {
           	return Matrix::CSC;
    }

    void sliceRows(int startRow, int endRow, NVMatrix& target) const {
    	cout << "performing slice rows on a CSC nvmatrix";
    	exit(-1);
    }


};




#endif /* CSC_NVMATRIX_CUH_ */
