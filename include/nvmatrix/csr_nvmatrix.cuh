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



class CsrNVMatrix : public NVMatrix {
protected:
	 int* _csrColIndA;
	 int* _csrRowPtrA;

	 int _nzz;

	 bool _ownsDataColInd;
	 bool _ownsDataRowInd;

	 static cusparseMatDescr_t _descr;

public:
	 CsrNVMatrix();

    ~CsrNVMatrix();

    CsrNVMatrix(float* devData,int* csrColInd, int* csrRowPtr,  int numRows, int numCols, int nzz);

    bool resize(const CsrMatrix &like);

    void copyFromHost(const CsrMatrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix);

    virtual void addProductChanged( const NVMatrix &b, float scaleTarget, float scaleAB, NVMatrix &target) const;

    virtual void rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const;

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


    void sliceRows(int startRow, int endRow, NVMatrix& target) const {

    	 if (target.getNumRows() != (endRow = startRow)|| target.getNumCols() != getNumCols()) {
    	    	        target.resize((endRow = startRow), getNumCols());
    	    	 }

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
