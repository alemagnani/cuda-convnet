#include <csr_nvmatrix.cuh>
#include <cusparse_v2.h>
#include "nvmatrix.cuh"
#include "cuda_setup.cuh"


CsrNVMatrix::CsrNVMatrix() {

	_csrColIndA = NULL;
	_csrRowPtrA = NULL;

	_ownsDataColInd = true;
	_ownsDataRowInd = true;
	_nzz = 0;

}

CsrNVMatrix::CsrNVMatrix(float* devData,int* csrColInd, int* csrRowPtr,  int numRows, int numCols, int nzz): NVMatrix( devData, 1, nzz, 1, false) {

	_nzz = nzz;
	_numRows =  numRows;
	_numCols = numCols;
	_numElements = _numRows * _numCols;
	_ownsDataColInd = false;
	_ownsDataRowInd = false;
	_csrColIndA = csrColInd;
	_csrRowPtrA = csrRowPtr;
	_isTrans = false;
}


CsrNVMatrix::~CsrNVMatrix() {
	if (_ownsDataColInd && _numElements > 0) {
		cublasStatus status = cublasFree(_csrColIndA);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on _csrColIndA\n");
			exit(EXIT_FAILURE);
		}
	}
	if (_ownsDataRowInd && _numElements > 0) {
		cublasStatus status = cublasFree(_csrRowPtrA);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on _csrRowPtrA\n");
			exit(EXIT_FAILURE);
		}
	}

}


void CsrNVMatrix::copyFromHost(const CsrMatrix& hostMatrix) {
	assert(hostMatrix.get_non_zeros() == _nzz);
	assert(isSameDims(hostMatrix));
	setTrans(hostMatrix.isTrans());

	if (_nzz > 0) {
		 cudaMemcpy(_devData, hostMatrix.getData(),
				sizeof(float) * _nzz , cudaMemcpyHostToDevice);
		 //ERRORCHECK();

		 cudaMemcpy( _csrColIndA, hostMatrix.getColInd(),
				sizeof(int) * _nzz , cudaMemcpyHostToDevice);
		 //ERRORCHECK();

		 cudaMemcpy(_csrRowPtrA, hostMatrix.getRowPtr(),
				sizeof(int) * (getNumRows()+1) , cudaMemcpyHostToDevice);
		 //ERRORCHECK();
	}
}


void CsrNVMatrix::copyFromHost(const Matrix& hostMatrix) {
	copyFromHost((CsrMatrix&) hostMatrix);
}
void CsrNVMatrix::copyFromHost(const Matrix& hostMatrix,
		bool resizeDeviceMatrix) {
	if (resizeDeviceMatrix) {
		resize((CsrMatrix&) hostMatrix);
	}
	copyFromHost(hostMatrix);

}




bool CsrNVMatrix::resize(const CsrMatrix &like) {
	bool reallocated = false;
	if (like.get_non_zeros() != _nzz) {
		assert(_ownsData);
		assert(_ownsDataColInd);
		assert(_ownsDataRowInd);

		_numRows = like.getNumRows();
		_numCols = like.getNumCols();

		if (_nzz > 0) { // free old memory
			cublasStatus status = cublasFree(_devData);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error: %X\n", status);
				exit(EXIT_FAILURE);
			}
			status = cublasFree(_csrColIndA);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error: %X\n", status);
				exit(EXIT_FAILURE);
			}
			status = cublasFree(_csrRowPtrA);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error: %X\n", status);
				exit(EXIT_FAILURE);
			}

		}
		_nzz = like.get_non_zeros();
				if (_nzz > 0) { // allocate new memory
					cublasStatus status = cublasAlloc(_nzz, sizeof(float),
							(void**) &_devData);
					if (status != CUBLAS_STATUS_SUCCESS) {
						fprintf(stderr, "!!!! device memory allocation error\n");
						exit(EXIT_FAILURE);
					}
					status = cublasAlloc(_nzz, sizeof(int), (void**) &_csrColIndA);
					if (status != CUBLAS_STATUS_SUCCESS) {
						fprintf(stderr, "!!!! device memory allocation error\n");
						exit(EXIT_FAILURE);
					}
					status = cublasAlloc( like.getNumRows() + 1, sizeof(int),
							(void**) &_csrRowPtrA);
					if (status != CUBLAS_STATUS_SUCCESS) {
						fprintf(stderr, "!!!! device memory allocation error\n");
						exit(EXIT_FAILURE);
					}

				} else {
					_devData = NULL;
					_csrColIndA = NULL;
					_csrRowPtrA = NULL;
					_nzz = 0;
				}
		reallocated = true;

		_numRows =  like.getNumRows();
		_numCols = like.getNumCols();
		_numElements = _numRows * _numCols;

	}
	return true;
}

/*
 * Does SOFT transpose and returns result, leaving this matrix unchanged
 */
NVMatrix& CsrNVMatrix::getTranspose(){
	throw string("Not implemented!");
}

/*
 * Does HARD transpose and puts result in target
 */
void CsrNVMatrix::transpose(NVMatrix& target){
	throw string("Not implemented!");
}
/*
 * Does SOFT transpose
 */
void CsrNVMatrix::transpose(){
	throw string("Not implemented!");
}
bool CsrNVMatrix::transpose(bool trans){
	throw string("Not implemented!");
}

void CsrNVMatrix::rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const{
	cout << "right mul CSR\n";
	addProductChanged(b, 0, scaleAB, target);
}




void CsrNVMatrix::addProductChanged( const NVMatrix &b, float scaleTarget, float scaleAB, NVMatrix &target)const{
	cout << "addproduct changed CSR\n";
	assert(_numCols == b.getNumRows());
	if(&target != this) {
		target.resize(_numRows, b.getNumCols());
		target.setTrans(true);
	}
	assert(target.getNumRows() == _numRows);
	assert(target.getNumCols() == b.getNumCols());

	target.resize(_numRows, b.getNumCols());
	target.setTrans(true);

	cusparseStatus_t cusparseStatus = cusparseScsrmm2(cudaSetup::_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE ,CUSPARSE_OPERATION_TRANSPOSE,
			getNumRows(), b.getNumCols(), getNumCols(),_nzz,
			&scaleAB, cudaSetup::_sparseDescr,
			getDevData(), _csrRowPtrA, _csrColIndA,
			b.getDevData(),  b.getLeadingDim() ,
			&scaleTarget,
			target.getDevData(), getNumRows());

	checkCudaErrors(cusparseStatus);

}
