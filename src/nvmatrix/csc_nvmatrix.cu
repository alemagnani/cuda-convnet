
#include <csr_nvmatrix.cuh>
#include <csc_nvmatrix.cuh>
#include <cusparse_v2.h>


CscNVMatrix::CscNVMatrix() {
	setCusparseHandle();
	_cscRowInd = NULL;
	_cscColPtr = NULL;

	_ownsDataRowInd = true;
	_ownsDataColPtr = true;
	_nzz = 0;

}

CscNVMatrix::~CscNVMatrix() {
	if (_ownsDataRowInd && _numElements > 0) {
		cublasStatus status = cublasFree(_cscRowInd);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on _cscRowInd\n");
			exit(EXIT_FAILURE);
		}
	}
	if (_ownsDataColPtr && _numElements > 0) {
		cublasStatus status = cublasFree(_cscColPtr);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on _cscColPtr\n");
			exit(EXIT_FAILURE);
		}
	}

}

CscNVMatrix::CscNVMatrix(float* devData,int* cscRowInd, int* cscColPtr,  int numRows, int numCols, int nzz) : NVMatrix( devData, 1, nzz,  1, false) {
	setCusparseHandle();
	_nzz = nzz;
	_numRows =  numRows;
	_numCols = numCols;
	_numElements = _numRows * _numCols;
	_ownsDataRowInd = false;
	_ownsDataColPtr = false;
	_cscRowInd = cscRowInd;
	_cscColPtr = cscColPtr;
	_isTrans = false;
}

void CscNVMatrix::copyFromHost(const CscMatrix& hostMatrix) {
	assert(hostMatrix.get_non_zeros() == _nzz);
	assert(isSameDims(hostMatrix));
	setTrans(hostMatrix.isTrans());

	if (_nzz > 0) {
		cudaMemcpy(_devData, hostMatrix.getData(),
				sizeof(float) * _nzz , cudaMemcpyHostToDevice);

		cudaMemcpy(_cscRowInd, hostMatrix.getRowInd(),
				sizeof(int) * _nzz , cudaMemcpyHostToDevice);

		cudaMemcpy( _cscColPtr, hostMatrix.getColPtr(),
				sizeof(int) * (getNumCols()+1) , cudaMemcpyHostToDevice);
		//ERRORCHECK();
	}
}


NVMatrix& CscNVMatrix::sliceCols(int startCol, int endCol) const{
	const int begin = _cscColPtr[startCol];
	const int end = _cscColPtr[endCol];
	const int nzz = end -begin;
	return * new CscNVMatrix(_devData+begin,_cscRowInd +begin, _cscColPtr+ begin,  getNumRows(), (endCol-startCol), nzz);
}


void CscNVMatrix::copyFromHost(const Matrix& hostMatrix) {
	copyFromHost((CscMatrix&) hostMatrix);
}
void CscNVMatrix::copyFromHost(const Matrix& hostMatrix,
		bool resizeDeviceMatrix) {
	if (resizeDeviceMatrix) {
		resize((CscMatrix&) hostMatrix);
	}
	copyFromHost(hostMatrix);

}




bool CscNVMatrix::resize(const CscMatrix &like) {
	bool reallocated = false;
	if (like.get_non_zeros() != _nzz) {
		assert(_ownsData);
		assert(_ownsDataRowInd);
		assert(_ownsDataColPtr);

		_numRows = like.getNumRows();
		_numCols = like.getNumCols();

		if (_nzz > 0) { // free old memory
			cublasStatus status = cublasFree(_devData);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error: %X\n", status);
				exit(EXIT_FAILURE);
			}
			status = cublasFree(_cscRowInd);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error: %X\n", status);
				exit(EXIT_FAILURE);
			}
			status = cublasFree(_cscColPtr);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error: %X\n", status);
				exit(EXIT_FAILURE);
			}

		}
		_nzz = like.get_non_zeros();
		if (_nzz > 0) { // allocate new memory
			printf("allocating new memory\n");
			cublasStatus status = cublasAlloc(_nzz, sizeof(float),
					(void**) &_devData);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! device memory allocation error\n");
				exit(EXIT_FAILURE);
			}
			status = cublasAlloc(_nzz, sizeof(int), (void**) &_cscRowInd);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! device memory allocation error\n");
				exit(EXIT_FAILURE);
			}
			status = cublasAlloc( (like.getNumCols() + 1), sizeof(int),
					(void**) &_cscColPtr);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! device memory allocation error\n");
				exit(EXIT_FAILURE);
			}

		} else {
			_devData = NULL;
			_cscRowInd = NULL;
			_cscColPtr = NULL;
			_nzz = 0;
		}
		reallocated = true;

		_numRows =  like.getNumRows();
		_numCols = like.getNumCols();
		_numElements = _numRows * _numCols;
		_isTrans = false;

	}
	return true;
}


/*
 * Does SOFT transpose and returns result, leaving this matrix unchanged
 */
NVMatrix& CscNVMatrix::getTranspose(){
	return * new CsrNVMatrix(_devData,_cscRowInd, _cscColPtr,  getNumCols(), getNumRows(), _nzz);
}

/*
 * Does HARD transpose and puts result in target
 */
void CscNVMatrix::transpose(NVMatrix& target){
	throw string("Not implemented!");
}
/*
 * Does SOFT transpose
 */
void CscNVMatrix::transpose(){
	throw string("Not implemented!");
}
bool CscNVMatrix::transpose(bool trans){
	throw string("Not implemented!");
}
void CscNVMatrix::rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const{
	addProductChanged(b, 0, scaleAB, target);
}




void CscNVMatrix::addProductChanged( const NVMatrix &b, float scaleTarget, float scaleAB, NVMatrix &target)const{
	assert(_numCols == b.getNumRows());
	if(&target != this) {
		target.resize(_numRows, b.getNumCols());
		target.setTrans(true);
	}
	assert(target.getNumRows() == _numRows);
	assert(target.getNumCols() == b.getNumCols());

	target.resize(_numRows, b.getNumCols());
	target.setTrans(true);

	cusparseStatus_t cusparseStatus = cusparseScsrmm2(NVMatrix::getCusparseHandle(), CUSPARSE_OPERATION_TRANSPOSE ,CUSPARSE_OPERATION_TRANSPOSE,
			getNumCols(), b.getNumCols(), getNumRows(),_nzz,
			&scaleAB, CscNVMatrix::getDescription(),
			getDevData(), _cscColPtr, _cscRowInd,
			b.getDevData(),  b.getLeadingDim() ,
			&scaleTarget,
			target.getDevData(), getNumRows());

	checkCudaErrors(cusparseStatus);

}

void CscNVMatrix::rightMult(const NVMatrix &b, float scaleAB){
	throw string("rightMult Not implemented for CSC!");
}


