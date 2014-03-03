
#include <csr_nvmatrix.cuh>
#include <csc_nvmatrix.cuh>




CscNVMatrix::CscNVMatrix() {
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


void CscNVMatrix::copyFromHost(const CscMatrix& hostMatrix) {
	assert(hostMatrix.get_non_zeros() == _nzz);
	assert(isSameDims(hostMatrix));
	setTrans(hostMatrix.isTrans());

	ERRORCHECK();
	if (_nzz > 0) {
		 cudaMemcpy(_devData, hostMatrix.getData(),
				sizeof(float) * _nzz , cudaMemcpyHostToDevice);
		 ERRORCHECK();

		 cudaMemcpy(_cscRowInd, hostMatrix.getRowInd(),
				sizeof(int) * _nzz , cudaMemcpyHostToDevice);
		 ERRORCHECK();

		 cudaMemcpy( _cscColPtr, hostMatrix.getColPtr(),
				sizeof(int) * (getNumCols()+1) , cudaMemcpyHostToDevice);
		 ERRORCHECK();
	}
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

	}
	return true;
}
