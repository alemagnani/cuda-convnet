#include <csr_nvmatrix.cuh>




CsrNVMatrix::CsrNVMatrix() {
	_csrColIndA = NULL;
	_csrRowPtrA = NULL;

	_ownsDataColInd = true;
	_ownsDataRowInd = true;
	_nzz = 0;

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
		 ERRORCHECK();

		 cudaMemcpy( _csrColIndA, hostMatrix.getColInd(),
				sizeof(int) * _nzz , cudaMemcpyHostToDevice);
		 ERRORCHECK();

		 cudaMemcpy(_csrRowPtrA, hostMatrix.getRowPtr(),
				sizeof(int) * (getNumRows()+1) , cudaMemcpyHostToDevice);
		 ERRORCHECK();
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
