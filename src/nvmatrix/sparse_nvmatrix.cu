
#include <sparse_nvmatrix.cuh>
#include <cusparse_v2.h>
#include "nvmatrix.cuh"
#include "cuda_setup.cuh"

SparseNVMatrix::SparseNVMatrix() {
	_sparseInd = NULL;
	_sparsePtr = NULL;

	_ownsDataInd = true;
	_ownsDataPtr = true;
	_nzz = 0;
	_sparse_type = SparseMatrix::CSC;

}

SparseNVMatrix::~SparseNVMatrix() {
	if (_ownsDataInd && _numElements > 0) {
		cublasStatus status = cublasFree(_sparseInd);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on delete _sparseInd\n");
			exit(EXIT_FAILURE);
		}
	}
	if (_ownsDataPtr && _numElements > 0) {
		cublasStatus status = cublasFree(_sparsePtr);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on delete _sparsePtr\n");
			exit(EXIT_FAILURE);
		}
	}

}

SparseNVMatrix::SparseNVMatrix(float* devData,int* sparseInd, int* sparsePtr,  int numRows, int numCols, int nzz, SparseMatrix::SPARSE_TYPE type) : NVMatrix( devData, 1, nzz,  1, false) {

	_nzz = nzz;
	_numRows =  numRows;
	_numCols = numCols;
	_numElements = _numRows * _numCols;
	_ownsDataInd = false;
	_ownsDataPtr = false;
	_sparseInd = sparseInd;
	_sparsePtr = sparsePtr;
	_isTrans = false;
	_sparse_type = type;
}

void SparseNVMatrix::copyFromHost(const SparseMatrix& hostMatrix) {
	assert(hostMatrix.get_non_zeros() == _nzz);
	assert(isSameDims(hostMatrix));
	assert(hostMatrix.get_sparse_type() == get_sparse_type());

	if (_nzz > 0) {
		checkCudaErrors(cudaMemcpy(_devData, hostMatrix.getData(),
				sizeof(float) * _nzz , cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(_sparseInd, hostMatrix.getSparseInd(),
				sizeof(int) * _nzz , cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy( _sparsePtr, hostMatrix.getSparsePtr(),
				sizeof(int) * (getNumCols()+1) , cudaMemcpyHostToDevice));
		//ERRORCHECK();
	}
}


NVMatrix& SparseNVMatrix::sliceCols(int startCol, int endCol) const{
	cout << "slicing cols with no target\n\n";
	if (_sparse_type == SparseMatrix::CSR){
		throw string("CSR is not supported for column slicing");
	}

	cout << "start col "<< startCol << " endcol: " << endCol << " rows: " << getNumRows() << " cols: " << getNumCols() <<"\n";
	int begin=0;
	int* d_answer;
	checkCudaErrors(cudaMalloc(&d_answer, sizeof(int)));

	read_one_entry<<<1,1>>>(_sparsePtr,startCol, d_answer);
	checkCudaErrors(cudaMemcpy(&begin, d_answer, sizeof(int), cudaMemcpyDeviceToHost));

	int end=0;
	read_one_entry<<<1,1>>>(_sparsePtr,endCol, d_answer);
	checkCudaErrors(cudaMemcpy(&end, d_answer, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_answer));
	//cudaThreadSynchronize();
	cout << "begin" << begin <<"\n";
	cout << "end" << end <<"\n";
	const int nzz = end -begin;
	cout << "the slice of sparse as nzz " <<  nzz <<"\n";
	return * new SparseNVMatrix(_devData+begin,_sparseInd +begin, _sparsePtr+ begin,  getNumRows(), (endCol-startCol), nzz, SparseMatrix::CSC);
}


void SparseNVMatrix::copyFromHost(const Matrix& hostMatrix) {
	copyFromHost((SparseMatrix&) hostMatrix);
}
void SparseNVMatrix::copyFromHost(const Matrix& hostMatrix,
		bool resizeDeviceMatrix) {
	if (resizeDeviceMatrix) {
		resize((SparseMatrix&) hostMatrix);
	}
	copyFromHost(hostMatrix);

}


bool SparseNVMatrix::resize(const SparseMatrix &like) {
	bool reallocated = false;
	if (like.get_non_zeros() != _nzz) {
		assert(_ownsData);
		assert(_ownsDataInd);
		assert(_ownsDataPtr);

		_numRows = like.getNumRows();
		_numCols = like.getNumCols();

		if (_nzz > 0) { // free old memory
			cublasStatus status = cublasFree(_devData);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error during resize: %X\n", status);
				exit(EXIT_FAILURE);
			}
			status = cublasFree(_sparseInd);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error during resize: %X\n", status);
				exit(EXIT_FAILURE);
			}
			status = cublasFree(_sparsePtr);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! memory free error during resize: %X\n", status);
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
			status = cublasAlloc(_nzz, sizeof(int), (void**) &_sparseInd);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! device memory allocation error\n");
				exit(EXIT_FAILURE);
			}
			status = cublasAlloc( (like.getNumCols() + 1), sizeof(int),
					(void**) &_sparsePtr);
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf(stderr, "!!!! device memory allocation error\n");
				exit(EXIT_FAILURE);
			}

		} else {
			_devData = NULL;
			_sparseInd = NULL;
			_sparsePtr = NULL;
			_nzz = 0;
		}
		reallocated = true;

		_numRows =  like.getNumRows();
		_numCols = like.getNumCols();
		_numElements = _numRows * _numCols;
		_sparse_type = like.get_sparse_type();
		_isTrans = false;

	}
	return true;
}


/*
 * Does SOFT transpose and returns result, leaving this matrix unchanged
 */
NVMatrix& SparseNVMatrix::getTranspose(){
	return * new SparseNVMatrix(_devData,_sparseInd, _sparsePtr,   getNumRows(), getNumCols(), _nzz, (get_sparse_type() == SparseMatrix::CSR) ? SparseMatrix::CSC : SparseMatrix::CSR );
}

/*
 * Does HARD transpose and puts result in target
 */
void SparseNVMatrix::transpose(NVMatrix& target){
	cout << "sparse not implemented transpose with target args\n";
	throw string("Not implemented!");
}
/*
 * Does SOFT transpose
 */
void SparseNVMatrix::transpose(){
	if (_sparse_type == SparseMatrix::CSC){
		_sparse_type = SparseMatrix::CSR;
	}else{
		_sparse_type = SparseMatrix::CSC;
	}
	int numColsTmp = getNumCols();
	_numCols = getNumRows();
	_numRows = numColsTmp;
}

bool SparseNVMatrix::transpose(bool trans){
	bool oldTrans = get_sparse_type() == SparseMatrix::CSR;
	    if (oldTrans != trans) {
	        transpose();
	    }
	    return oldTrans;
}
void SparseNVMatrix::rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const{
	addProductChanged(b, 0, scaleAB, target);
}


void SparseNVMatrix::addProductChanged( const NVMatrix &b, float scaleTarget, float scaleAB, NVMatrix &target)const{
	cout << "addproduct changes Sparse\n";
	assert(_numCols == b.getNumRows());
	if(scaleTarget == 0.0) {
		target.resize(_numRows, b.getNumCols());
		target.setTrans(true);
	}else{
		assert(target.isTrans());
	}
	assert(target.getNumRows() == _numRows);
	assert(target.getNumCols() == b.getNumCols());



	if (_sparse_type == SparseMatrix::CSC){
		cusparseStatus_t cusparseStatus = cusparseScsrmm2(cudaSetup::_cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE ,b.isTrans()?CUSPARSE_OPERATION_NON_TRANSPOSE: CUSPARSE_OPERATION_TRANSPOSE,
				getNumCols(), b.getNumCols(), getNumRows(),_nzz,
				&scaleAB, cudaSetup::_sparseDescr,
				getDevData(), _sparsePtr, _sparseInd,
				b.getDevData(),  b.getLeadingDim() ,
				&scaleTarget,
				target.getDevData(), getNumRows());
		checkCudaErrors(cusparseStatus);
	}else{
		cusparseStatus_t cusparseStatus = cusparseScsrmm2(cudaSetup::_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE ,b.isTrans()?CUSPARSE_OPERATION_NON_TRANSPOSE: CUSPARSE_OPERATION_TRANSPOSE,
				getNumRows(), b.getNumCols(), getNumCols(),_nzz,
				&scaleAB, cudaSetup::_sparseDescr,
				getDevData(), _sparsePtr, _sparseInd,
				b.getDevData(),  b.getLeadingDim() ,
				&scaleTarget,
				target.getDevData(), getNumRows());
		checkCudaErrors(cusparseStatus);
	}

	cout << "done addproduct changes Sparse\n";

}

void SparseNVMatrix::rightMult(const NVMatrix &b, float scaleAB){
	throw string("rightMult Not implemented for sparse!");
}


