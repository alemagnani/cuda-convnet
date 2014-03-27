
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
	//cout << "sparse delete\n";
	if (_ownsDataInd && _numElements > 0) {
		cout << "freeing indeces for sparse matrix \n\n";
		cublasStatus status = cublasFree(_sparseInd);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on delete _sparseInd\n");
			exit(EXIT_FAILURE);
		}
	}
	if (_ownsDataPtr && _numElements > 0) {
		cout << "freeing pointer for sparse matrix \n\n";
		cublasStatus status = cublasFree(_sparsePtr);
		if (status != CUBLAS_STATUS_SUCCESS) {
			fprintf(stderr, "!!!! memory free error on delete _sparsePtr\n");
			exit(EXIT_FAILURE);
		}
	}

}

SparseNVMatrix::SparseNVMatrix(float* devData,int* sparseInd, int* sparsePtr,  int numRows, int numCols, int nzz, SparseMatrix::SPARSE_TYPE type) : NVMatrix( devData, 1, nzz,  1, false) {
	//cout << "new sparse matrix of size " << numRows << " , " << numCols << "\n";
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
				sizeof(int) * ( (get_sparse_type() == SparseMatrix::CSC ?getNumCols() : getNumRows())+1) , cudaMemcpyHostToDevice));

	}
}


NVMatrix& SparseNVMatrix::sliceCols(int startCol, int endCol) const{
	//cout << "slicing cols with no target\n\n";
	if (_sparse_type == SparseMatrix::CSR){
		throw string("CSR is not supported for column slicing");
	}

	//cout << "start col "<< startCol << " endcol: " << endCol << " rows: " << getNumRows() << " cols: " << getNumCols() << " nzz: "<< _nzz <<"\n";
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
	//cout << "begin" << begin <<"\n";
	//cout << "end" << end <<"\n";
	const int nzz = end -begin;
	//cout << "the slice of sparse as nzz " <<  nzz << " the old nzz was "<< _nzz << " start col "<< startCol <<"\n";
	return * new SparseNVMatrix(_devData,_sparseInd , _sparsePtr+ startCol,  getNumRows(), (endCol-startCol), nzz, SparseMatrix::CSC);
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
	//cout << "resize data of sparse matrix\n";
	bool reallocated = false;
	if (like.get_non_zeros() != _nzz) {
		assert(_ownsData);
		assert(_ownsDataInd);
		assert(_ownsDataPtr);

		_numRows = like.getNumRows();
		_numCols = like.getNumCols();

		if (_nzz > 0) { // free old memory
			//cout << "clearing memory from old psarse matrix during resize\n";
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
			//printf("allocating new memory\n");
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
			const int s =   (like.get_sparse_type() == SparseMatrix::CSC ? like.getNumCols() : like.getNumRows()) + 1;
			//cout << "allocating for matrix in device " << s << "cols\n";
			status = cublasAlloc(s , sizeof(int),
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
	//cout << "get transpose of sparse " << get_sparse_type() << "\n";
	return * new SparseNVMatrix(_devData,_sparseInd, _sparsePtr,   getNumCols(), getNumRows(), _nzz, (get_sparse_type() == SparseMatrix::CSR) ? SparseMatrix::CSC : SparseMatrix::CSR );
}

/*
 * Does HARD transpose and puts result in target
 */
void SparseNVMatrix::transpose(NVMatrix& target){
	//cout << "sparse not implemented transpose with target args\n";
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
		//cout << "doing transpose for sparse with bool argument\n";
		transpose();
	}
	return oldTrans;
}
void SparseNVMatrix::rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const{
	//cout << "right multiplication for sparse " << get_sparse_type() << "\n";
	addProductChanged(b, 0, scaleAB, target);
}


void SparseNVMatrix::addProductChanged( const NVMatrix &b, float scaleTarget, float scaleAB, NVMatrix &target)const{
	//cout << "addproduct changes Sparse\n";
	assert(_numCols == b.getNumRows());
	if(scaleTarget == 0.0) {
		//cout << "target is zero\n";
		target.resize(_numRows, b.getNumCols());
		target.setTrans(true);
	}else{
		assert(target.isTrans());
	}
	assert(target.getNumRows() == _numRows);
	assert(target.getNumCols() == b.getNumCols());
	assert(_numCols == b.getNumRows());

	//cout << "this:"<< getNumRows()<< ", "<< getNumCols()<< " b: "<<  b.getNumRows()<< ", "<< b.getNumCols() << " target:"<< target.getNumRows()<< ", "<< target.getNumCols()<< " nzz: "<< _nzz<<"\n";
	//cout << "b leading " <<  b.getLeadingDim() <<"\n";

	if (_sparse_type == SparseMatrix::CSC){

		//copyToHost(*(new Matrix()),true);
		//target.copyToHost(*(new Matrix()),true);
		//b.copyToHost(*(new Matrix()),true);
		//cusparseSetPointerMode(cudaSetup::_cusparseHandle,CUSPARSE_POINTER_MODE_HOST);
		//juto to check
		/*
		cout << "csc, rows " << getNumRows()<< ", cols: "<< getNumCols() << ", nzz: "<< _nzz<< " btrans: "<< b.isTrans() <<"\n";

		check_matrix<<<1,1>>>(getDevData(), _sparseInd, _sparsePtr, _nzz, getNumCols(), getNumRows());

		check_matrix_dense<<<1,1>>>(b.getDevData(), b.getNumRows(), b.getNumCols());
		check_matrix_dense<<<1,1>>>(target.getDevData(), target.getNumRows(), target.getNumCols());
        */
		if (b.isTrans()){

			//cout << "new implementation col by col";
			/*
			sparseMult(cudaSetup::_cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
							getNumCols(), b.getNumCols(), getNumRows(),_nzz,
							&scaleAB, cudaSetup::_sparseDescr,
							getDevData(), _sparsePtr, _sparseInd,
							b.getDevData(),  b.getLeadingDim() ,
							&scaleTarget,
							target.getDevData(), getNumRows());
			*/


			cusparseStatus_t cusparseStatus = cusparseScsrmm(cudaSetup::_cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE ,
							getNumCols(), b.getNumCols(), getNumRows(),_nzz,
							&scaleAB, cudaSetup::_sparseDescr,
							getDevData(), _sparsePtr, _sparseInd,
							b.getDevData(),  b.getLeadingDim() ,
							&scaleTarget,
							target.getDevData(), getNumRows());
			checkCudaErrors(cusparseStatus);


		}else{
		cusparseStatus_t cusparseStatus = cusparseScsrmm2(cudaSetup::_cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE ,b.isTrans()?CUSPARSE_OPERATION_NON_TRANSPOSE: CUSPARSE_OPERATION_TRANSPOSE,
				getNumCols(), b.getNumCols(), getNumRows(),_nzz,
				&scaleAB, cudaSetup::_sparseDescr,
				getDevData(), _sparsePtr, _sparseInd,
				b.getDevData(),  b.getLeadingDim() ,
				&scaleTarget,
				target.getDevData(), getNumRows());
		checkCudaErrors(cusparseStatus);
		}

	}else{
		//cout << "csr, rows " << getNumRows()<< ", cols: "<< getNumCols() << ", nzz: "<< _nzz<< " btrans: "<< b.isTrans() <<"\n";
		//check_matrix<<<1,1>>>(getDevData(), _sparseInd, _sparsePtr, _nzz,  getNumRows(),getNumCols());
		cusparseStatus_t cusparseStatus = cusparseScsrmm2(cudaSetup::_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE ,b.isTrans()?CUSPARSE_OPERATION_NON_TRANSPOSE: CUSPARSE_OPERATION_TRANSPOSE,
				getNumRows(), b.getNumCols(), getNumCols(),_nzz,
				&scaleAB, cudaSetup::_sparseDescr,
				getDevData(), _sparsePtr, _sparseInd,
				b.getDevData(),  b.getLeadingDim() ,
				&scaleTarget,
				target.getDevData(), getNumRows());
		checkCudaErrors(cusparseStatus);
	}

	//cout << "done addproduct changes Sparse\n";
}

void SparseNVMatrix::copyToHost(Matrix& hostMatrix, bool resizeTarget) const {
    if (resizeTarget) {
        hostMatrix.resize(_numRows, _numCols);
    }
    copyToHost(hostMatrix);
}


void SparseNVMatrix::copyToHost(Matrix& hostMatrix) const{
	//hostMatrix.apply(NVMatrixOps::Zero());
	NVMatrix tmpStorage = new NVMatrix();
	int r;
	int c;
	if (get_sparse_type() == SparseMatrix::CSC){
		r = getNumCols();
		c = getNumRows();
	}else{
		r = getNumRows();
		c = getNumCols();
	}
	tmpStorage.resize(r, c);
	tmpStorage.setTrans(true);

	cout<< "sparse to dense\n";
	cusparseStatus_t cusparseStatus = cusparseScsr2dense( cudaSetup::_cusparseHandle,
			r, c,
			cudaSetup::_sparseDescr,
			getDevData(),
			_sparsePtr, _sparseInd,
			tmpStorage.getDevData(), r);
	checkCudaErrors(cusparseStatus);
	if (get_sparse_type() == SparseMatrix::CSC){
		tmpStorage.transpose();
	}
	tmpStorage.copyToHost(hostMatrix);
}

void SparseNVMatrix::rightMult(const NVMatrix &b, float scaleAB){
	throw string("rightMult Not implemented for sparse!");
}


