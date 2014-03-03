#include <csr_matrix.h>
#include <matrix.h>
#include <iostream>

using namespace std;

CsrMatrix::CsrMatrix() {
	_updateDims(0, 0);
	_csrColInd = NULL;
	_csrRowPtr = NULL;
	_nzz = 0;

	_ownsDataColInd = true;
	_ownsDataRowInd = true;

}
CsrMatrix::CsrMatrix(long int numRows, long int numCols) {
	_updateDims(numRows, numCols);
	_csrColInd = NULL;
	_csrRowPtr = NULL;
	_nzz = 0;
	_ownsDataColInd = true;
	_ownsDataRowInd = true;

}

void readPythonArray(const PyArrayObject *src, int** data, long int& entries,
		bool& ownsData) {
	int num_dims = PyArray_NDIM(src);
	int item_size = PyArray_ITEMSIZE(src);
	cout << "\nnumber of dimensions: " << num_dims << " item size: " << item_size <<"\n";

	entries = PyArray_DIM(src, 0);

	cout << "\nrows: " << entries  << "\n\n";

	if (src->flags & NPY_CONTIGUOUS || src->flags & NPY_FORTRAN) {
		*data = (int*) src->data;
		ownsData = false;
	} else {
		*data = new  int[entries];
		for (long int i = 0; i < entries; i++) {
				(*data)[i] = *reinterpret_cast<int*>(PyArray_GETPTR1(
						src, i));
		}
		ownsData = true;
	}
}

#ifdef NUMPY_INTERFACE
CsrMatrix::CsrMatrix(const PyArrayObject *data, const PyArrayObject * csrColInd, const PyArrayObject * csrRowPtr, long int numRows, long int numCols ) : Matrix(data) {
	_updateDims(numRows, numCols);

	cout << "\ninitializing csr matrix. Is trans:" << isTrans() << "numRows: " << numRows << " numCOls: " << numCols << "\n";

	readPythonArray(csrColInd, & _csrColInd, _nzz, _ownsDataColInd);
	long int tmp = 0;
	readPythonArray(csrRowPtr, & _csrRowPtr, tmp, _ownsDataRowInd);
}
#endif

CsrMatrix::~CsrMatrix() {
	if (this->_csrColInd != NULL && this->_ownsDataColInd) {
		delete[] this->_csrColInd;
	}
	if (this->_csrRowPtr != NULL && this->_ownsDataRowInd) {
		delete[] this->_csrRowPtr;
	}
}
