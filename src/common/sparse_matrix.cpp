#include <matrix.h>
#include <iostream>
#include "sparse_matrix.h"

using namespace std;

SparseMatrix::SparseMatrix() {
	_updateDims(0, 0);
	_sparseInd = NULL;
	_sparsePtr = NULL;
	_nzz = 0;

	_ownsDataInd = true;
	_ownsDataPtr = true;

	_sparse_type = SparseMatrix::CSC;

}
SparseMatrix::SparseMatrix(long int numRows, long int numCols) {
	_updateDims(numRows, numCols);
	_sparseInd = NULL;
	_sparsePtr = NULL;
	_nzz = 0;
	_ownsDataInd = true;
	_ownsDataPtr = true;

	_sparse_type = SparseMatrix::CSC;

}

void readPythonArray(const PyArrayObject *src, int** data, long int& entries,
		bool& ownsData) {
	int num_dims = PyArray_NDIM(src);
	int item_size = PyArray_ITEMSIZE(src);

	entries = PyArray_DIM(src, 0);

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


SparseMatrix::SparseMatrix(const PyArrayObject *data, const PyArrayObject * sparseInd, const PyArrayObject * sparsePtr, long int numRows, long int numCols , SparseMatrix::SPARSE_TYPE sparse_type) : Matrix(data) {
	_updateDims(numRows, numCols);

	readPythonArray(sparseInd, & _sparseInd, _nzz, _ownsDataInd);
	long int tmp = 0;
	readPythonArray(sparsePtr, & _sparsePtr, tmp, _ownsDataPtr);

	_sparse_type = sparse_type;
}

SparseMatrix::~SparseMatrix() {
	if (this->_sparseInd != NULL && this->_ownsDataInd) {
		delete[] this->_sparseInd;
	}
	if (this->_sparsePtr != NULL && this->_ownsDataPtr) {
		delete[] this->_sparsePtr;
	}
}
