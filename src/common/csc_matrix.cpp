#include <csc_matrix.h>
#include <matrix.h>
#include <iostream>
#include "csr_matrix.h"

using namespace std;

CscMatrix::CscMatrix() {
	_updateDims(0, 0);
	_cscRowInd = NULL;
	_cscColPtr = NULL;
	_nzz = 0;

	_ownsDataRowInd = true;
	_ownsDataColPtr = true;

}
CscMatrix::CscMatrix(long int numRows, long int numCols) {
	_updateDims(numRows, numCols);
	_cscRowInd = NULL;
	_cscColPtr = NULL;
	_nzz = 0;
	_ownsDataRowInd = true;
	_ownsDataColPtr = true;

}



#ifdef NUMPY_INTERFACE
CscMatrix::CscMatrix(const PyArrayObject *data, const PyArrayObject * cscRowInd, const PyArrayObject * cscColPtr, long int numRows, long int numCols ) : Matrix(data) {
	_updateDims(numRows, numCols);

	cout << "\ninitializing c matrix. Is trans:" << isTrans() << "numRows: " << numRows << " numCOls: " << numCols <<  "\n";

	readPythonArray(cscRowInd, & _cscRowInd, _nzz, _ownsDataRowInd);
	long int tmp = 0;
	readPythonArray(cscColPtr, & _cscColPtr, tmp, _ownsDataColPtr);
}
#endif

CscMatrix::~CscMatrix() {
	if (this->_cscRowInd != NULL && this->_ownsDataRowInd) {
		delete[] this->_cscRowInd;
	}
	if (this->_cscColPtr != NULL && this->_ownsDataColPtr) {
		delete[] this->_cscColPtr;
	}
}
