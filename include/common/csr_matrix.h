/*
 * csr_matrix.h
 *
 *  Created on: Feb 14, 2014
 *      Author: alessandro
 */

#ifndef CSR_MATRIX_H_
#define CSR_MATRIX_H_

#include <matrix_funcs.h>
#ifdef NUMPY_INTERFACE
#include <Python.h>
#include <arrayobject.h>
#endif
#include <limits>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <matrix.h>

extern "C" {
#include <cblas.h>
}

class CsrMatrix : public Matrix  {
private:
    int* _csrColInd;
    int* _csrRowPtr;

    long int _nzz;

    bool _ownsDataColInd;
    bool _ownsDataRowInd;


public:
    CsrMatrix();
    CsrMatrix(long int numRows, long int numCols);
#ifdef NUMPY_INTERFACE
    CsrMatrix(const PyArrayObject *data, const PyArrayObject *csrColIndA, const PyArrayObject *csrRowPtrA, long int numRows, long int numCols );
#endif
    ~CsrMatrix();

    inline long int get_non_zeros() const{
    	return _nzz;
    }

    inline int * getColInd() const{
    	return _csrColInd;
    }

    inline int* getRowPtr() const{
    	return _csrRowPtr;
    }

    inline MATRIX_TYPE get_type() const {
        	return CSR;
        }
};

void readPythonArray(const PyArrayObject *src, int** data, long int& entries,
		bool& ownsData);

#endif /* CSR_MATRIX_H_ */
