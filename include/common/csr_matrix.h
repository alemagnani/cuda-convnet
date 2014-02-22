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
    long int* _csrColIndA;
    long int* _csrRowPtrA;
    long int _nzz;

public:
    CsrMatrix();
    CsrMatrix(long int numRows, long int numCols);
#ifdef NUMPY_INTERFACE
    CsrMatrix(const PyArrayObject *data, long int* csrColIndA, long int* csrRowPtrA, long int numRows, long int numCols );
#endif
    ~CsrMatrix();

    inline MATRIX_TYPE get_type() const {
        	return CSR;
        }


};


#endif /* CSR_MATRIX_H_ */
