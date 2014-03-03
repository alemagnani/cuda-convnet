/*
 * csc_matrix.h
 *
 *  Created on: Mar 3, 2014
 *      Author: alessandro
 */

#ifndef CSC_MATRIX_H_
#define CSC_MATRIX_H_

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

class CscMatrix : public Matrix  {
private:
    int* _cscRowInd;
    int* _cscColPtr;

    long int _nzz;

    bool _ownsDataRowInd;
    bool _ownsDataColPtr;


public:
    CscMatrix();
    CscMatrix(long int numRows, long int numCols);
#ifdef NUMPY_INTERFACE
    CscMatrix(const PyArrayObject *data, const PyArrayObject *cscRowIndA, const PyArrayObject *cscColPtrA, long int numRows, long int numCols );
#endif
    ~CscMatrix();

    inline long int get_non_zeros() const{
    	return _nzz;
    }

    inline int * getRowInd() const{
    	return _cscRowInd;
    }

    inline int* getColPtr() const{
    	return _cscColPtr;
    }

    inline MATRIX_TYPE get_type() const {
        	return CSC;
        }
};




#endif /* CSC_MATRIX_H_ */
