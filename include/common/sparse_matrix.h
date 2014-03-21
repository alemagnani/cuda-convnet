/*
 * sparse_matrix.h
 *
 *  Created on: Mar 16, 2014
 *      Author: alessandro
 */

#ifndef SPARSE_MATRIX_H_
#define SPARSE_MATRIX_H_

#include <matrix_funcs.h>
#include <Python.h>
#include <arrayobject.h>
#include <limits>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <matrix.h>

#include <iostream>

using namespace std;

extern "C" {
#include <cblas.h>
}

class SparseMatrix : public Matrix  {

public:

    enum SPARSE_TYPE {
   	        CSR, CSC
   	    };

private:
    int* _sparseInd;
    int* _sparsePtr;

    long int _nzz;

    bool _ownsDataInd;
    bool _ownsDataPtr;

    SPARSE_TYPE _sparse_type;


public:


    SparseMatrix();
    SparseMatrix(long int numRows, long int numCols);
    SparseMatrix(const PyArrayObject *data, const PyArrayObject *sparseInd, const PyArrayObject *sparsePtr, long int numRows, long int numCols, SPARSE_TYPE sparse_type );
    ~SparseMatrix();

    inline long int get_non_zeros() const{
    	return _nzz;
    }

    inline int * getSparseInd() const{
    	return _sparseInd;
    }

    inline int* getSparsePtr() const{
    	return _sparsePtr;
    }

    inline MATRIX_TYPE get_type() const {
        	return SPARSE;
        }

    inline SPARSE_TYPE get_sparse_type() const {
    	return _sparse_type;
    }

    inline long int getNumDataBytes() const {
    	  cout << "\ngetting right size\n";
          return _nzz * (sizeof(MTYPE) + sizeof(int)) + sizeof(int) *( getNumCols()+1);
      }

};





#endif /* SPARSE_MATRIX_H_ */
