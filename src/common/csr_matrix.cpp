#include <csr_matrix.h>
#include <matrix.h>

using namespace std;



    CsrMatrix::CsrMatrix(){
    	_updateDims(0, 0);
    	_csrColIndA = NULL;
    	_csrRowPtrA = NULL;
    	_nzz = 0;

    }
    CsrMatrix::CsrMatrix(long int numRows, long int numCols){
    	_updateDims(numRows, numCols);
    	_csrColIndA = NULL;
    	    	_csrRowPtrA = NULL;
    	    	_nzz = 0;
    }
#ifdef NUMPY_INTERFACE
    CsrMatrix::CsrMatrix(const PyArrayObject *data, long int* csrColIndA, long int* csrRowPtrA, long int numRows, long int numCols ) : Matrix(data){
    	_updateDims(numRows, numCols);
    	_csrColIndA = csrColIndA;
    	 _csrRowPtrA = csrRowPtrA;
    	    	_nzz = 0;
    }
#endif

    CsrMatrix::~CsrMatrix(){
    	if(this->_csrColIndA != NULL && this->_ownsData) {
    	        delete[] this->_csrColIndA;
    	    }
    	if(this->_csrRowPtrA != NULL && this->_ownsData) {
    	    	        delete[] this->_csrRowPtrA;
    	    	    }
    }
