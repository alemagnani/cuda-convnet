#include "cuda_setup.cuh"

#include <iostream>
#include <cusparse_v2.h>


namespace cudaSetup {


bool            _cudaInitialized = false;
cusparseHandle_t  _cusparseHandle;
cusparseMatDescr_t  _sparseDescr;


void CudaStart()
{
	if (!_cudaInitialized) {
		std::cout << "initizlizin cusparse\n\n";
		cusparseStatus_t cusparseStatus = cusparseCreate(&_cusparseHandle);
		if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
			std::cout << "CUSPARSE initialization failed" << std::endl;
			exit(1);
		}

		_sparseDescr = 0;
		cusparseStatus = cusparseCreateMatDescr(&_sparseDescr);
		if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
			std::cout << "CUSPARSE initialization failed" << std::endl;
			exit(1);
		}
		cusparseSetMatType(_sparseDescr,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(_sparseDescr,CUSPARSE_INDEX_BASE_ZERO);
		_cudaInitialized = true;
	}
}


void CudaStop()
{
	if (_cudaInitialized)
		std::cout << "descroying the cusparse handle\n";
		cusparseDestroy(_cusparseHandle);
	_cudaInitialized = false;
}


}


