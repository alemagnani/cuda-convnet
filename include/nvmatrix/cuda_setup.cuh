/*
 * cuda_setup.cuh
 *
 *  Created on: Mar 14, 2014
 *      Author: alessandro
 */

#ifndef CUDA_SETUP_CUH_
#define CUDA_SETUP_CUH_

#include <cusparse_v2.h>
#include <helper_cuda.h>

class CudaState {
public:
	static CudaState* instance();
	CudaState() {
		cusparseStatus_t cusparseStatus = cusparseCreate(&_cusparseHandle);
		if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
			std::cerr << "CUSPARSE initialization failed" << std::endl;
			exit(1);
		}

		_sparseDescr = 0;
		cusparseStatus = cusparseCreateMatDescr(&_sparseDescr);
		checkCudaErrors(cusparseStatus);
		cusparseSetMatType(_sparseDescr,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(_sparseDescr,CUSPARSE_INDEX_BASE_ZERO);
	}

	inline cusparseMatDescr_t& cusparseDescr(){
		return _sparseDescr;
	}

	inline cusparseHandle_t& cusparseHandle(){
		return _cusparseHandle;
	}

private :
	static CudaState* instance_;
	cusparseHandle_t _cusparseHandle;
	cusparseMatDescr_t _sparseDescr;


	CudaState& operator=(const CudaState&);
	CudaState(const CudaState&);
};


CudaState* CudaState::instance() {
	if( instance_==0 ) {
		instance_ = new CudaState;
	}
	return instance_;
}


#endif /* CUDA_SETUP_CUH_ */
