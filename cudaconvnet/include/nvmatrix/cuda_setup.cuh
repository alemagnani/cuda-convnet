/*
 * cuda_setup.cuh
 *
 *  Created on: Mar 14, 2014
 *      Author: alessandro
 */

#ifndef CUDA_SETUP_CUH_
#define CUDA_SETUP_CUH_

#include <iostream>
#include <cusparse_v2.h>



namespace cudaSetup {


extern bool            _cudaInitialized;
extern cusparseHandle_t  _cusparseHandle;
extern cusparseMatDescr_t  _sparseDescr;


void CudaStart();
void CudaStop();

}

#endif /* CUDA_SETUP_CUH_ */
