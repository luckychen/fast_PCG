#ifndef KERNEL_H
#define KERNEL_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
//#include <mpi.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "solver.h"
#define warpSize  32

const int memPerThread = 48;
const int threadELL = 1024;
const int warpPerBlock = threadELL/warpSize;
const int sharedPerBlock = memPerThread*threadELL;//1024 is the maximum threads per block
const int elementSize = 4; //if single precision, 4, if float precision, 8 
const int loopInKernel =  memPerThread/elementSize;
const int vectorCacheSize = sharedPerBlock/elementSize;
const int blockPerPart = sharedPerBlock/(warpSize*elementSize); 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


extern "C"
void initialize_all(const int dimension, float *pk_d, float *bp_d, float *x, float *zk, const float *vector_in_d);
void initialize_bp(int num, float *x);
void initialize_r(int num, float *rk, float *vector_in);
void myxpy(const int dimension, float gamak, const float *x, float *y);

void matrixVectorEHYB(matrixEHYB* inputMatrix, float* vector_in_d,
		float* vector_out_d, const int testPoint);

#endif
