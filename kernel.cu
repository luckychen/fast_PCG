#include "kernel.h"

#define FULL_MASK 0xffffffff
#define BASE 262144 //1024*1024

#define block_size 512	
#define thread_size 512
#define block_size2 16
#define thread_size2 512
#define warpSize 32

/*kernel function for initialize*/
__global__ void kernelInitialize(const int num, float *x)
{
	int idx=blockDim.x * blockIdx.x+ threadIdx.x;
	
	for (int n=idx;n<num;n+=BASE) x[n]=0;
}

__global__ void kernelInitializeAll(const int num, float *pk, float *bp, float *x, float *zk, const float *vector_in)
{
	int idx=blockDim.x * blockIdx.x+ threadIdx.x;
	float temp;
	for (int n=idx;n<num;n+=BASE) 
	{
		temp=zk[n];
		pk[n]=temp;
		bp[n]=0;
		x[n]=0;
	}
}

__global__ void kernelInitializeR(const int num,float *rk, const float *vector_in)
{
	int idx=blockDim.x * blockIdx.x+ threadIdx.x;
	float temp;
	for (int n=idx;n<num;n+=BASE) 
	{
		temp=vector_in[n];
		rk[n]=temp;
	}
}

//first version implementation,
//concern about performance loss from inbalance between blocks 
__global__ void kernelER(const int numOfRowER,
			const int* rowVecER,
			const int* biasVecER,  
			const int* widthVecER, 
			const int* colER, 
			const float *valER, const float * x, float * y)
{
	int width;  
	int bias;
	int dataIdx;
	int row, col;	
	float val;
	uint32_t idx = blockDim.x*blockIdx.x+threadIdx.x;
	uint32_t warpIdx = idx>>5;
	uint32_t warpLane = threadIdx.x- ((threadIdx.x>>5)<<5);
	if(idx < numOfRowER){
		row = rowVecER[idx];
		width = widthVecER[warpIdx];//cache will work when every threads read same global address
		bias = biasVecER[warpIdx];
		float dot = 0;
		for(int n=0; n < width; ++n){
			dataIdx = bias + warpLane + warpSize*n ;
			col=colER[dataIdx];
			val=valER[dataIdx];
			dot += val* x[col];
		}
		y[row]+=dot;
	}
}

__global__ void kernelCachedBlockedELL_test(const int* widthVecBlockELL,
		const int* biasVecBlockELL,  
		const int *colBlockELL, 
		const float *valBlockELL, 
		const float * x,
		float * y,
		const int* partBoundary,
		const int testPoint)
{
	int partIdx = blockIdx.x; 
	int xIdx = threadIdx.x;
	__shared__ volatile float cachedVec[vectorCacheSize];  
	int vecStart = partBoundary[blockIdx.x];
	int vecEnd = partBoundary[blockIdx.x + 1];
	int warpLane = xIdx - ((xIdx>>5)<<5); //xIdx%32 = xIdx - (xIdx/32)*32)
	int row = 0;
	int blockStartIdx = blockPerPart*partIdx;	
	for (int i = xIdx; i < vectorCacheSize; i += threadELL){
		cachedVec[i] = x[i + vecStart];
	}
	
	__syncthreads();
	float val, dot;
	int dataIdx; 
	int col;
	int biasIdx, bias, width;

	#pragma unroll
	for(int i = 0; i < loopInKernel; ++i){//the thread is step with stride threadELL
		dot = 0;
		//each iteration go through (1024/warpSize)=32 blocks in blockELL format, which is i >> 5
		//the warpIdx is xIdx>>5
		row = i*threadELL + vecStart + xIdx;
		if(row < vecEnd){
			biasIdx = i*warpPerBlock + (xIdx>>5) + blockStartIdx;
			bias = biasVecBlockELL[biasIdx]; 
			width = widthVecBlockELL[biasIdx];
			for(int n=0; n< width; ++n){
				dataIdx = bias + warpSize*n + warpLane;//however the data storage is stride with block_rowSize
				val= valBlockELL[dataIdx];
				col = colBlockELL[dataIdx]- vecStart;
				if(row == testPoint)
					dot += val*cachedVec[col] - 1 + 0.999;
				else
					dot += val*cachedVec[col];
			}
			//if(row == testPoint)
			//	y[row] = dot+0.01;
			//else 
			y[row] = dot;
		}
	}		
}

__global__ void kernelCachedBlockedELL(const int* widthVecBlockELL,
		const int* biasVecBlockELL,  
		const int *colBlockELL, 
		const float *valBlockELL, 
		const float * x,
		float * y,
		const int* partBoundary)
{
	int partIdx = blockIdx.x; 
	int xIdx = threadIdx.x;
	__shared__ volatile float cachedVec[vectorCacheSize];  
	//__shared__ volatile int sharedBias[blockPerPart];  
	//__shared__ volatile int sharedWidth[blockPerPart];  
	int vecStart = partBoundary[blockIdx.x];
	int vecEnd = partBoundary[blockIdx.x + 1];
	int warpLane = xIdx - ((xIdx>>5)<<5); //xIdx%32 = xIdx - (xIdx/32)*32)
	int row = 0;
	int blockStartIdx = blockPerPart*partIdx;	
	for (int i = xIdx; i < vectorCacheSize; i += threadELL){
		cachedVec[i] = x[i + vecStart];
	}
	//if(xIdx < blockPerPart){
	//	sharedBias[xIdx] = biasVecBlockELL[blockStartIdx + xIdx];	
	//	sharedWidth[xIdx] = widthVecBlockELL[blockStartIdx+ xIdx];	
	//}
	__syncthreads();
	float val, dot;
	int dataIdx; 
	int col;
	int biasIdx, bias, width;

	#pragma unroll
	for(int i = 0; i < loopInKernel; ++i){//the thread is step with stride threadELL
		dot = 0;
		row = i*threadELL + vecStart + xIdx;
		if(row < vecEnd){
			biasIdx = i*warpPerBlock + (xIdx>>5) + blockStartIdx;
			bias = biasVecBlockELL[biasIdx]; 
			width = widthVecBlockELL[biasIdx];
			for(int n=0; n< width; ++n){
				dataIdx = bias + warpSize*n + warpLane;//however the data storage is stride with block_rowSize
				val= valBlockELL[dataIdx];
				col = colBlockELL[dataIdx]- vecStart;
				dot += val*cachedVec[col];
			}
			//if(row == testPoint)
			//	y[row] = dot+0.01;
			//else 
			y[row] = dot;
		}
	}
}


//y=x+gamak*y
__global__ void kernelMyxpy(const int dimension, float gamak, const float *x, float *y)
{
	int idx=blockDim.x*blockIdx.x+threadIdx.x;
	int n=idx;
	while(n<dimension){
		y[n]=x[n]+gamak*y[n];
		n=n+BASE;
	}
}

extern "C"
void initialize_all(const int dimension, float *pk_d, float *bp_d, float *x, float *zk, const float *vector_in_d)
{
	kernelInitializeAll<<<block_size,thread_size>>>(dimension, pk_d, bp_d, x, zk, vector_in_d);
}

void initialize_bp(int num, float *x)
{
	kernelInitialize<<<block_size,thread_size>>>(num,x);
}

void initialize_r(int num, float *rk, float *vector_in)
{
	kernelInitializeR<<<block_size,thread_size>>>(num,rk,vector_in);
}
void myxpy(const int dimension, float gamak, const float *x, float *y)
{
	kernelMyxpy<<<block_size,thread_size>>>(dimension,gamak,x,y);
}

void initialDeviceArray(int num, float *x)
{
	kernelInitialize<<<512,512>>>(num,x);
}



void matrixVectorBlockELL(const int nParts, const int testPoint, 
		const int* widthVecBlockELL_d, 
		const int* biasVecBlockELL_d,    
		const int* colBlockELL_d,
		const float* valBlockELL_d, 
		const int* partBoundary_d,
		const float *x_d, float *y_d)
{

		if(testPoint >= 0){
			kernelCachedBlockedELL_test<<<nParts, threadELL>>>(widthVecBlockELL_d,
					biasVecBlockELL_d,  
					colBlockELL_d, valBlockELL_d, 
					x_d,
					y_d,
					partBoundary_d,
					testPoint);
		} else {
			kernelCachedBlockedELL<<<nParts, threadELL>>>(widthVecBlockELL_d,
					biasVecBlockELL_d,  
					colBlockELL_d, valBlockELL_d, 
					x_d,
					y_d,
					partBoundary_d);
		}

}

void matrixVectorER(const int numOfRowER, 
		const int* rowVecER_d, const int* biasVecER_d, 
		const int* widthVecER_d, 
		const int* colER_d, const float* valER_d,
		const float* vectorIn_d, float* vectorOut_d)
{

	int blockSizeLocal;
	blockSizeLocal=ceil(((float) numOfRowER)/threadELL);//for data with 2 million elements, we have interval size 200
	kernelER<<<blockSizeLocal, threadELL>>>(numOfRowER, 
			rowVecER_d, 
			biasVecER_d, 
			widthVecER_d,
			colER_d, 
			valER_d, 
			vectorIn_d, 
			vectorOut_d);

}

void matrixVectorEHYB(matrixEHYB* inputMatrix_d, float* vectorIn_d,
		float* vectorOut_d, const int testPoint)
{

	matrixVectorBlockELL(inputMatrix_d->nParts, 
			testPoint,
			inputMatrix_d->widthVecBlockELL,
			inputMatrix_d->biasVecBlockELL,  
			inputMatrix_d->colBlockELL, 
			inputMatrix_d->valBlockELL, 
			inputMatrix_d->partBoundary,
			vectorIn_d,
			vectorOut_d);
	
	matrixVectorER(inputMatrix_d->numOfRowER, inputMatrix_d->rowVecER, 
			inputMatrix_d->biasVecER,
			inputMatrix_d->widthVecER,
			inputMatrix_d->colER, 
			inputMatrix_d->valER,
			vectorIn_d, vectorOut_d);

}
