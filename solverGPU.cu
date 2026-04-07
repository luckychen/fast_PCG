#include "kernel.h"
#include "solver.h"
#include "convert.h"
#include "reordering.h"
#include "test.h"

static void cudaMallocTransDataEHYB(matrixEHYB* localMatrix, matrixEHYB* localMatrix_d, 
		const int sizeBlockELL, const int sizeER){


	localMatrix_d->dimension = localMatrix->dimension;
	localMatrix_d->numOfRowER = localMatrix->numOfRowER;
	localMatrix_d->nParts = localMatrix->nParts;
	int blockNumER = ceil(((float) localMatrix->numOfRowER)/warpSize);

    cudaMalloc((void **) &(localMatrix_d->biasVecBlockELL), localMatrix->nParts*blockPerPart*sizeof(int));
    cudaMalloc((void **) &(localMatrix_d->widthVecBlockELL), localMatrix->nParts*blockPerPart*sizeof(int));
    cudaMalloc((void **) &(localMatrix_d->partBoundary), (localMatrix->nParts+1)*sizeof(int));
    cudaMalloc((void **) &(localMatrix_d->valBlockELL), sizeBlockELL*sizeof(float));
    cudaMalloc((void **) &(localMatrix_d->colBlockELL), sizeBlockELL*sizeof(int));

    cudaMalloc((void **) &(localMatrix_d->rowVecER), localMatrix_d->numOfRowER*sizeof(int));
    cudaMalloc((void **) &(localMatrix_d->biasVecER), blockNumER*sizeof(int));
    cudaMalloc((void **) &(localMatrix_d->widthVecER), blockNumER*sizeof(int));
    cudaMalloc((void **) &(localMatrix_d->colER), sizeER*sizeof(int));
    cudaMalloc((void **) &(localMatrix_d->valER), sizeER*sizeof(float));

    cudaMemcpy(localMatrix_d->biasVecBlockELL, localMatrix->biasVecBlockELL, localMatrix->nParts*blockPerPart*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->widthVecBlockELL, localMatrix->widthVecBlockELL, localMatrix->nParts*blockPerPart*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->partBoundary, localMatrix->partBoundary, (localMatrix->nParts+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->valBlockELL, localMatrix->valBlockELL, sizeBlockELL*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->colBlockELL, localMatrix->colBlockELL, sizeBlockELL*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(localMatrix_d->rowVecER, localMatrix->rowVecER, localMatrix_d->numOfRowER*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->biasVecER, localMatrix->biasVecER, blockNumER*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->widthVecER, localMatrix->widthVecER, blockNumER*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->colER, localMatrix->colER, sizeER*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(localMatrix_d->valER, localMatrix->valER, sizeER*sizeof(float), cudaMemcpyHostToDevice);
}
extern "C"
void solverGPuUnprecondEHYB(matrixCOO* localMatrix, 
		const float *vectorIn, float *vectorOut,  
		const int MAXIter, int *realIter)
{
	//This function treat y as input and x as output, (solve the equation Ax=y) y is the vector we already known, x is the vector we are looking for
	float dotp0,dotr0,dotr1,doth;

	int sizeBlockELL, sizeER;
	int dimension = localMatrix->dimension;
	int totalNum = localMatrix->totalNum;
	matrixEHYB localMatrixEHYB, localMatrixEHYB_d;

        ELL_block_cols_vec = (uint32_t*)malloc(ELL_blocks*sizeof(uint32_t));  
        ELL_block_bias_vec = (uint32_t*)malloc((ELL_blocks +1)*sizeof(uint32_t));  
		COO2ELL_block(&totalNumCOO, 
				ELL_block_cols_vec, 
				ELL_block_bias_vec,
				&colELL, 
				&matrixELL, 
				&I_COO, 
				&J_COO, 
				&V_COO,
				I, 
				J, 
				V, 
				row_idx, 
				numInRow, 
				maxRowNum, 
				totalNum, 
				dimension,
				part_size, 
				ELL_blocks,
				part_boundary, 
				RODR,
				CACHE);
		ELL_cuda_malloc_trans_data_block(&col_d, 
				&V_d, 
				&ELL_block_cols_vec_d, 
				&ELL_block_bias_vec_d,
				colELL, 
				matrixELL,
				ELL_block_cols_vec, 
				ELL_block_bias_vec,
				dimension, 
				ELL_blocks);

	cudaMallocTransDataEHYB(&localMatrixEHYB,
			&localMatrixEHYB_d, 
			sizeBlockELL,
			sizeER);
	printf("sizeER is %d\n", sizeER);
	cublasHandle_t handle;
	cublasCreate(&handle);

	float *bp_d, *pk_d, *rk_d, *vectorOut_d;
	size_t size1 = dimension*sizeof(float);
	cudaMalloc((void **) &bp_d,size1);
	cudaMalloc((void **) &pk_d,size1);
	cudaMalloc((void **) &rk_d,size1);
	cudaMalloc((void **) &vectorOut_d,size1);
	//float *x=(float *) malloc(size1);
	float threshold=0.0000001;
	int iter=0;
	float const1 = 1.0;
	float error, alphak, _alphak, gamak;
	error=1000;
	//initialize
	doth=0;
    cudaMemcpy(pk_d, vectorIn, dimension*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rk_d, vectorIn, dimension*sizeof(float), cudaMemcpyHostToDevice);
	for (int i=0;i<dimension;i++) {
		doth=doth+vectorIn[i]*vectorIn[i];
	}
	struct timeval start1, end1;
	float *bp=(float *) malloc(size1);
	float *bp_g =(float *) malloc(size1);
	float *pk=(float *) malloc(size1);
	float *rk=(float *) malloc(size1);

	//float *x=(float *) malloc(size1);
	error=1000;
	//initialize
	for (int i=0;i<dimension;i++)
	{
		pk[i]=vectorIn[i];
		rk[i]=vectorIn[i];
		vectorOut[i]=0;
		bp[i]=0;
		bp_g[i]=0;
	}
	gettimeofday(&start1, NULL);
	while (error>threshold&&iter<MAXIter){
		dotp0=0;
		dotr0=0;
		dotr1=0;
		int errorIdx = 0;
		float compareError;
		cudaMemset(bp_d, 0, size1);

		//matrix_vectorTest(localMatrix, vector_in, bp, 0);
		
		matrix_vectorHYB(&localMatrixHYB_d, pk_d, bp_d, cb, 0,
			   part_size, part_boundary_d, true);
		//cudaMemcpy(bp_g, bp_d, dimension*sizeof(double), cudaMemcpyDeviceToHost);
		//for(int i = 0; i < dimension; ++i){
		//	compareError = 	(bp_g[i] - bp[i])/bp[i];
		//	if(errorIdx == 0  && (compareError > 0.000001 || compareError < -0.00001)){ 	
		//		printf("bp[%d] of GPU result is %f, test value is %f\n", i, bp_g[i], bp[i]);
		//		if(errorIdx == 0) errorIdx = i;	
		//	}
		//}
		
		//matrix_vectorTest(localMatrix, vector_in, bp, errorIdx);
		//cudaMemset(bp_d, 0, size1);
		//matrix_vectorHYB(&localMatrixHYB_d, pk_d, bp_d, cb, errorIdx,
		//	   part_size, part_boundary_d, errorIdx);
		
		//for (uint32_t k=0;k<5;k++) printf("pk %d at iter %d is %f\n",k, iter, pk[k]);
		//printf("CPU start\n");
		
		cublasDdot(handle,dimension,bp_d,1,pk_d,1,&dotp0);
		cublasDdot(handle,dimension,rk_d,1,rk_d,1,&dotr0);
			
		alphak=dotr0/dotp0;
		_alphak = -alphak;
		
		cublasSaxpy(handle,dimension,&alphak,pk_d,1,vectorOut_d,1);
		cublasSaxpy(handle,dimension,&_alphak,bp_d,1,rk_d,1);
		cublasSdot(handle,dimension,rk_d,1,rk_d,1,&dotr1);
		
		gamak=dotr1/dotr0;

		cublasSscal(handle,dimension,&gamak, pk_d,1);
		cublasSaxpy(handle,dimension,&const1, rk_d, 1, pk_d, 1);
		
		//printf("at iter %d, alphak is %f, gamak is %f\n",iter, alphak,gamak);
		error=sqrt(dotr1)/sqrt(doth);
		//error_track[iter]=error;
		//printf("error at %d is %f\n",iter, error);
		iter++;
	}
	cudaMemcpy(vectorOut, vectorOut_d, dimension*sizeof(float), cudaMemcpyDeviceToHost);
	gettimeofday(&end1, NULL);	
	float timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;
	printf("iter is %d, time is %f ms, GPU Gflops is %f, under estimate flops is %f\n ",iter, timeByMs, 
			(1e-9*(totalNum*2+13*dimension)*1000*iter)/timeByMs, (1e-9*(totalNum*2)*1000*iter)/timeByMs);
	cudaFree(localMatrixEHYB_d.valER);
	cudaFree(localMatrixEHYB_d.colER);
	cudaFree(localMatrixEHYB_d.biasVecER);
	cudaFree(localMatrixEHYB_d.widthVecER);
	cudaFree(localMatrixEHYB_d.rowVecER);
	cudaFree(localMatrixEHYB_d.biasVecBlockELL);
	cudaFree(localMatrixEHYB_d.widthVecBlockELL);
	cudaFree(localMatrixEHYB_d.colBlockELL);
	cudaFree(localMatrixEHYB_d.valBlockELL);
	cudaFree(localMatrixEHYB_d.partBoundary);
}

void solverGPuUnprecondCUSPARSE(matrixCOO* localMatrix, 
		const float *vector_in, float *vector_out,  
		const int MAXIter)
{
	//exampine the performance using cusparse library functions with
	//CSR format
	//float dotp0,dotr0,dotr1,doth;
	float dotp0,dotr0,dotr1,doth;
	int dimension, totalNum; 
    int *rowIdx, *J; 
    float* V;
    dimension = localMatrix->dimension; 
    totalNum = localMatrix->totalNum; 

	rowIdx = localMatrix->rowIdx; 
    J = localMatrix->J;
    V = localMatrix->V;
	
	int* col_d;
	int* rowIdx_d;
	float *V_d;
	cublasHandle_t handleBlas;
	cublasCreate(&handleBlas);
	cusparseHandle_t handleSparse;
	cusparseCreate(&handleSparse);

	float *bp_d, *pk_d, *rk_d, *vector_out_d;
	size_t size1=dimension*sizeof(float);
	cudaMalloc((void **) &bp_d,size1);
	cudaMalloc((void **) &pk_d,size1);
	cudaMalloc((void **) &rk_d,size1);
	cudaMalloc((void **) &rowIdx_d,size1);
	cudaMalloc((void **) &vector_out_d,size1);
	cudaMalloc((void **) &col_d,totalNum*sizeof(int));
	cudaMalloc((void **) &V_d,totalNum*sizeof(float));
	//float *x=(float *) malloc(size1);
	float threshold=0.0000001;
	int iter=0;
	float const1 = 1.0;
	float error, alphak, _alphak, gamak;
	error=1000;
	//initialize
	doth=0;
    cudaMemcpy(pk_d, vector_in, dimension*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rk_d, vector_in, dimension*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rowIdx_d, rowIdx, (dimension+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_d, J, totalNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, totalNum*sizeof(float), cudaMemcpyHostToDevice);
	for (int i=0;i<dimension;i++) {
		doth=doth+vector_in[i]*vector_in[i];
	}


	uint32_t *colELL, *I_COO, *J_COO, ELL_width, ELL_widthL, ELL_widthLP;
	double* matrixELL, *V_COO;
	uint32_t *col_d;
	double *V_d;
	volatile bool RODR, BLOCK, CACHE;
	uint32_t *I_COO_d, *J_COO_d;
	double *V_COO_d;
	BLOCK = cb.BLOCK;
	CACHE = cb.CACHE;
	RODR = cb.RODR;
    uint32_t ELL_blocks;
    uint32_t *ELL_block_cols_vec, *ELL_block_cols_vec_d;    
    uint32_t *ELL_block_bias_vec, *ELL_block_bias_vec_d;	
	uint32_t *ELL_block_cols_vec_L, *ELL_block_cols_vec_L_d;    
    uint32_t *ELL_block_bias_vec_L, *ELL_block_bias_vec_L_d;
	uint32_t *ELL_block_cols_vec_LP, *ELL_block_cols_vec_LP_d;    
	uint32_t *ELL_block_bias_vec_LP, *ELL_block_bias_vec_LP_d;
	uint32_t totalNumCOO, totalNumCOO_L, totalNumCOO_LP;
    if(BLOCK){
        //RODR will change the number of blocks since it provides cache option
        if(RODR)
			ELL_blocks = part_size*block_per_part; 
		else
			ELL_blocks = ceil((double) dimension/ELL_threadSize);

        ELL_block_cols_vec = (uint32_t*)malloc(ELL_blocks*sizeof(uint32_t));  
        ELL_block_bias_vec = (uint32_t*)malloc((ELL_blocks +1)*sizeof(uint32_t));  
		
		COO2ELL_block(&totalNumCOO, 
				ELL_block_cols_vec, 
				ELL_block_bias_vec,
				&colELL, 
				&matrixELL, 
				&I_COO, 
				&J_COO, 
				&V_COO,
				I, 
				J, 
				V, 
				row_idx, 
				numInRow, 
				maxRowNum, 
				totalNum, 
				dimension,
				part_size, 
				ELL_blocks,
				part_boundary, 
				RODR,
				CACHE);
		ELL_cuda_malloc_trans_data_block(&col_d, 
				&V_d, 
				&ELL_block_cols_vec_d, 
				&ELL_block_bias_vec_d,
				colELL, 
				matrixELL,
				ELL_block_cols_vec, 
				ELL_block_bias_vec,
				dimension, 
				ELL_blocks);
	} else {
		COO2ELL(I,J,V,&colELL,&matrixELL,&I_COO, &J_COO, &V_COO, 
			numInRow, row_idx, totalNum, dimension, &totalNumCOO, maxRowNum, &ELL_width);

        ELL_cuda_malloc_trans_data(&col_d, &V_d, colELL, matrixELL, dimension, ELL_width);
	}
	
	if (totalNumCOO>0){
        COO_cuda_malloc_trans_data(&I_COO_d, &J_COO_d, &V_COO_d, 
				I_COO, J_COO, V_COO, 
				dimension, totalNumCOO);
	}	

	/*matrix L'*/
	uint32_t *colELL_precond, *I_COO_L, *J_COO_L;
	double* matrixELL_precond, *V_COO_L;
	uint32_t *col_precond_d;
	double *V_precond_d;
	uint32_t *I_COO_L_d, *J_COO_L_d;
	double *V_COO_L_d;

    if(BLOCK){
		ELL_block_cols_vec_L = (uint32_t*)malloc(ELL_blocks*sizeof(uint32_t));  
        ELL_block_bias_vec_L = (uint32_t*)malloc((ELL_blocks + 1)*sizeof(uint32_t));
		COO2ELL_block(&totalNumCOO_L, 
				ELL_block_cols_vec_L, 
				ELL_block_bias_vec_L,
				&colELL_precond, 
				&matrixELL_precond,
			   	&I_COO_L, &J_COO_L, &V_COO_L,
				I_precond, J_precond, V_precond, 
				row_idxL, numInRowL, maxRowNumPrecond,
				totalNumPrecond, dimension, part_size,
				ELL_blocks, part_boundary, RODR, CACHE);
		ELL_cuda_malloc_trans_data_block(&col_precond_d, &V_precond_d, 
				&ELL_block_cols_vec_L_d, &ELL_block_bias_vec_L_d,
				colELL_precond, matrixELL_precond,
				ELL_block_cols_vec_L, ELL_block_bias_vec_L,
				dimension, ELL_blocks);
	} else {
		COO2ELL(I_precond,J_precond,V_precond,&colELL_precond, &matrixELL_precond, 
				&I_COO_L, &J_COO_L, &V_COO_L, numInRowL, row_idxL, totalNumPrecond, 
				dimension, &totalNumCOO_L, maxRowNumPrecond, &ELL_widthL);
        ELL_cuda_malloc_trans_data(&col_precond_d, &V_precond_d, colELL_precond, 
				matrixELL_precond, dimension, ELL_widthL);
	}
	//printf("ELL_widthL is %d, and totalNumCOO_L is %d\n", ELL_widthL, totalNumCOO_L);
	if (totalNumCOO_L>0){
		COO_cuda_malloc_trans_data(&I_COO_L_d, &J_COO_L_d, &V_COO_L_d, 
				I_COO_L, J_COO_L, V_COO_L, 
				dimension, totalNumCOO_L);
	}	

	/*matrix L*/
	uint32_t *colELL_precondP, *I_COO_LP, *J_COO_LP;
	double* matrixELL_precondP, *V_COO_LP;
	uint32_t *col_precondP_d;
	double *V_precondP_d;
	uint32_t *I_COO_LP_d, *J_COO_LP_d;
	double *V_COO_LP_d;

	if(FACT){
		if(BLOCK){
			ELL_block_cols_vec_LP = (uint32_t*)malloc(ELL_blocks*sizeof(uint32_t));  
			ELL_block_bias_vec_LP = (uint32_t*)malloc((ELL_blocks + 1)*sizeof(uint32_t));
			COO2ELL_block(&totalNumCOO_LP, ELL_block_cols_vec_LP, ELL_block_bias_vec_LP,
					&colELL_precondP, &matrixELL_precondP, &I_COO_LP, &J_COO_LP, &V_COO_LP,
					I_precondP, J_precondP, V_precondP, 
					row_idxLP, numInRowLP, maxRowNumPrecondP, 
					totalNumPrecondP, dimension, part_size,
					ELL_blocks, part_boundary, RODR, CACHE);
			printf("the last cols_vec_LP is %d\n", ELL_block_cols_vec_LP[ELL_blocks - 1]);
			ELL_cuda_malloc_trans_data_block(&col_precondP_d, &V_precondP_d, 
					&ELL_block_cols_vec_LP_d, &ELL_block_bias_vec_LP_d,
					colELL_precondP, matrixELL_precondP,
					ELL_block_cols_vec_LP, ELL_block_bias_vec_LP,
					dimension, ELL_blocks);
		} else {
			COO2ELL(I_precondP,J_precondP,V_precondP,&colELL_precondP, &matrixELL_precondP, 
					&I_COO_LP, &J_COO_LP, &V_COO_LP, numInRowLP, row_idxLP, totalNumPrecondP, 
					dimension, &totalNumCOO_LP, maxRowNumPrecondP, &ELL_widthLP);
			ELL_cuda_malloc_trans_data(&col_precondP_d, &V_precondP_d, colELL_precondP, 
					matrixELL_precondP, dimension, ELL_widthLP);
		}
		//printf("ELL_widthL is %d, and totalNumCOO_L is %d\n", ELL_widthL, totalNumCOO_L);
		if (totalNumCOO_LP>0){
			COO_cuda_malloc_trans_data(&I_COO_LP_d, &J_COO_LP_d, &V_COO_LP_d, 
					I_COO_LP, J_COO_LP, V_COO_LP, 
					dimension, totalNumCOO_LP);
		}
	}	
	printf("totalNumCOO is  %d totalNumCOO_L is %d, and totalNumCOO_LP is %d\n", totalNumCOO, totalNumCOO_L, totalNumCOO_LP);
	//printf("fill rate original is %f, fill rate L is %f, fill rate lp is %f\n", calFilledRate(totalNum, ELL_blocks,), calFilledRate, calFilledRate);
	
	
	uint32_t* part_boundary_d;
	
	double *rk=(double *)malloc(size0);
	double *zk=(double *)malloc(size0);
	double *zk_1=(double *)malloc(size0);
	
	double *vector_in_d;
	double alphak,betak,dotp0, dotrz0,dotrz1,doth,alphak_1,dotr0;
	
	//double dotz0,dotz0_compare;
	
	double error=10000;
	double threshold;
	
	if(BLOCK && RODR){
	   	cudaMalloc((void **) &part_boundary_d, (part_size +1)*sizeof(uint32_t));
		cudaMemcpy(part_boundary_d, part_boundary, 
				(part_size +1)*sizeof(uint32_t), cudaMemcpyHostToDevice);
	}

	cudaMalloc((void **) &vector_in_d,size0);
	//cudaMalloc((void **) &vector_out_d,size0);
	cudaMemcpy(vector_in_d, vector_in, size0, cudaMemcpyHostToDevice);	
	
	//
	initialize_bp(dimension,zk_d);
	initialize_bp(dimension,zk1_d);
	initialize_r(dimension, rk_d, vector_in_d);

	double* pkT = (double*) malloc(dimension*sizeof(double));	
	double* bpT = (double*) calloc(dimension, sizeof(double));	
	double* bp_g = (double*) calloc(dimension, sizeof(double));

	cublasHandle_t handle;
	cublasCreate(&handle);			
	double* zk1T = (double*) calloc(dimension, sizeof(double));	
	double* rkT = (double*) calloc(dimension, sizeof(double));	
	double* zk1_g = (double*) calloc(dimension, sizeof(double));
	if(!BLOCK){
		matrix_vectorELL(dimension, dimension, ELL_widthL, col_precond_d,V_precond_d,
				rk_d,zk1_d, false, 0, NULL);
					
	} else {
		if(RODR){
			matrix_vectorELL_block(dimension, 0, ELL_block_cols_vec_L_d, 
					ELL_block_bias_vec_L_d,
                    col_precond_d,V_precond_d,rk_d,zk1_d,
					CACHE, part_size, part_boundary_d, tex);
		}
		else{
			matrix_vectorELL_block(dimension, 0, ELL_block_cols_vec_L_d, 
					ELL_block_bias_vec_L_d,
					col_precond_d,V_precond_d,rk_d,zk1_d,
					false, 0, NULL, tex);
		}
	}
	if (totalNumCOO_L>0) matrix_vectorCOO(totalNumCOO_L, I_COO_L_d, J_COO_L_d, V_COO_L_d, 
									rk_d, zk1_d, 0, tex);
	cudaMemcpy(zk1_g, zk1_d, dimension*sizeof(double), cudaMemcpyDeviceToHost);
	uint32_t errorIdx = 0;
	float compareError;
	matrix_vectorTest(localMatrix_precond, vector_in, zk1T, 0);
	for(int i = 0; i < dimension; ++i){
		rkT[i] = vector_in[i];
		compareError = 	(zk1T[i] - zk1_g[i])/zk1_g[i];
		if(errorIdx == 0  && (compareError > 0.000001 || compareError < -0.00001)){ 	
			printf("zk1[%d] of GPU result is %f, test value is %f\n", i, zk1_g[i], zk1T[i]);
			if(errorIdx == 0) errorIdx = i;	
		}
	}
	if(errorIdx > 0){
		//matrix_vectorTest(localMatrix_precond, vector_in, zk1T, errorIdx);
		matrix_vectorELL_block(dimension, errorIdx, ELL_block_cols_vec_L_d, 
				ELL_block_bias_vec_L_d,
				col_precond_d,V_precond_d,rk_d,zk1_d,
				CACHE, part_size, part_boundary_d, tex);
	}
	if(FACT){	
		if (!BLOCK) {
			matrix_vectorELL(dimension, dimension, ELL_widthLP, col_precondP_d, 
					V_precondP_d, zk1_d, zk_d, false, 0, NULL);
		} else {
			if(RODR){
				matrix_vectorELL_block(dimension, 0, ELL_block_cols_vec_LP_d, ELL_block_bias_vec_LP_d,  
						col_precondP_d,V_precondP_d, 
						zk1_d, zk_d, CACHE, part_size, part_boundary_d, tex); 

			} else {
				matrix_vectorELL_block(dimension, 0, ELL_block_cols_vec_LP_d, ELL_block_bias_vec_LP_d,  
						col_precondP_d, V_precondP_d, 
						zk1_d, zk_d, false, 0, NULL, tex);

			}	
		}
		if (totalNumCOO_LP>0) matrix_vectorCOO(totalNumCOO_LP, I_COO_LP_d, J_COO_LP_d, V_COO_LP_d, zk1_d, zk_d, 0, tex);
	}	
	//double* zkT = (double*) calloc(dimension, sizeof(double));	
	//double* zk_g = (double*) calloc(dimension, sizeof(double));	
	//cudaMemcpy(zk_g, zk_d, dimension*sizeof(double), cudaMemcpyDeviceToHost);
	//matrix_vectorTest(localMatrix_precondP, zk1T, zkT);
	//for(int i = 0; i < dimension; ++i){
	//	pkT[i] = zkT[i];
	//	if(abs((zkT[i] - zk_g[i])/zk_g[i]) > 0.000001) 	
	//		printf("zk[%d] of GPU result is %f, test value is %f\n", i, zk_g[i], zkT[i]);
	//}
	//cublasDdot(handle,dimension,zk_d,1,zk_d,1,&dotz0);
	//for (uint32_t i=0;i<5;i++) printf("dotz0 is %f, dotz0_compare is %f\n",dotz0,dotz0_compare);
	//give an initial value as r0, p0, and set all 
	initialize_all(dimension,pk_d,bp_d,x_d,zk_d,vector_in_d);
	
	
	cublasDdot(handle, dimension,vector_in_d,1,vector_in_d,1,&doth);
	double errorNorm=sqrt(doth);
	uint32_t iter=0;
	struct timeval start1, end1;
	float *bp=(float *) malloc(size1);
	float *bp_g =(float *) malloc(size1);
	float *pk=(float *) malloc(size1);
	float *rk=(float *) malloc(size1);

	//float *bp_dt = (float *) malloc(size1);
	//float *pk_dt = (float *) malloc(size1);
	//float *rk_dt = (float *) malloc(size1);
	//float *x=(float *) malloc(size1);
	error=1000;
	//initialize
	for (int i=0;i<dimension;i++)
	{
		pk[i]=vector_in[i];
		rk[i]=vector_in[i];
		vector_out[i]=0;
		bp[i]=0;
	}
	//if BSR doing the format change 
	//cusparseStatus_tcusparseDcsr2gebsr_bufferSize(handle, dir, m, n, descrA, csrValA, csrRowPtrA, 
	//		csrColIndA, rowBlockDim, colBlockDim, pBufferSize);
	cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseMatDescr_t descr = 0;
	int status = cusparseCreateMatDescr(&descr);
	if (status != CUSPARSE_STATUS_SUCCESS ) {
		exit(0);	
	}
	cusparseSetMatType (descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase (descr, CUSPARSE_INDEX_BASE_ZERO);
	gettimeofday(&start1, NULL);
	float one = 1.0;
	float zero = 0.0;
	while (error>threshold&&iter<MAXIter){
		dotp0=0;
		dotr0=0;
		dotr1=0;
		//int errorIdx = 0;
		//float compareError;
		
		cudaMemset(bp_d, 0, size1);
		cusparseStatus_t smpvStatus = 
		cusparseScsrmv(handleSparse,
				transA,
				dimension,
				dimension,
				totalNum,
				&one,
				descr,
				V_d,
				rowIdx_d,
				col_d,
				pk_d,
				&zero,
				bp_d);

		cublasSdot(handleBlas,dimension,bp_d,1,pk_d,1,&dotp0);
		cublasSdot(handleBlas,dimension,rk_d,1,rk_d,1,&dotr0);
			
		alphak=dotr0/dotp0;
		_alphak = -alphak;
		
		cublasSaxpy(handleBlas,dimension,&alphak,pk_d,1,vector_out_d,1);
		cublasSaxpy(handleBlas,dimension,&_alphak,bp_d,1,rk_d,1);
		cublasSdot(handleBlas,dimension,rk_d,1,rk_d,1,&dotr1);
		
		gamak=dotr1/dotr0;

		cublasSscal(handleBlas,dimension,&gamak,pk_d,1);
		cublasSaxpy(handleBlas,dimension,&const1, rk_d, 1, pk_d, 1);
		
		//printf("at iter %d, alphak is %f, gamak is %f\n",iter, alphak,gamak);
		error=sqrt(dotr1)/sqrt(doth);
		//error_track[iter]=error;
		//printf("error at %d is %f\n",iter, error);
		iter++;
	}
	cudaMemcpy(vector_out, vector_out_d, dimension*sizeof(float), cudaMemcpyDeviceToHost);
	gettimeofday(&end1, NULL);	
	float timeByMs=((end1.tv_sec * 1000000 + end1.tv_usec)-(start1.tv_sec * 1000000 + start1.tv_usec))/1000;
	printf("iter is %d, time is %f ms, GPU csrmv Gflops is %f, under estimate flops is %f\n ",iter, timeByMs, 
			(1e-9*(totalNum*2+13*dimension)*1000*iter)/timeByMs, (1e-9*(totalNum*2)*1000*iter)/timeByMs);

}

