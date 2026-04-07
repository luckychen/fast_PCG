#include "kernel.h"
#include "solver.h"
#include "convert.h"
#include "Partition.h"

#define warpSize 32

static void sortRordrListFull(unsigned int dimension,
		int* reorderList, 
		int* numInRow)
{
	//This function generated the reorderList according to the
	//numInRow, since it is a wrapper of qsort function, and 
	//the logic is a bit different from similar function at
	//reordering.c, we make a seperate function in this file
	//instead of reuse the function in reordering.c
	rowS* rodrSVec = (rowS*) malloc(dimension*sizeof(rowS));
	
	for(unsigned int i = 0; i < dimension; ++i){
		//the location of elements at SVec will be changed 
		//record the original idx since it will be used for
		//new rodr_list
		rodrSVec[i].idx = i; 
		rodrSVec[i].nonzeros = numInRow[i];	
	}	
	qsort(rodrSVec, dimension, sizeof(rowS), rowSCompare);	
	for(unsigned int i = 0; i < dimension; ++i){
		reorderList[rodrSVec[i].idx] = i;
	}
	free(rodrSVec);
}

static void ELL_block_cols_vec_gen_rodr(unsigned int* ELL_block_cols_vec, 
		unsigned int* size_COO, 
		const uint32_t* col,
		unsigned int* block_boundary, 
		const unsigned int* part_boundary, 
		const int dimension, 
		const int max_col, 
		const int part_num,
	   	const unsigned int* row_idx,
		const bool CACHED)
{
	/*Reoder functions did an extra iteration 
	 * which provides numInRowER without ER reordering
	 * */
	// int block_nonz = 0;
	int blockIdx = 0;
	int numCols = 0;
	int maxCol = inputMatrix->maxCol;
	int nonZeros;
	//int colTh = 0, colAccum = 0; 
	//int *colHist = ( int*)calloc(maxCol, sizeof( int));
	int *numInRow = inputMatrix->numInRow;
	int *numInRow2 = inputMatrix->numInRow2; 

	*size_COO = 0;
	block_boundary[0] = 0;
	uint32_t ELL_val = 0;
	unsigned int extraRows = 0;
			
	for(unsigned int part_idx = 0; part_idx < part_num; ++part_idx){
		unsigned int block_end = part_boundary[part_idx] + ELL_threadSize*block_per_part;
		unsigned int part_end = part_boundary[part_idx + 1];
		unsigned int part_start = part_boundary[part_idx];

	for(int partIdx = 0; partIdx < inputMatrix->nParts; ++partIdx){
		int partStart = inputMatrix->partBoundary[partIdx];
		int partEnd = inputMatrix->partBoundary[partIdx + 1];
		if(partEnd > partStart + blockPerPart*warpSize)
			extraRows = partEnd - partStart - blockPerPart*warpSize;
		else
			extraRows = 0;

		for(int iter = 0; iter < blockPerPart; ++iter){
			blockIdx = iter + blockPerPart*partIdx;
			int blockStart = partStart + iter*warpSize;
			numCols = 0;
			for(int row = blockStart; (row < blockStart + warpSize && row < partEnd); ++row){
				if(numInRow2[row] > numCols)
					numCols = numInRow2[row];
				if(numInRow2[row] != numInRow[row]){
					numOfRowER += 1;
					numInRowER[row] = numInRow[row] - numInRow2[row];
					toER += numInRowER[row];
				}
			}
			if(numCols == 0 && blockStart < partEnd){
				printf("error at ELL_block convert rodr\n");
				exit(0);
			}		
			for(int row_val = local_start; row_val < local_end; ++row_val){
				if(CACHED){
					ELL_val = 0;
					for(uint32_t j = row_idx[row_val]; j < row_idx[row_val + 1]; ++j){
						if(col[j] < part_start || col[j] >= block_end)
							*size_COO += 1;
						else 
							ELL_val += 1; 
					}
					if(ELL_val > num_cols)
						*size_COO += ELL_val - num_cols;

				} else {

					unsigned int nonzeros = row_idx[row_val + 1] - row_idx[row_val];
					if(nonzeros > num_cols){
						*size_COO += nonzeros - num_cols;	
					} 
				}
			}
			outputMatrix->widthVecBlockELL[blockIdx] = numCols;
		}
		if(extraRows > 0){
			for(int row = partStart + blockPerPart*warpSize; row < partEnd; ++row){
				numOfRowER += 1;
				numInRowER[row] += numInRow[row];
				toER += numInRowER[row];
			}	
		}
	}
	if(numOfRowER == 0){
		printf("perfect matrix, not in the scope of this study\n");		
		exit(0);
	}
	printf("toER is %d\n", toER);
	outputMatrix->numOfRowER = numOfRowER;
	outputMatrix->rowVecER = (int*)malloc(numOfRowER*sizeof(int));
	outputMatrix->biasVecER = (int*)malloc(ceil(((float) numOfRowER)/warpSize)*sizeof(int));
}

static void COO2ELL_block_core(unsigned int* colELL, double* matrixELL,
		unsigned int* I_COO, unsigned int* J_COO, double* V_COO, const unsigned int size_COO,
		const unsigned int* row_idx, const unsigned int* numInRow, 
		const unsigned int loc_num_of_row, 
        const unsigned int* ELL_block_bias_vec, const unsigned int* ELL_block_cols_vec, 
		const unsigned int* row_local, const unsigned int* col_local, 
		const double* matrix_local,
		unsigned int block_num, 
		const unsigned int* part_boundary, 
		volatile bool RODR, bool CACHED)
{ 
	unsigned int block_rowSize;
	unsigned int block_end = 0, part_start = 0, part_end = 0, extraRows = 0;
	unsigned int irregular=0;
	block_rowSize = ELL_threadSize;
	unsigned pointCOO = 0;
	uint32_t ELL_idx = 0;
	unsigned int boundary_start, boundary_end;
	unsigned int part_idx;
	uint32_t padding = 0;
	for (unsigned int block_idx = 0; block_idx < block_num; ++block_idx){
		unsigned int num_cols = ELL_block_cols_vec[block_idx];
		unsigned int block_bias = ELL_block_bias_vec[block_idx];
		extraRows = 0;
		if(RODR){
			part_idx = block_idx/block_per_part;
			part_start = part_boundary[part_idx];
			part_end = part_boundary[part_idx + 1];
			boundary_start = part_boundary[part_idx] + (block_idx%block_per_part)*ELL_threadSize;
			block_end = part_boundary[part_idx] + block_per_part*ELL_threadSize;	

			if((block_idx + 1)%block_per_part == 0){
				if(part_end > block_end){
				   	extraRows = part_end - block_end;
					boundary_end = block_end;
				}
				else boundary_end = part_end;
			} else {
				boundary_end = boundary_start + ELL_threadSize;
			}
		}
			
		block_rowSize = boundary_end - boundary_start;
		for(unsigned int i = 0; i < block_rowSize; ++i){
			unsigned int row_val = i + boundary_start;
			if(row_val >= loc_num_of_row) 
				break;
			ELL_idx = 0;
			for(uint32_t j = 0; j < numInRow[row_val]; ++j ){

				uint32_t col_val = col_local[row_idx[row_val] + j] ;

				if(ELL_idx < num_cols ){
					if(CACHED == true && (col_val < part_start || col_val >= block_end)){
						I_COO[pointCOO]=row_local[row_idx[row_val]+j];
						J_COO[pointCOO]=col_local[row_idx[row_val]+j];
						V_COO[pointCOO]=matrix_local[row_idx[row_val]+j];
						pointCOO++;
						if(pointCOO > size_COO){
							printf("error at pointCOO %d\n", pointCOO);
							exit(0);
						}
					} else {
					//goto ELL format
					colELL[block_bias + i + ELL_idx*block_rowSize] = col_val;
					matrixELL[block_bias + i + ELL_idx*block_rowSize]
						= matrix_local[row_idx[row_val] + j];
					ELL_idx += 1;
					}
				} else {
					//assign COO value
					I_COO[pointCOO]=row_local[row_idx[row_val]+j];
					J_COO[pointCOO]=col_local[row_idx[row_val]+j];
					V_COO[pointCOO]=matrix_local[row_idx[row_val]+j];
					pointCOO++;
					if(pointCOO > size_COO){
						printf("error at pointCOO %d at point 1\n", pointCOO);
						exit(0);
					}

				}

			}
			if(ELL_idx < num_cols){
				for(uint32_t j = ELL_idx; j < num_cols; ++j){
					colELL[block_bias + i + j*block_rowSize] = 0;
					matrixELL[block_bias + i + j*block_rowSize] = 0;
					if(part_start > 0 && (ELL_idx == 0 || colELL[block_bias] == 0))
						printf("debug poit at %d\n", row_val);
					padding += 1;
				}

			}
			

		}
		if(extraRows > 0){
			for(int rowVal = (partStart + warpSize*blockPerPart); rowVal < partEnd; ++rowVal){
				if(rowVecER[reorderListER[rowVal]] != rowVal){
					printf("error at rowVecER\n");
					exit(0);
				}
				rowLocER = reorderListER[rowVal];
				blockIdxER = rowLocER/warpSize;
				rowLocInBlockER = rowLocER - blockIdxER*warpSize;
				biasER = biasVecER[blockIdxER];
				widthER = widthVecER[blockIdxER];
				for(int j = 0; j < numInRow[rowVal]; ++j){
					colER[biasER+rowLocInBlockER+j*warpSize]=J[rowIdx[rowVal]+j];
					valER[biasER+rowLocInBlockER+j*warpSize]=V[rowIdx[rowVal]+j];
					irregular+=1;
					if(pointCOO > size_COO){
						printf("error at pointCOO  %d at point 2\n", pointCOO);

						exit(0);
					}
				}
			}
		}
		extraRows = 0;
	}

}

/*parameters, first line: output of HYB related parameters, 2nd line: output of HYB matrices, 3rd line:input of local COO matrix 
  4th line: ER index (didn't include values), 5th line: input variables*/

void COO2ELL_block(unsigned int *size_COO, 
		unsigned int* ELL_block_cols_vec, unsigned int* ELL_block_bias_vec,
		unsigned int **colELL, double **matrixELL, unsigned int **I_COO, 
		unsigned int **J_COO, double **V_COO,
		const unsigned int *row_local, const unsigned int *col_local, const double* matrix_local, 
		const unsigned int *row_idx, const unsigned int *numInRow, const unsigned int max_col,
		const unsigned int localMatrixSize, const unsigned int loc_num_of_row, 
		const unsigned int part_num, const unsigned int block_num, 
		const unsigned int* part_boundary, bool RODR, bool CACHED)
{ 

	int* reorderListER = (int*)calloc(inputMatrix->dimension, sizeof(int));
	int* numInRowER = (int*)calloc(inputMatrix->dimension, sizeof(int));
	outputMatrix->dimension = inputMatrix->dimension;
	outputMatrix->nParts = inputMatrix->nParts;
	outputMatrix->partBoundary = inputMatrix->partBoundary;
	outputMatrix->widthVecBlockELL = (int*)calloc(outputMatrix->nParts*blockPerPart, sizeof(int));
	outputMatrix->biasVecBlockELL = (int*)calloc(outputMatrix->nParts*blockPerPart, sizeof(int));
	/*block_boundary will not be used outside this function*/
	unsigned int* block_boundary = (unsigned int*)malloc((block_num + 1) * sizeof(unsigned int));
	*size_COO=0;	
	if(RODR){
		ELL_block_cols_vec_gen_rodr(ELL_block_cols_vec, 
				size_COO, 
				col_local,
				block_boundary,
				part_boundary, 
				loc_num_of_row, 
				max_col, 
				part_num,
				row_idx,
				CACHED);
	} else {
		ELL_block_cols_vec_gen(ELL_block_cols_vec, size_COO,
			loc_num_of_row, max_col, row_idx); 
	}

	if(*size_COO > 0){	
		*I_COO=(unsigned int *)malloc((*size_COO)*sizeof(unsigned int));
		*J_COO=(unsigned int *)malloc((*size_COO)*sizeof(unsigned int));
		*V_COO=(double *)malloc((*size_COO)*sizeof(double));
	}
	unsigned int ELL_matrixSize = 0;
	for(unsigned int i = 0; i < block_num; ++i){
		ELL_block_bias_vec[i] = ELL_matrixSize;
		if(RODR){
			ELL_matrixSize += (block_boundary[i+1] - block_boundary[i])*ELL_block_cols_vec[i];
		} else {
			ELL_matrixSize += ELL_threadSize*ELL_block_cols_vec[i];
		}
	}
	ELL_block_bias_vec[block_num] = ELL_matrixSize;
	*colELL=(unsigned int *)malloc(ELL_matrixSize*sizeof(unsigned int));
	*matrixELL=(double *)malloc(ELL_matrixSize*sizeof(double));

	COO2ELL_block_core(*colELL, *matrixELL,
			*I_COO, *J_COO, *V_COO, *size_COO,
			row_idx, numInRow, loc_num_of_row, 
			ELL_block_bias_vec, ELL_block_cols_vec, 
			row_local, col_local, matrix_local, block_num, 
			part_boundary, RODR, CACHED);
	
	//checkBlockELL(outputMatrix);
	outputMatrix->reorderListER = reorderListER;
	free(numInRowER);
}

