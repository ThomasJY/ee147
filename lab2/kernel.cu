/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
	__shared__ float ds_A[TILE_SIZE][TILE_SIZE];
	__shared__ float ds_B[TILE_SIZE][TILE_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by*blockDim.y + ty;
	int col = bx*blockDim.x + tx;
	
	float result = 0;

	for(int i = 0; i < n/TILE_SIZE; i++){
		ds_A[ty][tx] = A[row*k + i*TILE_SIZE + tx];
		ds_B[ty][tx] = B[(i*TILE_SIZE + ty)*n + col];
		__syncthreads();
		
		for(int j = 0; j < TILE_SIZE; j++){
			result += ds_A[ty][j] * ds_B[j][tx];
			__syncthreads();
		}
		C[row*n + col] = result;
	}





}



void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
	dim3 dim_grid, dim_block;
	dim_block = BLOCK_SIZE;
	dim_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
	mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);


}


