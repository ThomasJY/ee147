/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE


__global__ void histo_kernel(unsigned int* input, unsigned int *bins, unsigned int num_elements,
        unsigned int num_bins)
{
	__shared__ unsigned int histo_private[4096];
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < 4096)
		histo_private[i] = 0;
	__syncthreads();
	
	int stride = blockDim.x*gridDim.x;
        while(i < num_elements){
                atomicAdd(&histo_private[input[i]], 1);
		i += stride;
	}
	__syncthreads();
	
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if(j < 4096)
		atomicAdd(&bins[j], histo_private[j]);
        
}









/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
        
        // Initialize thread block and kernel grid dimensions ---------------------
        const unsigned int BLOCK_SIZE = 128;

	dim3 dim_grid = 1024;
	dim3 dim_block = BLOCK_SIZE;
	
        // Invoke CUDA kernel -----------------------------------------------------
	histo_kernel<<<dim_grid, dim_block>>>(input, bins, num_elements, num_bins);


}


