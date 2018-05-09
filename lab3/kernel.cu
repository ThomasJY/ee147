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
	extern __shared__ unsigned int histo_private[];
	
	int i = threadIdx.x;
	int stride = blockDim.x;
	while(i < num_bins){
		histo_private[i] = 0;
		i += stride;
	}
	__syncthreads();
	
	i = blockIdx.x*blockDim.x + threadIdx.x;
	stride = blockDim.x*gridDim.x;
        while(i < num_elements){
                atomicAdd(&histo_private[input[i]], 1);
		i += stride;
	}
	__syncthreads();
	
	i = threadIdx.x;
	stride = blockDim.x;
	while(i < num_bins){
		atomicAdd(&bins[i], histo_private[i]);
		i += stride;
	}
        
}









/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
        // Initialize thread block and kernel grid dimensions ---------------------
        const unsigned int BLOCK_SIZE = 512;

	dim3 dim_grid = 32;
	dim3 dim_block = BLOCK_SIZE;
	
        // Invoke CUDA kernel -----------------------------------------------------
	size_t SIZEOF_SHARED_MEMORY_IN_BYTES = num_bins*sizeof(unsigned int);
	histo_kernel<<<dim_grid, dim_block, SIZEOF_SHARED_MEMORY_IN_BYTES>>>(input, bins, num_elements, num_bins);

}


