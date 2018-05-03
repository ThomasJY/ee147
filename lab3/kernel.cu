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
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        
        if (i < num_elements)
	{
		unsigned int a = input[i];
                atomicAdd(&bins[a], 1);
	}
        
        
}









/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
        
        // Initialize thread block and kernel grid dimensions ---------------------
        const unsigned int BLOCK_SIZE = 128;

	dim3 dim_grid = (num_elements + BLOCK_SIZE - 1)/BLOCK_SIZE;
	dim3 dim_block = BLOCK_SIZE;
	
	unsigned int* histo;
	histo = (unsigned int*)bins;

        // Invoke CUDA kernel -----------------------------------------------------
	histo_kernel<<<dim_grid, dim_block>>>(input, histo, num_elements, num_bins);


}


