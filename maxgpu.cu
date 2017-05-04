#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void getmaxcu(unsigned int* numbers_d, unsigned int* max_d, int n) {

	extern __shared__ unsigned int shared[];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	shared[tid] = 0;

	if (gid < n) {
		shared[tid] = numbers_d[gid];
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s = s / 2) 
	{
		if (tid < s && gid < n) {
			shared[tid] = max(shared[tid], shared[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0)
		max_d[blockIdx.x] = shared[tid];;
}

int main(int argc, char *argv[]) {
	unsigned int size = 0;  // The size of the array
	unsigned int i;  // loop index
	unsigned int * numbers; //pointer to the array
	unsigned int max;
	
	if(argc !=2) {
	   printf("usage: maxseq num\n");
	   printf("num = size of the array\n");
	   exit(1);
	}
   
	size = atol(argv[1]);

	numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
	if( !numbers ) {
	   printf("Unable to allocate mem for an array of size %u\n", size);
	   exit(1);
	}    

	srand(time(NULL)); // setting a seed for the random number generator
	// Fill-up the array with random numbers from 0 to size-1 
	for( i = 0; i < size; i++) {
	   numbers[i] = rand()  % size;    
	}

	// Memory allocation in the device
	unsigned int* numbers_d;
	unsigned int* max_d;
	cudaMalloc((void**)&numbers_d, size * sizeof(unsigned int));
	cudaMemcpy(numbers_d, numbers, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&max_d, size * sizeof(unsigned int));


	// Call kernel
	//getmaxcu<<<ceil(size/512),512>>>(numbers_d, max_d, size);

	// Free memory
	cudaFree(numbers_d);
	cudaFree(max_d);
	free(numbers);
	exit(0);
}