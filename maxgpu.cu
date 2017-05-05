#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

const int TPB = 128;

// __device__ unsigned int myMax(unsigned int* address, unsigned int val)
// {
// 	unsigned int old = *address;
// 	unsigned int assumed;
// 	while (val > old) {
// 		assumed = old;
// 		old = atomicCAS(address, assumed, val);
// 	}
// 	return old;
// }

__global__ void getmaxcu(unsigned int* numbers_d, unsigned int* max_d, int n) {

	extern __shared__ unsigned int shared[];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	shared[tid] = 0;

	// if (gid < n)
	// 	shared[tid] = numbers_d[gid];
	// __syncthreads();

	// for (unsigned int s = blockDim.x / 2; s > 0; s = s / 2) {
	// 	if (blockDim.x - tid > s && gid < n)
	// 		shared[tid] = max(shared[tid], shared[tid + s]);
	// 	__syncthreads();
	// }

	//if (tid == 0) {
		max_d[blockIdx.x] = 199;
		//shared[tid];
	//}
}

unsigned int getmax(unsigned int num[], unsigned int size) {
	unsigned int i;
	unsigned int max = num[0];

	for(i = 1; i < size; i++)
		if(num[i] > max)
			max = num[i];
	return( max );
}

void printArr(unsigned int num[], unsigned int size) {
	unsigned int i;

	for(i = 0; i < size; i++)
		printf("%u\n", num[i]);
}

int main(int argc, char *argv[]) {
	unsigned int size = 0;  // The size of the array
	unsigned int i;  // loop index
	unsigned int * numbers; //pointer to the array
	unsigned int * max;
	
	if(argc !=2) {
	   printf("usage: maxseq num\n");
	   printf("num = size of the array\n");
	   exit(1);
	}
   
	size = atol(argv[1]);

	numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
	max = (unsigned int *)malloc((size/TPB + 1) * sizeof(unsigned int));
	if( !numbers ) {
	   printf("Unable to allocate mem for an array of size %u\n", size);
	   exit(1);
	}    

	srand(time(NULL)); // setting a seed for the random number generator
	// Fill-up the array with random numbers from 0 to size-1 
	for( i = 0; i < size; i++) {
	   numbers[i] = rand()  % size;    
	}

	printf(" The maximum number in the array is: %u\n", 
           getmax(numbers, size));

	// Memory allocation in the device
	unsigned int* numbers_d;
	unsigned int* max_d;
	cudaMalloc((void**)&numbers_d, size * sizeof(unsigned int));
	cudaMalloc((void**)&max_d, (size/TPB + 1) * sizeof(unsigned int));

	// Call kernel
	int done = 0;
	for( i = size; i > 0 && done == 0;) {
		printf("Iteration: %u\n", i);
		cudaMemcpy(numbers_d, numbers, i * sizeof(unsigned int), cudaMemcpyHostToDevice);
		getmaxcu<<<(int)ceil((float)i / TPB),TPB>>>(numbers_d, max_d, i);
		i = (int)ceil((float)i / TPB);
		printArr(max, size/TPB + 1);
		cudaMemcpy(max, max_d, i * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if(i == 1) {
			done = 1;
		}
		printArr(max, size/TPB + 1);

	}

	// Copy memory from device to host
	//cudaMemcpy(numbers, max_d, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//max = numbers[0];

	// Print info
	printf(" The maximum number in the array is: %u\n", max[0]);

	// Free memory
	cudaFree(numbers_d);
	cudaFree(max_d);
	free(numbers);
	exit(0);
}