#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

unsigned int getmax(unsigned int *, unsigned int);

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
   
	// printf(" The maximum number in the array is: %u\n", 
	//        getmax(numbers, size));

	// Memory allocation in the device
	unsigned int* numbers_d;
	unsigned int* max_d;
	cudaMalloc((void**)&numbers_d, size * sizeof(unsigned int));
	cudaMemcpy(numbers_d, numbers, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&max_d, size * sizeof(unsigned int));


	// Call kernel
	//getmaxcu<<<ceil(size/256),256>>>(numbers_d, max_d, size);

	// Free memory
	cudaFree(numbers_d);
	cudaFree(max_d);
	free(numbers);
	exit(0);
}

__global__ void getmaxcu(unsigned int* numbers_d, unsigned int* max_d, int n) {

	for(int i = 1;;;) {

		__syncthreads();
	}
}

/*
   input: pointer to an array of long int
		  number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmax(unsigned int num[], unsigned int size) {
  unsigned int i;
  unsigned int max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}
