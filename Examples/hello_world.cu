#include <stdio.h>

__device__ const char *STR = "HELLO WORLD!";
const char STR_LENGTH = 12;

__global__ void hello()
{
	printf("%c from thread %d\n", STR[threadIdx.x % STR_LENGTH], threadIdx.x);
}

int main(void)
{
	int num_threads = STR_LENGTH;
	int num_blocks = 1;
	hello<<<num_blocks,num_threads>>>();
	cudaDeviceSynchronize();

	return 0;
}
