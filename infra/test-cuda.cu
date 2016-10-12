// since compilation is slow, do NOT depend on library wide types...
#include <infra/config.h>
#include <infra/infra-api.h>
#include <iostream>
#include <sstream>
#include "test-cuda.h"

 
const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
 
namespace nll
{
namespace cuda
{
	std::string testCuda()
	{
		char a[N] = "Hello \0\0\0\0\0\0";
		int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
		char *ad;
		int *bd;
		const int csize = N*sizeof(char);
		const int isize = N*sizeof(int);

      std::string start = a;
 
		cudaMalloc( (void**)&ad, csize ); 
		cudaMalloc( (void**)&bd, isize ); 
		cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
		cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
	
		dim3 dimBlock( blocksize, 1 );
		dim3 dimGrid( 1, 1 );
		hello<<<dimGrid, dimBlock>>>(ad, bd);
		cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
		cudaFree( ad );
		cudaFree( bd );
	
      std::stringstream ss;
      ss << start;
      ss << a;
		return ss.str();
	}
}
}