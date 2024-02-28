#include <builtin_types.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include "../drvapi_error_string.h"

using namespace std;
	__global__
void saxpy(int n, float a, float *x, float *y)
{
	for (int i=0 ; i<10000000000 ; i++)
	{
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) y[i] = a*x[i] + y[i];
		 asm("exit;");
	}
}

void prepareSaxpy()
{

	int N = 1<<20;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float)); 
	cudaMalloc(&d_y, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

}

int main(int argc, char *argv[])
{
	struct timeval popContext_1, popContext_2, cudaCallStart_1, cudaCallEnd_1;
	struct timeval cudaCallStart_2, cudaCallEnd_2, cudaCallStart_3, cudaCallEnd_3;
	CUcontext cuCurrent = NULL;
	CUresult result;
	CUdevice cuDevice; 
	char name[100];
	
	result = cuInit(0);
	if (result != CUDA_SUCCESS)
	{
		printf("cuDeviceGet: %d\n", getCudaDrvErrorString(result));
		return 0;
	}

	//Get device of the current thread
	//result = cuCtxGetDevice (&cuDevice);
	result = cuDeviceGet (&cuDevice, 1);
	if (result != CUDA_SUCCESS)
	{
		printf("cuDeviceGet: %d\n", getCudaDrvErrorString(result));
		return 0;
	}


	result = cuCtxCreate(&cuCurrent, 0, cuDevice);
	if (result != CUDA_SUCCESS)
	{
		printf("cuCtxCreate: %d\n", getCudaDrvErrorString(result));
		return 0;
	}

	cuDeviceGetName(name, 100, cuDevice);
	cout<<"Name "<<name<<"  of dev: "<<cuDevice<<endl;

	/*   -- Get context of a thread -- */
	gettimeofday(&popContext_1,NULL);
	double t1 = popContext_1.tv_sec  * 1000000 +  popContext_1.tv_usec;
	{
		//Get context of the current thread
		result = cuCtxPopCurrent(&cuCurrent);
		if (result != CUDA_SUCCESS)
		{
			printf("cuCtxPopCurrent: %d\n", result);
			return 0;
		}
	}
	gettimeofday(&popContext_2,NULL);
	double t2 = popContext_2.tv_sec  * 1000000 +  popContext_2.tv_usec;                                
	long double durationofctx = (t2 - t1)/1000 ;
	cout<<"Duration of getContext: "<<durationofctx<<endl;

	/*   -- DONE: Get context of a thread -- */
	//prepareSaxpy();
	/*   -- Destroy context of a thread -- */
	gettimeofday(&cudaCallStart_1,NULL);
	double cuda_t1 = cudaCallStart_1.tv_sec  * 1000000 +  cudaCallStart_1.tv_usec;
	{
		result = cuCtxDestroy(cuCurrent);
		if (result != CUDA_SUCCESS)
		{
			printf("cuCtxDestroy: %d\n", result);
			return 0;
		}

	}
	gettimeofday(&cudaCallEnd_1,NULL);
	double cuda_t2 =  cudaCallEnd_1.tv_sec  * 1000000 +  cudaCallEnd_1.tv_usec;
	long double durationCudaCall = (cuda_t2 -cuda_t1)/1000 ;
	cout<<"Duration of destroy context: "<<durationCudaCall<<endl;
	/*   -- DONE: Destroy context of a thread -- */


	float *d_a;
	int size = sizeof(int);
	cudaError_t error;

	/*   -- Create context of a thread -- */
	/*
	gettimeofday(&cudaCallStart_2,NULL);
	double cuda_t3 = cudaCallStart_2.tv_sec  * 1000000 +  cudaCallStart_2.tv_usec;
	{
		result = cuCtxCreate(&cuCurrent, 0, cuDevice);
		if (result != CUDA_SUCCESS)
		{
			printf("cuCtxPopCreate: %d\n", result);
			return 0;
		}

	}
	gettimeofday(&cudaCallEnd_2,NULL);
	double cuda_t4 =  cudaCallEnd_2.tv_sec  * 1000000 +  cudaCallEnd_2.tv_usec;
	long double durationCudaCall2 = (cuda_t4 -cuda_t3)/1000 ;
	cout<<"Duration of create context: "<<durationCudaCall2<<endl;
*/
	gettimeofday(&cudaCallStart_3,NULL);
	double cuda_t5 = cudaCallStart_3.tv_sec  * 1000000 +  cudaCallStart_3.tv_usec;

	error = cudaMalloc((void**)&d_a,size);

	if (error != cudaSuccess)
	{
		cerr <<"Malloc failed"<<endl;
		return false;
	}
	gettimeofday(&cudaCallEnd_3,NULL);
	double cuda_t6 =  cudaCallEnd_3.tv_sec  * 1000000 +  cudaCallEnd_3.tv_usec;
	long double durationCudaCall3 = (cuda_t6 -cuda_t5)/1000 ;
	cout<<"Duration of 1st malloc: "<<durationCudaCall3<<endl;

	/*   -- DONE: Create context of a thread -- */
	return 1;
}
