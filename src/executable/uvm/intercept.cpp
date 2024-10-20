#include <cstdint>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>

cudaError_t cudaMalloc ( void** devPtr, size_t size )
{
  cudaError_t (*lcudaMallocManaged) ( void**, size_t, unsigned int) = (cudaError_t (*) ( void** , size_t, unsigned int  ))dlsym(RTLD_NEXT, "cudaMallocManaged");
  cudaError_t ret = lcudaMallocManaged( devPtr, size, cudaMemAttachGlobal );
  return ret;
}

cudaError_t cudaFree ( void* devPtr )
{
  cudaError_t (*lcudaFree) ( void*) = (cudaError_t (*) ( void*  ))dlsym(RTLD_NEXT, "cudaFree");
  return lcudaFree( devPtr );
}
