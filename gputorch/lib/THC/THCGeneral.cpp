#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"

void THCudaInit()
{
  int count = 0;
  //THCudaCheck(cudaGetDeviceCount(&count));

  int device = 0;
  //THCudaCheck(cudaGetDevice(&device));

  /*state->rngState = (THCudaRNGState*)malloc(sizeof(THCudaRNGState));
  THCRandom_init(state->rngState, count, device);

  THCudaBlas_init(count, device);

  int i,j;
  for(i=0; i < count; ++i)
  {
    THCudaCheck(cudaSetDevice(i));
    for (j=0; j < count; ++j)
    {
      if(i != j)
      {
        int can = 0;
        THCudaCheck(cudaDeviceCanAccessPeer(&can, i, j));
        if(can)
          THCudaCheck(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }
  THCudaCheck(cudaSetDevice(device));*/
     mdevice = 0;
     mcontext = 0;
     mqueue = 0;
	
    cl_int err;
    cl_platform_id platform = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    int ret = 0;
    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &mdevice, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return;
    }
    props[1] = (cl_context_properties)platform;
    mcontext = clCreateContext(props, 1, &mdevice, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return;
    }
    mqueue = clCreateCommandQueue(mcontext, mdevice, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(mcontext);
        return;
    }
    THCudaBlas_init(count, device);
}

void THCudaShutdown()
{
  //THCRandom_shutdown(state->rngState);
  //free(state->rngState);
  THCudaBlas_shutdown();
}


void THCudaGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size)
{
  const int nThreadPerBlock = 256;
  long nBlockPerGrid = size / nThreadPerBlock;
  long nBlockPerColumn = 0L;
  long nBlockPerRow = 0L;

  if(size % nThreadPerBlock)
    nBlockPerGrid++;

  if(nBlockPerGrid <= 65535)
  {
    nBlockPerRow = nBlockPerGrid;
    nBlockPerColumn = 1;
  }
  else if(nBlockPerGrid <= (65355L * 65355L))
  {
    unsigned int uiSqrt = (unsigned int)(sqrt((float)nBlockPerGrid));
    nBlockPerRow = uiSqrt;
    nBlockPerColumn = uiSqrt;
    while((nBlockPerRow * nBlockPerColumn) < nBlockPerGrid)
      nBlockPerRow++;
  }
  else
    THError("too large vector for Cuda, sorry");

  *nBlockPerColumn_ = (int)nBlockPerColumn;
  *nBlockPerRow_ = (int)nBlockPerRow;
  *nThreadPerBlock_ = (int)nThreadPerBlock;
}
