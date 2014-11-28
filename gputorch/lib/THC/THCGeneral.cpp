#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"

cl_device_id mdevice=0;
cl_context mcontext=0;
cl_command_queue mqueue=0;

void THGPUInit()
{
  int count = 0;
  //THGPUCheck(gpuGetDeviceCount(&count));

  int device = 0;
  //THGPUCheck(gpuGetDevice(&device));

  /*state->rngState = (THGPURNGState*)malloc(sizeof(THGPURNGState));
  THCRandom_init(state->rngState, count, device);

  THGPUBlas_init(count, device);

  int i,j;
  for(i=0; i < count; ++i)
  {
    THGPUCheck(gpuSetDevice(i));
    for (j=0; j < count; ++j)
    {
      if(i != j)
      {
        int can = 0;
        THGPUCheck(gpuDeviceCanAccessPeer(&can, i, j));
        if(can)
          THGPUCheck(gpuDeviceEnablePeerAccess(j, 0));
      }
    }
  }
  THGPUCheck(gpuSetDevice(device));*/
	
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
    THGPUBlas_init(count, device);
}

void THGPUShutdown()
{
  //THCRandom_shutdown(state->rngState);
  //free(state->rngState);
  THGPUBlas_shutdown();
}


void THGPUGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size)
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
    THError("too large vector for GPU, sorry");

  *nBlockPerColumn_ = (int)nBlockPerColumn;
  *nBlockPerRow_ = (int)nBlockPerRow;
  *nThreadPerBlock_ = (int)nThreadPerBlock;
}
