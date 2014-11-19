#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"

void THCudaInit(void)
{
/*  if (cublasInit() != CUBLAS_STATUS_SUCCESS)
  THError("unable to initialize cublas");

  int count = 0;
  THCudaCheck(cudaGetDeviceCount(&count));

  int device = 0;
  THCudaCheck(cudaGetDevice(&device));

  THCRandom_init(count, device);

  int i, j;
  for (i = 0; i < count; ++i)
  {
    THCudaCheck(cudaSetDevice(i));
    for (j = 0; j < count; ++j)
    {
      if (i != j)
      {
        int can = 0;
        THCudaCheck(cudaDeviceCanAccessPeer(&can, i, j));
        if (can)
          THCudaCheck(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }
  THCudaCheck(cudaSetDevice(device));*/
}

void THCudaShutdown(void)
{
  THCRandom_shutdown();

/*  if (cublasShutdown() != CUBLAS_STATUS_SUCCESS)
      THError("unable to shutdown cublas");*/
}



void THCudaGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size)
{
  const int nThreadPerBlock = 256;
  long nBlockPerGrid = size;
  long nBlockPerColumn = 0L;
  long nBlockPerRow = 0L;
  int maxGrid1d = 65335;
  unsigned long long maxGrid = (unsigned long long)(maxGrid1d * maxGrid1d);

  if (size % nThreadPerBlock)
    nBlockPerGrid++;

  if (nBlockPerGrid <= 65535)
  {
    nBlockPerRow = nBlockPerGrid;
    nBlockPerColumn = 1;
  }
  else if (nBlockPerGrid <= maxGrid)
  {
    unsigned int uiSqrt = (unsigned int)(sqrt((float)nBlockPerGrid));
    nBlockPerRow = uiSqrt;
    nBlockPerColumn = uiSqrt;
    while ((nBlockPerRow * nBlockPerColumn) < nBlockPerGrid)
      nBlockPerRow++;
  }
  else
    THError("too large vector for Camp, sorry");

  *nBlockPerColumn_ = (int)(nBlockPerColumn + nThreadPerBlock - 1) &~(nThreadPerBlock - 1);
  *nBlockPerRow_ = (int)nBlockPerRow;
  *nThreadPerBlock_ = (int)nThreadPerBlock;
}

