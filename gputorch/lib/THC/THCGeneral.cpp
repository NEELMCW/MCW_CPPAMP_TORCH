#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"
#include "cl_manage.h"

cl_device_id mdevice=0;
cl_context mcontext=0;
cl_command_queue mqueue=0;

void THGPUInit()
{
  int count = 0;

  int device = 0;

  mdevice = Concurrency::getAllocator().device;
  mcontext = Concurrency::getAllocator().context;
  mqueue = Concurrency::getAllocator().getQueue();
  THGPUBlas_init(count, device);
}

void THGPUShutdown()
{
  THGPUBlas_shutdown();
}

void __THGPUCheck(int err, const char *file, const int line)
{
  if(err != 0)
  {
    THError("%s(%i) : cuda runtime error : %s",
            file, line, "");
  }
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
