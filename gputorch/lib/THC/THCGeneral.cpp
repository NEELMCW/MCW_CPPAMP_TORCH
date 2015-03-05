#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"
#include "cl_manage.h"


void THGPUInit()
{
  // To Be implemented
}

void THGPUShutdown()
{ 
   // To Be implemented
}

void __THGPUCheck(int err, const char *file, const int line)
{
  if (err != 0)
  {
    THError("%s(%i) : GPU runtime error : %s",
            file, line, "");
  }
}


void THGPUGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size)
{
  const int nThreadPerBlock = 256;
  long nBlockPerGrid = size / nThreadPerBlock;
  long nBlockPerColumn = 0L;
  long nBlockPerRow = 0L;

  if (size % nThreadPerBlock)
    nBlockPerGrid++;

  if (nBlockPerGrid <= 65535)
  {
    nBlockPerRow = nBlockPerGrid;
    nBlockPerColumn = 1;
  }
  else if (nBlockPerGrid <= (65355L * 65355L))
  {
    unsigned int uiSqrt = (unsigned int)(sqrt((float)nBlockPerGrid));
    nBlockPerRow = uiSqrt;
    nBlockPerColumn = uiSqrt;
    while ((nBlockPerRow * nBlockPerColumn) < nBlockPerGrid)
      nBlockPerRow++;
  }
  else
    THError("too large vector for GPU, sorry");

  *nBlockPerColumn_ = (int)nBlockPerColumn;
  *nBlockPerRow_ = (int)nBlockPerRow;
  *nThreadPerBlock_ = (int)nThreadPerBlock;
}
