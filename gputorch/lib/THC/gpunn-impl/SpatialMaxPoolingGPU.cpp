#include "amp.h"
#include"amp_math.h"

#ifndef SPATIAL_POOL_FPROP_CU
#define SPATIAL_POOL_FPROP_CU

#define MIN(a,b) (a) < (b) ? (a) : (b)
#define MAX(a,b) (a) > (b) ? (a) : (b)
#ifndef DIVUP
#define DIVUP(x,y) (((x) + (y) - 1) / (y))
#endif

class AvgPooler
{
 public:
  inline float operator()(const float a, const float b) const restrict(amp)
  {
    return a + b;
  }

  inline float getBaseValue() const restrict (amp)
  {
    return 0;
  }

  inline float output(const float a, const int regionSize) const
  {
    return a / regionSize;
  }
};

class MaxPooler {
 public:
  inline float operator()(const float a, const float b) const restrict(amp){
    return Concurrency::fast_math::fmaxf(a, b);
  }

  inline float getBaseValue()  const restrict(amp){
    return -2e38;
  }

  inline float output(const float a, const int regionSize) const restrict(amp){
    return a;
  }
};

class MaxAbsPooler
{
 public:
  inline float operator()(const float a, const float b) const restrict(amp){
    return fabsf(a) > fabsf(b) ? a : b;
  }

  inline float getBaseValue() const restrict(amp) {
    return 0.0f;
  }

  inline float output(const float a, const int regionSize) const restrict(amp) {
    return a;
  }
};

/*
 * Block size B_YxB_X
 * tidx.tile[1] determines output.x, image idx in batches of B_X*imgsPerThread
 * tidx.tile[0] determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * tidx.local[1] determines img idx
 * tidx.local[0] determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
void kLocalPool(Concurrency::array_view<float,1> &avImages, long imgOffset,
                Concurrency::array_view<float,1> &avTargets, long targetOffset,
                int imgSize, int numFilters, int numImages, int subsX,
                int startX, int strideX, int outputsX, Agg agg, int blockX, int blockY)
{
  Concurrency::extent<3> grdExt(1, blockY * 4, blockX * 32);
  Concurrency::tiled_extent<1, 4, 32> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 4, 32> tidx) restrict(amp)
  {
    float *imgs = avImages.data() + imgOffset;
    float *target = avTargets.data() + targetOffset;
    const int numImgBlocks = DIVUP(numImages,B_X * imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y * filtersPerThread);
    const int outputIdxX = tidx.tile[2] / numImgBlocks;
    const int outputIdxY = tidx.tile[1] / numFilterBlocks;
    const int blockImgIdx = (tidx.tile[2] % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (tidx.tile[1] % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + tidx.local[1] * filtersPerThread);

    if (myFilterIdx >= numFilters)
      return;

    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + tidx.local[2];

    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;

    float prod[filtersPerThread][imgsPerThread];
    for (int f = 0; f < filtersPerThread; f++)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        prod[f][i] = agg.getBaseValue();
      }
    }

    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    for (int y = loopStartY; y < loopEndY; y++)
    {
      for (int x = loopStartX; x < loopEndX; x++)
      {
        const int imgPx = y * imgSize + x;
        for (int i = 0; i < imgsPerThread; i++)
        {
          if (!checkCaseBounds || imgIdx + i * B_X < numImages)
          {
            for (int f = 0; f < filtersPerThread; f++)
            {
              prod[f][i] = agg(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
            }
          }
        }
      }
    }

    for (int i = 0; i < imgsPerThread; i++)
    {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages)
      {
        for (int f = 0; f < filtersPerThread; f++)
        {
          target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize);
        }
      }
    }
  });
}


/*
 * Block size 16xB_X
 * tidx.tile[1] determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * tidx.tile[0] determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does a 4x4 region for some number of images/filters.
 * 
 * tidx.local[1] determines img idx
 * tidx.local[0] determines pixel idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 * 
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
void kLocalPool2(Concurrency::array_view<float,1> &avImages, long imgOffset,
                 Concurrency::array_view<float,1> &avTargets, long targetOffset,
                 int imgSize, int numFilters, int numImages, int subsX, int startX,
                 int outputsX, Agg agg, int blockX, int blockY)
{
  Concurrency::extent<3> grdExt(1, blockY * 16, blockX * 8);
  Concurrency::tiled_extent<1, 16, 8> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 16, 8> tidx) restrict(amp)
  {
    tile_static float shImgs[filtersPerThread][B_X * imgsPerThread];
    float *imgs = avImages.data() + imgOffset;
    float *target = avTargets.data() + targetOffset;

    const int numImgBlocks = DIVUP(numImages,B_X * imgsPerThread);
    const int numFilterBlocks = numFilters / (filtersPerThread);
    const int blockOutputX = 4 * (tidx.tile[2] / numImgBlocks);
    const int blockOutputY = 4 * (tidx.tile[1] / numFilterBlocks);
    const int blockImgIdx = (tidx.tile[2] % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (tidx.tile[1] % numFilterBlocks) * filtersPerThread;

    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int t_idx = tidx.local[1] * B_X + tidx.local[2];
    const int loadY = t_idx / 32, loadX = t_idx % 32;

    const int myX = tidx.local[1] % 4;
    const int myY = tidx.local[1] / 4;

    const int myOutputIdxY = blockOutputY + myY;
    const int myOutputIdxX = blockOutputX + myX;
    const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;

    const int startImgPxX = startX + blockOutputX;
    const int startImgPxY = startX + blockOutputY;
    const int endImgPxX = startImgPxX + subsX;
    const int endImgPxY = startImgPxY + subsX;

    const int myStartImgPxY = startImgPxY + myY;
    const int myStartImgPxX = startImgPxX + myX;
    const int myEndImgPxY = endImgPxY + myY;
    const int myEndImgPxX = endImgPxX + myX;

    const int loopStartY = MAX(startImgPxY, 0);
    const int loopStartX = MAX(startImgPxX, 0);
    const int loopEndY = MIN(imgSize, endImgPxY + 3);
    const int loopEndX = MIN(imgSize, endImgPxX + 3);
    const int imgIdx = blockImgIdx + tidx.local[2];

    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;

    float prod[filtersPerThread][imgsPerThread];
    for (int f = 0; f < filtersPerThread; f++)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        prod[f][i] = agg.getBaseValue();
      }
    }
    int regionSize = 0;
    for (int y = loopStartY; y < loopEndY; y++)
    {
      const bool isInY = y >= myStartImgPxY && y < myEndImgPxY;
      for (int x = loopStartX; x < loopEndX; x++)
      {
        // Load a pixel
        const int px = y * imgSize + x;
        for (int ly = 0; ly < filtersPerThread; ly += B_X / 2)
        {
          if (filtersPerThread % (B_X / 2) == 0 || ly + loadY < filtersPerThread)
          {
            for (int lx = 0; lx < B_X * imgsPerThread; lx += 32)
            {
              if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages)
              {
                shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
              }
            }
          }
        }
        tidx.barrier.wait();

        // Is this pixel in my region?
        if (isInY && x >= myStartImgPxX && x < myEndImgPxX)
        {
          for (int i = 0; i < imgsPerThread; i++)
          {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages)
            {
              for (int f = 0; f < filtersPerThread; f++)
              {
                prod[f][i] = agg(prod[f][i], shImgs[f][tidx.local[2] + i * B_X]);
              }
            }
          }
          ++regionSize;
        }
        tidx.barrier.wait();
      }
    }

    if (myOutputIdxY < outputsX && myOutputIdxX < outputsX)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages)
        {
          for (int f = 0; f < filtersPerThread; f++)
          {
            target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize);
          }
        }
      }
    }
  });
}

/*
 * images:       (numFilters, imgPixels, numImages)
 * targets:      (numFilters, outputs, numImages)
 */
template<class Pooler>
void spatialMaxPooling_updateOutput
(
 // raw pointers:
 Concurrency::array_view<float,1>&images, long imgOffset,
 Concurrency::array_view<float,1>&targets, long targetOffset,
 // numImgColors == numFilters
 int numFilters, int imgSizeY, int imgSizeX, int numImages,
 // numModulesY == numModulesX == outputsX
 int numModulesY, int numModulesX,
 // kH == kW == subsXs
 int filterSizeY, int filterSizeX,
 // 0 == startX, dW == dH == strideX
 int paddingStart, int moduleStride)
{
  MaxPooler pooler;

  int imgPixels = imgSizeY * imgSizeX;
  int imgSize = int(sqrt((float)imgPixels));
 /// TODO SQUARE !
  assert(imgSize * imgSize == imgPixels);

  int subsX = filterSizeX;
  assert(filterSizeX == filterSizeY);

  int startX = paddingStart;
  int strideX = moduleStride;

  int outputsX = numModulesX;
  // int outputs = numModulesY * numModulesX;
  /// TODO SQUARE !
  assert(numModulesY == numModulesX);

  if (strideX == 1 && subsX >= 6)
  {
    int imgsPerThread = numImages % 128 == 0 ? 8 : 4;
    int filtersPerThread = numFilters % 4 == 0 ? 4 : numFilters % 3 == 0 ? 3 : numFilters % 2 == 0 ? 2 : 1;
    int bx = 8;
    bool checkCaseBounds = numImages % (bx * imgsPerThread) != 0;
    assert((imgsPerThread * bx) % 32 == 0);
    assert(numFilters % filtersPerThread == 0);

    int blockX, blockY;
    blockX = DIVUP(outputsX, 4) * DIVUP(numImages, bx * imgsPerThread);
    blockY = DIVUP(outputsX, 4) * numFilters / filtersPerThread;

    if (imgsPerThread == 8)
    {
      if (filtersPerThread == 1)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 1, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 1, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                              startX, outputsX, pooler, blockX, blockY);
        }
      }
      else if (filtersPerThread == 2)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 2, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 2, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                              startX, outputsX, pooler, blockX, blockY);
        }
      }
      else if (filtersPerThread == 3)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 3, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 3, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                               startX, outputsX, pooler,blockX, blockY);
        }
      }
      else if (filtersPerThread == 4)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 4, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 8, 4, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                              startX, outputsX, pooler, blockX, blockY);
        }
      }
    }
    else if (imgsPerThread == 4)
    {
      if (filtersPerThread == 1)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 1, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 1, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                              startX, outputsX, pooler, blockX, blockY);
        }
      }
      else if (filtersPerThread == 2)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 2, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 2, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                              startX, outputsX, pooler, blockX, blockY);
        }
      }
      else if (filtersPerThread == 3)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 3, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 3, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                              startX, outputsX, pooler, blockX, blockY);
        }
      }
      else if (filtersPerThread == 4)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, true>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 4, true>(images, imgOffset, targets, targetOffset,
                                             imgSize, numFilters, numImages, subsX,
                                             startX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, false>, gpuFuncCachePreferShared);
          kLocalPool2<Pooler, 8, 4, 4, false>(images, imgOffset, targets, targetOffset,
                                              imgSize, numFilters, numImages, subsX,
                                              startX, outputsX, pooler, blockX, blockY);
        }
      }
    }
  }
  else
  {
    int filtersPerThread = numFilters % 8 == 0 ? 2 : 1;
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;

    int blockX, blockY;
    blockX = DIVUP(numImages,32*imgsPerThread) * outputsX;
    blockY = DIVUP(numFilters, 4 * filtersPerThread) * outputsX;

    if (imgsPerThread == 4)
    {
      if (filtersPerThread == 1)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, true>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 4, 1, true>(images, imgOffset, targets, targetOffset,
                                                imgSize, numFilters, numImages, subsX, startX,
                                                strideX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, false>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 4, 1, false>(images, imgOffset, targets, targetOffset,
                                                 imgSize, numFilters, numImages, subsX, startX,
                                                 strideX, outputsX, pooler, blockX, blockY);
        }
      }
      else
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, true>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 4, 2, true>(images, imgOffset, targets, targetOffset,
                                                imgSize, numFilters, numImages, subsX, startX,
                                                strideX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, false>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 4, 2, false>(images, imgOffset, targets, targetOffset,
                                                 imgSize, numFilters, numImages, subsX, startX,
                                                 strideX, outputsX, pooler, blockX, blockY);
        }
      }
    }
    else if (imgsPerThread == 2)
    {
      if (filtersPerThread == 1)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, true>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 2, 1, true>(images, imgOffset, targets, targetOffset,
                                                imgSize, numFilters, numImages, subsX, startX,
                                                strideX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, false>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 2, 1, false>(images, imgOffset, targets, targetOffset,
                                                 imgSize, numFilters, numImages, subsX, startX,
                                                 strideX, outputsX, pooler, blockX, blockY);
        }
      }
      else
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 2, true>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 2, 2, true>(images, imgOffset, targets, targetOffset,
                                                imgSize, numFilters, numImages, subsX, startX,
                                                strideX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 2, false>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 2, 2, false>(images, imgOffset, targets, targetOffset,
                                                 imgSize, numFilters, numImages, subsX, startX,
                                                 strideX, outputsX, pooler, blockX, blockY);
        }
      }
    }
    else
    {
      if (filtersPerThread == 1)
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, true>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 1, 1, true>(images, imgOffset, targets, targetOffset,
                                                imgSize, numFilters, numImages, subsX, startX,
                                                strideX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, false>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 1, 1, false>(images, imgOffset, targets, targetOffset,
                                                 imgSize, numFilters, numImages, subsX, startX,
                                                 strideX, outputsX, pooler, blockX, blockY);
        }
      }
      else
      {
        if (checkCaseBounds)
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 2, true>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 1, 2, true>(images, imgOffset, targets, targetOffset,
                                                imgSize, numFilters, numImages, subsX, startX,
                                                strideX, outputsX, pooler, blockX, blockY);
        }
        else
        {
          //gpuFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 2, false>, gpuFuncCachePreferL1);
          kLocalPool<Pooler, 4, 32, 1, 2, false>(images, imgOffset, targets, targetOffset,
                                                 imgSize, numFilters, numImages, subsX, startX,
                                                 strideX, outputsX, pooler, blockX, blockY);
        }
      }
    }
  }
}

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
inline void kLocalMaxUndo(Concurrency::array_view<float,1> &avImages,
                          Concurrency::array_view<float,1> &avMaxGrads,
                          Concurrency::array_view<float,1> &avMaxActs,
                          Concurrency::array_view<float,1> &avTargets,
                          int imgSize, int numFilters, int numImages, int subsX,
                          int startX, int strideX, int outputsX, float scaleTargets,
                          float scaleOutputs, int blockX, int blockY)
{
  Concurrency::extent<3> grdExt(1, blockY * 4, blockX * 32);
  Concurrency::tiled_extent<1, 4, 32> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 4, 32> tidx) restrict(amp)
  {
    tile_static float shImgs[B_Y * filtersPerThread][B_X * imgsPerThread];
    float* imgs = avImages.data();
    float* maxGrads = avMaxGrads.data();
    float* maxActs = avMaxActs.data();
    float* target = avTargets.data();

    const int numImgBlocks = DIVUP(numImages,B_X * imgsPerThread);
    const int blockPxX = tidx.tile[2] / numImgBlocks;
    const int blockPxY = tidx.tile[1] / (numFilters / (B_Y * filtersPerThread));

    const int blockImgIdx = (tidx.tile[2] % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (tidx.tile[1] % (numFilters / (B_Y * filtersPerThread))) * B_Y * filtersPerThread;

    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);

    const int imgIdx = blockImgIdx + tidx.local[2];

    imgs += ((blockFilterIdx + tidx.local[1]) * imgPixels + blockPx) * numImages + imgIdx;
    maxGrads += ((blockFilterIdx + tidx.local[1]) * numOutputs) * numImages + imgIdx;
    maxActs += ((blockFilterIdx + tidx.local[1]) * numOutputs) * numImages + imgIdx;

    target += ((blockFilterIdx + tidx.local[1]) * imgPixels + blockPx) * numImages + imgIdx;

    float prod[filtersPerThread][imgsPerThread];
    for (int f = 0; f < filtersPerThread; f++)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        prod[f][i] = 0;
      }
    }

    if (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX
        && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages)
        {
          for (int f = 0; f < filtersPerThread; f++)
          {
            shImgs[tidx.local[1] + B_Y * f][tidx.local[2] + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
          }
        }
      }
      for (int my = startOutputY; my < endOutputY; my++)
      {
        for (int mx = startOutputX; mx < endOutputX; mx++)
        {
          const int outputIdx = my * outputsX + mx;
          for (int i = 0; i < imgsPerThread; i++)
          {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages)
            {
              for (int f = 0; f < filtersPerThread; f++)
              {
                const float ma = maxActs[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                const float mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                const float img = shImgs[tidx.local[1] + B_Y * f][tidx.local[2] + B_X * i];
                prod[f][i] += (img == ma) * mg;
              }
            }
          }
        }
      }
    }

    if (!add)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages)
        {
          for (int f = 0; f < filtersPerThread; f++)
          {
            target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
          }
        }
      }
    }
    else
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages)
        {
          for (int f = 0; f < filtersPerThread; f++)
          {
            target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
          }
        }
      }
    }
  });
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */

void spatialMaxPooling_updateGradInput
(
 // raw pointers:
 Concurrency::array_view<float,1>&images, long imOffset,
 Concurrency::array_view<float,1>&maxgrads, long gradOffset,
 Concurrency::array_view<float,1>&maxacts, long actsOffset,
 Concurrency::array_view<float,1>&targets, long targetOffset,
 // numImgColors == numFilters
 int numFilters, int imgSizeY, int imgSizeX, int numImages,
 // numModulesY == numModulesX == outputsX
 int numModulesY, int numModulesX, 
 // kH == kW == subsXs
 int filterSizeY, int filterSizeX, 
 // 0 == startX, dW == dH == strideX
 int paddingStart, int moduleStride, 
 // aux.
 float scaleTargets = 0, float scaleOutput = 1)
{ 
  int imgPixels = imgSizeY * imgSizeX;
  int imgSize = int(sqrt((float)imgPixels));
  // TODO: SQUARE !
  assert(imgSize * imgSize == imgPixels);

  int subsX = filterSizeX;
  assert(filterSizeX == filterSizeY);

  int startX = paddingStart;
  int strideX = moduleStride;
  int outputsX = numModulesX;

  // TODO: SQUARE !
  assert(numModulesY == numModulesX);
  assert(numFilters % 16 == 0);
  assert(strideX <= subsX);

  int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
  int checkCaseBounds = numImages % (32*imgsPerThread) != 0;

  int blockX, blockY;
  blockX = DIVUP(numImages,32 * imgsPerThread) * imgSize;
  blockY = (numFilters / (4 * 2)) * imgSize;

  if (imgsPerThread == 4)
  {
    if (checkCaseBounds)
    {
      if (scaleTargets == 0 && scaleOutput == 1)
      {
        kLocalMaxUndo<4, 32, 4, 2, false, true>(images, maxgrads, maxacts, targets,
                                                imgSize, numFilters, numImages, subsX,
                                                startX, strideX, outputsX, scaleTargets,
                                                scaleOutput, blockX, blockY);
      }
      else
      {
        kLocalMaxUndo<4, 32, 4, 2, true, true>(images, maxgrads, maxacts, targets,
                                               imgSize, numFilters, numImages, subsX,
                                               startX, strideX, outputsX, scaleTargets,
                                               scaleOutput, blockX, blockY);
      }
    }
    else
    {
      if (scaleTargets == 0 && scaleOutput == 1)
      {
        kLocalMaxUndo<4, 32, 4, 2, false, false>(images, maxgrads, maxacts, targets,
                                                 imgSize, numFilters, numImages, subsX,
                                                 startX, strideX, outputsX, scaleTargets,
                                                 scaleOutput, blockX, blockY);
      }
      else
      {
        kLocalMaxUndo<4, 32, 4, 2, true, false>(images, maxgrads, maxacts, targets,
                                                imgSize, numFilters, numImages, subsX,
                                                startX, strideX, outputsX, scaleTargets,
                                                scaleOutput, blockX, blockY);
      }
    }
  }
  else if (imgsPerThread == 2)
  {
    if (checkCaseBounds)
    {
      if (scaleTargets == 0 && scaleOutput == 1)
      {
        kLocalMaxUndo<4, 32, 2, 2, false, true>(images, maxgrads, maxacts, targets,
                                                imgSize, numFilters, numImages, subsX,
                                                startX, strideX, outputsX, scaleTargets,
                                                scaleOutput, blockX, blockY);
      }
      else
      {
        kLocalMaxUndo<4, 32, 2, 2, true, true>(images, maxgrads, maxacts, targets,
                                               imgSize, numFilters, numImages, subsX,
                                               startX, strideX, outputsX, scaleTargets,
                                               scaleOutput, blockX, blockY);
      }
    }
    else
    {
      if (scaleTargets == 0 && scaleOutput == 1)
      {
        kLocalMaxUndo<4, 32, 2, 2, false, false>(images, maxgrads, maxacts, targets,
                                                 imgSize, numFilters, numImages, subsX,
                                                 startX, strideX, outputsX, scaleTargets,
                                                 scaleOutput, blockX, blockY);
      }
      else
      {
        kLocalMaxUndo<4, 32, 2, 2, true, false>(images, maxgrads, maxacts, targets,
                                                imgSize, numFilters, numImages, subsX,
                                                startX, strideX, outputsX, scaleTargets,
                                                scaleOutput, blockX, blockY);
      }
    }
  }
  else
  {
    if (checkCaseBounds)
    {
      if (scaleTargets == 0 && scaleOutput == 1)
      {
        kLocalMaxUndo<4, 32, 1, 2, false, true>(images, maxgrads, maxacts, targets,
                                                imgSize, numFilters, numImages, subsX,
                                                startX, strideX, outputsX, scaleTargets,
                                                scaleOutput, blockX, blockY);
      }
      else
      {
        kLocalMaxUndo<4, 32, 1, 2, true, true>(images, maxgrads, maxacts, targets,
                                               imgSize, numFilters, numImages, subsX,
                                               startX, strideX, outputsX, scaleTargets,
                                               scaleOutput, blockX, blockY);
      }
    }
    else
    {
      if (scaleTargets == 0 && scaleOutput == 1)
      {
        kLocalMaxUndo<4, 32, 1, 2, false, false>(images, maxgrads, maxacts, targets,
                                                 imgSize, numFilters, numImages, subsX,
                                                 startX, strideX, outputsX, scaleTargets,
                                                 scaleOutput, blockX, blockY);
      }
      else
      {
        kLocalMaxUndo<4, 32, 1, 2, true, false>(images, maxgrads, maxacts, targets,
                                                imgSize, numFilters, numImages, subsX,
                                                startX, strideX, outputsX, scaleTargets,
                                                scaleOutput, blockX, blockY);
      }
    }
  }
}
#endif	/* SPATIAL_POOL_FPROP_CU */

static int gpunn_SpatialMaxPoolingGPU_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 4, 2, "4D (batch) tensor expected");

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nInputPlane = input->size[0];
  long batchSize = input->size[3];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  luaL_argcheck(L, THGPUTensor_isContiguous(input), 2, "input must be contiguous");
  
  THGPUTensor_resize4d(output, nInputPlane, nOutputRows, nOutputCols, batchSize);

  auto avInput = input->get_array_view();
  auto avOutput = output->get_array_view();

  spatialMaxPooling_updateOutput<MaxPooler> (avInput, input->storageOffset,
                                             avOutput, output->storageOffset,
                                             nInputPlane, nInputRows, nInputCols,
                                             batchSize, nOutputRows, nOutputCols,
                                             kH, kW, 0, dW);
  return 1;
}

static int gpunn_SpatialMaxPoolingGPU_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nInputPlane = input->size[0];
  long batchSize = input->size[3];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  THGPUTensor_resizeAs(gradInput, input);
  THGPUTensor_zero(gradInput);

  auto avInput = input->get_array_view();
  auto avOutput = output->get_array_view();
  auto avGradInput = gradInput->get_array_view();
  auto avGradOutput = gradOutput->get_array_view();

  spatialMaxPooling_updateGradInput (avInput, input->storageOffset,
                                     avGradOutput, gradOutput->storageOffset,
                                     avOutput, output->storageOffset,
                                     avGradInput, gradInput->storageOffset,
                                     nInputPlane, nInputRows, nInputCols, batchSize,
                                     nOutputRows, nOutputCols, kH, kW, 0, dW);
    return 1;
}

static const struct luaL_Reg gpunn_SpatialMaxPoolingGPU__ [] = {
  {"SpatialMaxPoolingGPU_updateOutput", gpunn_SpatialMaxPoolingGPU_updateOutput},
  {"SpatialMaxPoolingGPU_updateGradInput", gpunn_SpatialMaxPoolingGPU_updateGradInput},
  {NULL, NULL}
};

static void gpunn_SpatialMaxPoolingGPU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialMaxPoolingGPU__, "nn");
  lua_pop(L,1);
}
