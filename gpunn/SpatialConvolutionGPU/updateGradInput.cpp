/* 
 * This code has been adapted from Alex Krizhevsky's GPU library
 * (http://code.google.com/p/gpu-convnet/), and was originally
 * licensed under a BSD license.
 */

/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread.
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numColors, filterPixels, numFilters)                               if conv
 *              (numModulesY, numModulesX, numColors, filterPixels, numFilters)     otherwise
 * targets:     (numColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * Number of filters must be divisible by 16.
 * Number of images must be divisible by 16*imgsPerThread  if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 */
template <int imgsPerThread, int numColors, bool scale, bool checkCaseBounds, bool conv>
void img_acts_color(THGPUTensor* hidActsTensor, THGPUTensor* filterTensor, THGPUTensor* targetTensor,
                    const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                    const int filterSize, const int imgSizeY, const int imgSizeX,
                    const int paddingStart, const int moduleStride,
                    const float scaleTargets, const float scaleOutputs,
                    int blockX, int blockY, int numFilterColors)
{
  Concurrency::array_view<float,1> avhidActs(Concurrency::extent<1>(hidActsTensor->storage->size), THGPUTensor_data(hidActsTensor));
  Concurrency::array_view<float,1> avFilters(Concurrency::extent<1>(filterTensor->storage->size), THGPUTensor_data(filterTensor));
  Concurrency::array_view<float,1> avTargets(Concurrency::extent<1>(targetTensor->storage->size), THGPUTensor_data(targetTensor));
#if (numFilterColors % 8 == 0)
  blockX = (blockX + 31) &~31;
  blockY = (blockY + 3) &~3;
  Concurrency::extent<3> grdExt(1, blockY, blockX);
  Concurrency::tiled_extent<1, 4, 32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 4, 32> tidx) restrict(amp)
#else
  blockX = (blockX + 15) &~15;
  blockY = (blockY + 15) &~15;
  Concurrency::extent<3> grdExt(1, blockY, blockX);
  Concurrency::tiled_extent<1, 16, 16> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 16, 16> tidx) restrict(amp)
#endif
  {
    float hidActs = 0;//avhidActs.data();
    float targets = 0;//avFilters.data();
    float filters = 0;//avTargets.data();

    tile_static float shFilters[numColors*16][16 + 1];
    tile_static float shHidActs[16][16*imgsPerThread];

    const int blockCaseIdx = tidx.tile[2] * 16*imgsPerThread;
    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int blockRegionIdx = tidx.tile[1];
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = tidx.local[1] / 4, pxXInRegion = tidx.local[1] % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSizeX + pxX;
    const bool isPxInImg = pxY < imgSizeY && pxX < imgSizeX;
    const int numModules = numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeX * imgSizeY;
    const int t_idx = tidx.local[1] * 16 + tidx.local[2];
    const int loadY = t_idx / 32, loadX = t_idx % 32;

    hidActs += blockCaseIdx + loadY * numImages * numModules + loadX;
    filters += tidx.local[2];
    targets += pxIdx * numImages + blockCaseIdx + tidx.local[2];


    float prod[numColors][imgsPerThread];
    for (int c = 0; c < numColors; c++)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        prod[c][i] = 0;
      }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);
    
    float* shilterLoad = &shFilters[tidx.local[1]][tidx.local[2]];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++)
    {
      const int moduleTop = paddingStart + my * moduleStride;
      const int pxInModuleY = pxY - moduleTop;

      for (int mx = startX; mx < endX; mx++)
      {
        const int moduleIdx = my * numModulesX + mx;
        const int moduleLeft = paddingStart + mx * moduleStride;
        const int pxInModuleX = pxX - moduleLeft;

        const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
        const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

        for (int f = 0; f < numFilters; f += 16)
        {   // multiply with 16 filters at a time
          // Now the threads split up into half-warps, and each half-warp decides if it's interested.
          // const float* hLoad = &avhidActs[hidActs + (moduleIdx + f * numModules) * numImages];

          for (int i = 0; i < imgsPerThread * 16; i += 32)
          {
            if (!checkCaseBounds || blockCaseIdx + i + loadX < numImages)
            {
              for (int j = 0; j < 16; j += 8)
              {   // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                shHidActLoad[j * 16 * imgsPerThread + i] = avhidActs[hidActs + (moduleIdx + f * numModules) * numImages + j * numModules * numImages + i];
              }
            }
            else
            {
              for (int j = 0; j < 16; j += 8)
              {   // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                shHidActLoad[j * 16 * imgsPerThread + i] = 0;
              }
            }
          }

          if (isPxInImg && isPxInModule)
          {
            // This half-warp is interested, so it's going to load the weights from this module to its pixel.
            // Not fully coalesced read :(
            // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
            //const float* fLoad = conv ? &avFilters[filters + pxIdxInModule * numFilters + f]
            //                          : &avFilters[filters + (moduleIdx * numColors * filterPixels + pxIdxInModule) * numFilters + f];

            for (int c = 0; c < numColors; c++)
            {
              shilterLoad[c * 16 * (16 + 1)] = (conv ? avFilters[filters + pxIdxInModule * numFilters + f + c * filterPixels * numFilters]:avFilters[filters + (moduleIdx * numColors * filterPixels + pxIdxInModule) * numFilters + f + c * filterPixels * numFilters]);
            }
          }

          tidx.barrier.wait();
          // Do some actual computation
          if (isPxInImg && isPxInModule)
          {
            for (int c = 0; c < numColors; c++)
            {
              for (int w = 0; w < 16; w++)
              {
                for (int i = 0; i < imgsPerThread; i++)
                {
                  prod[c][i] += shFilters[tidx.local[1] + c * 16][w] * shHidActs[w][tidx.local[2] + i * 16];
                }
              }
            }
          }
          tidx.barrier.wait();
        }
      }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg)
    {
      if (scale)
      {
        for (int i = 0; i < imgsPerThread; i++)
        {
          if (!checkCaseBounds || blockCaseIdx + tidx.local[2] + i * 16 < numImages)
          {
            for (int c = 0; c < numColors; c++)
            {
              avTargets[targets + c * imgPixels * numImages + i * 16] = scaleTargets * avTargets[ targets + c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
            }
          }
        }
      }
      else
      {
        for (int i = 0; i < imgsPerThread; i++)
        {
          if (!checkCaseBounds || blockCaseIdx + tidx.local[2] + i * 16 < numImages)
          {
            for (int c = 0; c < numColors; c++)
            {
              avTargets[ targets + c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
            }
          }
        }
      }
    }
  });
  avTargets.synchronize();
}
/*
 * Block size: 16x16.
 * blockIdx.x determines case in batches of 16*imgsPerThread, also color in batches of colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(16*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/colorsPerThread
 * blockIdx.y determines 4x4 image region in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines pixel.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one 4x4 pixels from 16*imgsPerThread cases.
 *
 * numImages must be divisible by 16*imgsPerThread if checkCaseBounds is false.
 * 16 * imgsPerThread must be divisible by 32.
 * numImageColors/numGroups must be divisible by colorsPerThread.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are 4-16 color channels.
 */
template <int imgsPerThread, int colorsPerThread,  bool scale, bool checkCaseBounds, bool conv>
void img_acts_mediumcolor(THGPUTensor* hidActsTensor, THGPUTensor* filterTensor, THGPUTensor* targetTensor,
                          const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                          const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart,
                          const int moduleStride, const int numImgColors, const int numGroups,
                          const float scaleTargets, const float scaleOutputs,
                          int blockX, int blockY)
{
  const int numFilterColors = numImgColors / numGroups;

  Concurrency::array_view<float,1> avhidActs(Concurrency::extent<1>(hidActsTensor->storage->size), THGPUTensor_data(hidActsTensor));
  Concurrency::array_view<float,1> avFilters(Concurrency::extent<1>(filterTensor->storage->size), THGPUTensor_data(filterTensor));
  Concurrency::array_view<float,1> avTargets(Concurrency::extent<1>(targetTensor->storage->size), THGPUTensor_data(targetTensor));
#if (numFilterColors % 8 == 0)
  blockX = (blockX + 31) &~31;
  blockY = (blockY + 3) &~3;
  Concurrency::extent<3> grdExt(1, blockY, blockX);
  Concurrency::tiled_extent<1, 4, 32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 4, 32> tidx) restrict(amp)
#else
  blockX = (blockX + 15) &~15;
  blockY = (blockY + 15) &~15;
  Concurrency::extent<3> grdExt(1, blockY, blockX);
  Concurrency::tiled_extent<1, 16, 16> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 16, 16> tidx) restrict(amp)
#endif
  {
    float hidActs = 0;//avhidActs.data();
    float targets = 0;//avFilters.data();
    float filters = 0;//avTargets.data();

    tile_static float shFilters[colorsPerThread*16][16 + 1];
    tile_static float shHidActs[16][16*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,16*imgsPerThread);
    const int blockCaseIdx = (tidx.tile[2] % numImgBlocks) * 16*imgsPerThread;

    const int imgColorIdx = (tidx.tile[2] / numImgBlocks) * colorsPerThread; // color idx globally
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;
    
    const int numRegionsX = DIVUP(imgSizeX, 4);
    const int blockRegionIdx = tidx.tile[1];
    const int blockRegionIdxX = blockRegionIdx % numRegionsX;
    const int blockRegionIdxY = blockRegionIdx / numRegionsX;
    const int blockRegionLeft = blockRegionIdxX * 4;
    const int blockRegionTop = blockRegionIdxY * 4;
    const int pxYInRegion = tidx.local[1] / 4, pxXInRegion = tidx.local[1] % 4;
    const int pxY = blockRegionTop + pxYInRegion;
    const int pxX = blockRegionLeft + pxXInRegion;
    const int pxIdx = pxY * imgSizeX + pxX;
    const bool isPxInImg = pxY < imgSizeY && pxX < imgSizeX;
    const unsigned int numModules = numModulesY * numModulesX;
    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int t_idx = tidx.local[1] * 16 + tidx.local[2];
    const int loadY = t_idx / 32, loadX = t_idx % 32;

    hidActs += blockCaseIdx + (blockFilterIdx + loadY) * numImages * numModules + loadX;
    filters += blockFilterIdx + filterColorIdx * filterPixels * numFilters + tidx.local[2];
    targets += imgColorIdx * imgPixels * numImages + pxIdx * numImages + blockCaseIdx + tidx.local[2];

    float prod[colorsPerThread][imgsPerThread];
    for (int c = 0; c < colorsPerThread; c++)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        prod[c][i] = 0;
      }
    }
    const int startY = blockRegionTop - paddingStart < filterSize ? 0
                        : 1 + (blockRegionTop - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockRegionTop + 3 - paddingStart) / moduleStride);
    const int startX = blockRegionLeft - paddingStart < filterSize ? 0
                        : 1 + (blockRegionLeft - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockRegionLeft + 3 - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[tidx.local[1]][tidx.local[2]];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    for (int my = startY; my < endY; my++)
    {
      const int moduleTop = paddingStart + my * moduleStride;
      const int pxInModuleY = pxY - moduleTop;

      for (int mx = startX; mx < endX; mx++)
      {
        const int moduleIdx = my * numModulesX + mx;
        const int moduleLeft = paddingStart + mx * moduleStride;
        const int pxInModuleX = pxX - moduleLeft;

        const bool isPxInModule = pxInModuleY >= 0 && pxInModuleY < filterSize && pxInModuleX >= 0 && pxInModuleX < filterSize;
        const int pxIdxInModule = pxInModuleY * filterSize + pxInModuleX;

        for (int f = 0; f < numFiltersPerGroup; f += 16)
        {
          // multipply with 16 filters at a time
          // Now the threads split up into half-warps, and each half-warp decides if it's interested.
          //const float* hLoad = &avhidActs[hidActs + (moduleIdx + f * numModules) * numImages];
          for (int i = 0; i < imgsPerThread * 16; i += 32)
          {
            if (!checkCaseBounds || blockCaseIdx + loadX + i < numImages)
            {
              for (int j = 0; j < 16; j += 8)
              {
                // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                shHidActLoad[j * 16 * imgsPerThread + i] = avhidActs[hidActs + (moduleIdx + f * numModules) * numImages +j * numModules * numImages + i];
              }
            }
            else
            {
              for (int j = 0; j < 16; j += 8)
              { // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                shHidActLoad[j * 16 * imgsPerThread + i] = 0;
              }
            }
          }

          if (isPxInImg && isPxInModule) {
            // This half-warp is interested, so it's going to load the weights from this module to its pixel.
         
            // Not fully coalesced read :(
            // But taking out this read entirely only reduces the runtime by ~2.8%, so it isn't costing me much.
            // const float* fLoad = conv ? &avFilters[filters +pxIdxInModule * numFilters + f]
            //                           : &avFilters[ filters +moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInModule * numFilters + f];
            for (int c = 0; c < colorsPerThread; c++)
            {
              shFilterLoad[c * 16 * (16 + 1)] = (conv ? avFilters[filters +pxIdxInModule * numFilters + f + c * filterPixels * numFilters]: avFilters[ filters +moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInModule * numFilters + f + c * filterPixels * numFilters]);
            }
          }
          tidx.barrier.wait();
          // Do some actual computation
          if (isPxInImg && isPxInModule)
          {
            for (int c = 0; c < colorsPerThread; c++)
            {
              for (int w = 0; w < 16; w++)
              {
                for (int i = 0; i < imgsPerThread; i++)
                {
                  prod[c][i] += shFilters[tidx.local[1] + c * 16][w] * shHidActs[w][tidx.local[2] + i * 16];
                }
              }
            }
          }
          tidx.barrier.wait();
        }
      }
    }
    // Not fully coalesced write :(... shmem (and fully coalesced) version is actually slightly slower, though
    if (isPxInImg)
    {
      if (scale)
      {
        for (int i = 0; i < imgsPerThread; i++)
        {
          if (!checkCaseBounds || blockCaseIdx + tidx.local[2] + i * 16 < numImages)
          {
            for (int c = 0; c < colorsPerThread; c++)
            {
              avTargets[targets + c * imgPixels * numImages + i * 16] = scaleTargets * avTargets[targets + c * imgPixels * numImages + i * 16] + scaleOutputs * prod[c][i];
            }
          }
        }
      }
      else
      {
        for (int i = 0; i < imgsPerThread; i++)
        {
          if (!checkCaseBounds || blockCaseIdx + tidx.local[2] + i * 16 < numImages)
          {
            for (int c = 0; c < colorsPerThread; c++)
            {
              avTargets[ targets + c * imgPixels * numImages + i * 16] = scaleOutputs * prod[c][i];
            }
          }
        }
      }
    }
  });
  avTargets.synchronize();
}

/*
 * Block size: B_YxB_X.
 * blockIdx.x determines case in batches of B_X*imgsPerThread, also color in batches of B_Y*colorsPerThread.
 *  In essence, blockIdx.x.x = 1..numImages/(B_X*imgsPerThread)
 *              blockIdx.x.y = 1..numImgColors/(B_Y*colorsPerThread)
 * blockIdx.y determines image pixel in target image.
 *
 * threadIdx.x determines case.
 * threadIdx.y determines color.
 *
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)                             if conv
 *              (numModulesY, numModulesX, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgSizeY, imgSizeX, numImages)
 *
 * Each block reconstructs one B_Y*colorsPerThread colors from 1 pixel from B_X*imgsPerThread cases.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false.
 * numFiltersPerGroup must be divisible by 16.
 * 
 * B_X * imgsPerThread must be divisible by 32.
 * numFilterColors must be divisible by B_Y*colorsPerThread.
 * B_X*B_Y must be divisible by 32.
 *
 * This version loads 32 cases at a time, so it gets full coalescing on that load.
 * It only loads 16 weights at a time, so those aren't fully coalesced.
 * This version conserves shared memory by loading 16 filters at a time rather than 32.
 * 
 * To be used when there are >= 16 color channels.
 */
template <int B_Y, int B_X, int imgsPerThread, int colorsPerThread, bool scale, bool checkCaseBounds, bool conv>
void conv_img_acts_manycolor(THGPUTensor* hidActsTensor, THGPUTensor* filterTensor, THGPUTensor* targetTensor,
                             const int numModulesY, const int numModulesX, const int numImages, const int numFilters,
                             const int filterSize, const int imgSizeY, const int imgSizeX, const int paddingStart, const int moduleStride,
                             const int numImgColors, const int numGroups, const float scaleTargets, const float scaleOutputs,
                             int blockX, int blockY)
{
  const int numFilterColors = numImgColors / numGroups;

  Concurrency::array_view<float,1> avhidActs(Concurrency::extent<1>(hidActsTensor->storage->size), THGPUTensor_data(hidActsTensor));
  Concurrency::array_view<float,1> avFilters(Concurrency::extent<1>(filterTensor->storage->size), THGPUTensor_data(filterTensor));
  Concurrency::array_view<float,1> avTargets(Concurrency::extent<1>(targetTensor->storage->size), THGPUTensor_data(targetTensor));
#if (numFilterColors % 8 == 0)
  blockX = (blockX + 31) &~31;
  blockY = (blockY + 3) &~3;
  Concurrency::extent<3> grdExt(1, blockY, blockX);
  Concurrency::tiled_extent<1, 4, 32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 4, 32> tidx) restrict(amp)
#else
  blockX = (blockX + 15) &~15;
  blockY = (blockY + 15) &~15;
  Concurrency::extent<3> grdExt(1, blockY, blockX);
  Concurrency::tiled_extent<1, 16, 16> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 16, 16> tidx) restrict(amp)
#endif
  {
    float hidActs = 0; //avhidActs.data();
    float targets = 0; //avFilters.data();
    float filters = 0; //avTargets.data();

    tile_static float shFilters[colorsPerThread*B_Y][16 + 1]; // TODO: perhaps reconsider this 16
    tile_static float shHidActs[16][B_X*imgsPerThread];

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockCaseIdx = (tidx.tile[2] % numImgBlocks) * B_X*imgsPerThread;
    const int imgColorIdx = (tidx.tile[2] / numImgBlocks) * B_Y*colorsPerThread; // color idx globally
    const int numFilterColors = numImgColors / numGroups;
    const int blockGroupIdx = imgColorIdx / numFilterColors;
    const int filterColorIdx = imgColorIdx % numFilterColors; // color idx within group
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockFilterIdx = blockGroupIdx * numFiltersPerGroup;

    const int blockPixelIdx = tidx.tile[1];
    const int blockPixelIdxX = blockPixelIdx % imgSizeX;
    const int blockPixelIdxY = blockPixelIdx / imgSizeX;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;
    const int t_idx = tidx.local[1] * B_X + tidx.local[2];
    const int hidActLoadY = t_idx / 32, hidActLoadX = t_idx % 32;
    const int filtersLoadY = t_idx / 16, filtersLoadX = t_idx % 16;
    const int numModules = numModulesY * numModulesX;

    hidActs += blockCaseIdx + (blockFilterIdx + hidActLoadY) * numImages * numModules + hidActLoadX;
    filters += blockFilterIdx + (filterColorIdx + filtersLoadY) * filterPixels * numFilters + filtersLoadX;
    targets += (imgColorIdx + tidx.local[1]) * imgPixels * numImages + blockPixelIdx * numImages + blockCaseIdx + tidx.local[2];

    float prod[colorsPerThread][imgsPerThread];
    for (int c = 0; c < colorsPerThread; c++)
    {
        for (int i = 0; i < imgsPerThread; i++)
        {
            prod[c][i] = 0;
        }
    }

    const int startY = blockPixelIdxY - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxY - paddingStart - filterSize) / moduleStride;
    const int endY = MIN(numModulesY, 1 + (blockPixelIdxY - paddingStart) / moduleStride);
    const int startX = blockPixelIdxX - paddingStart < filterSize ? 0
                        : 1 + (blockPixelIdxX - paddingStart - filterSize) / moduleStride;
    const int endX = MIN(numModulesX, 1 + (blockPixelIdxX - paddingStart) / moduleStride);

    float* shFilterLoad = &shFilters[filtersLoadY][filtersLoadX];
    float* shHidActLoad = &shHidActs[hidActLoadY][hidActLoadX];

    for (int my = startY; my < endY; my++)
    {
      const int moduleTop = paddingStart + my * moduleStride;
      const int pxInFilterY = blockPixelIdxY - moduleTop;

      for (int mx = startX; mx < endX; mx++)
      {
        const int moduleIdx = my * numModulesX + mx;
        const int moduleLeft = paddingStart + mx * moduleStride;
        const int pxInFilterX = blockPixelIdxX - moduleLeft;
            
        const int pxIdxInFilter = pxInFilterY * filterSize + pxInFilterX;

        for (int f = 0; f < numFiltersPerGroup; f += 16)
        {
          // multiply with 16 filters at a time
          //const float* hLoad = &avhidActs[hidActs +(moduleIdx + f * numModules) * numImages];
          for (int i = 0; i < imgsPerThread * B_X; i += 32)
          {
            if (!checkCaseBounds || blockCaseIdx + hidActLoadX + i < numImages)
            {
              for (int j = 0; j < 16; j += B_X*B_Y/32)
              {
                // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                shHidActLoad[j * B_X * imgsPerThread + i] = avhidActs[hidActs +(moduleIdx + f * numModules) * numImages +j * numModules * numImages + i];
              }
            }
            else
            {
              for (int j = 0; j < 16; j += B_X*B_Y/32)
              {
                // load 16 rows of imgsPerThread*16 cols, 8 * 32 elements at a time.
                shHidActLoad[j * B_X * imgsPerThread + i] = 0;
              }
            }
          }
          //const float* fLoad = conv ? &avFilters[filters + pxIdxInFilter * numFilters + f]
          //                          : &avFilters[filters + moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f];
          for (int i = 0; i < colorsPerThread*B_Y; i+= B_X*B_Y/16)
          {
            if ((colorsPerThread*B_Y) % (B_X*B_Y/16) == 0 || i + filtersLoadY < colorsPerThread*B_Y)
            {
              shFilterLoad[i * (16 + 1)] = (conv ? avFilters[filters + pxIdxInFilter * numFilters + f + i * filterPixels * numFilters]:avFilters[filters + moduleIdx * numFilterColors * filterPixels * numFilters + pxIdxInFilter * numFilters + f +  i * filterPixels * numFilters]);
            }
          }
          tidx.barrier.wait();
          // Do some actual computation
          for (int c = 0; c < colorsPerThread; c++)
          {
            for (int w = 0; w < 16; w++)
            {
              for (int i = 0; i < imgsPerThread; i++)
              {
                prod[c][i] += shFilters[c * B_Y + tidx.local[1]][w] * shHidActs[w][tidx.local[2] + i * B_X];
              }
            }
          }
          tidx.barrier.wait();
        }
      }
    }
    if (scale)
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        if (!checkCaseBounds || blockCaseIdx + tidx.local[2] + i * B_X < numImages)
        {
          for (int c = 0; c < colorsPerThread; c++)
          {
            avTargets[targets + c * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * avTargets[ targets + c * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[c][i];
          }
        }
      }
    }
    else
    {
      for (int i = 0; i < imgsPerThread; i++)
      {
        if (!checkCaseBounds || blockCaseIdx + tidx.local[2] + i * B_X < numImages)
        {
          for (int c = 0; c < colorsPerThread; c++)
          {
            avTargets[ targets + c * B_Y * imgPixels * numImages + i * B_X] = scaleOutputs * prod[c][i];
          }
        }
      }
    }
  });
  avTargets.synchronize();
}

/*
 * hidActs:         (numFilters, numModules, numImages)
 * filters:         (numFilterColors, filterPixels, numFilters)               if conv
 *                  (numModules, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:         (overSample, numImgColors, imgPixels, numImages)
 * 
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128. 
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast. 
 */
void spatialConv_updateGradInput( THGPUTensor *hidActs, THGPUTensor *filters, THGPUTensor *targets, int numImgColors,
                                  int imgSizeY, int imgSizeX, int numImages,int numFilters,int numModulesY,
                                  int numModulesX, int filterSizeY, int filterSizeX, int paddingStart,
                                  int moduleStride, float scaleTargets, float scaleOutput, bool conv)
{
  int numGroups = 1;
  int numFilterColors = numImgColors / numGroups;
  /* int filterModuleMult = conv ? 1 : numModules; */
  int filterSize = filterSizeX;
  int imgPixels = imgSizeY * imgSizeX;
    
  assert(numImgColors % numGroups == 0);
  assert(numFilters % (16*numGroups) == 0);
  assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
  assert(numGroups == 1 || numFilterColors % 4 == 0);

  assert(filterSizeX == filterSizeY);
  assert(numModulesY == numModulesX);
    
  assert(paddingStart <= 0);
  assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
  assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
  assert(moduleStride <= filterSize);

  int blockX, blockY;
  int threadX = 16;
  int threadY = 16;
  int colorsPerThread;
  int imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
  if (numFilterColors % 8 == 0)
  {
      //threads = dim3(32, 4);
      threadX = 32;
      threadY = 4;
      colorsPerThread = numFilterColors % 16 == 0 ? 4 : 2;
      imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
      assert(numFilterColors % (threadY * colorsPerThread) == 0);
        
      blockX = DIVUP(numImages, threadX * imgsPerThread) * (numImgColors/(threadY*colorsPerThread));
      blockY =  imgPixels;
  }
  else if (numFilterColors > 3)
  {
      colorsPerThread = numFilterColors % 4 == 0 ? 4 : 2;
      blockX = DIVUP(numImages,threadX * imgsPerThread) * (numImgColors / colorsPerThread);
      blockY = DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4);
  }
  else
  {
      blockX = DIVUP(numImages,threadX * imgsPerThread);
      blockY = DIVUP(imgSizeY,4) * DIVUP(imgSizeX,4);
  }
  bool checkCaseBounds = numImages % (threadX * imgsPerThread) != 0;
    
  if (conv) { // convolutional units
      if (scaleTargets == 0) { // do not scale or use targets matrix
          if (numFilterColors % 8 == 0) {
              if (imgsPerThread == 4) {
                  if (checkCaseBounds) {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 2)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, false, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, false, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, false, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, false, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, false, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, false, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, false, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, false, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else if (numFilterColors > 3)
          {
              if (imgsPerThread == 8)
              {
                  if (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else
          {
              if (imgsPerThread == 8)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
              else if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, false, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, false, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, false, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, false, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
          }
      }
      else
      { // do scale
          if (numFilterColors % 8 == 0)
          {
              if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 2)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, true, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, true, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, true, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, true, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, true, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, true, true, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, true, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, true, false, true>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else if (numFilterColors > 3)
          {
              if (imgsPerThread == 8)
              {
                  if  (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 4)
              {
                  if  (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if  (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else
          {
              if (imgsPerThread == 8)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      } 
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              } 
              else if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, true, true, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, true, true, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, true, false, true>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, true, false, true> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
          }
      }
  }
  else
  { // local, unshared units
      if (scaleTargets == 0)
      { // do not scale or use targets matrix
          if (numFilterColors % 8 == 0)
          {
              if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 2)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, false, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, false, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, false, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, false, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, false, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, false, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, false, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, false, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else if (numFilterColors > 3)
          {
              if (imgsPerThread == 8)
              {
                  if (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else
          {
              if (imgsPerThread == 8)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
              else if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, false, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, false, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, false, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, false, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
          }
      }
      else
      { // do scale
          if (numFilterColors % 8 == 0)
          {
              if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 4, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 4, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 2)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, true, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, true, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 4, true, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 4, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 2, 2, true, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 2, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, true, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, true, true, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (numFilterColors % 16 == 0)
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 4, true, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 4, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 1, 2, true, false, false>, gpuFuncCachePreferShared);
                          conv_img_acts_manycolor<4, 32, 1, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else if (numFilterColors > 3)
          {
              if (imgsPerThread == 8)
              {
                  if  (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 4, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<8, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else if (imgsPerThread == 4)
              {
                  if  (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 4, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 4, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<4, 2, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<4, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
              else
              {
                  if  (checkCaseBounds)
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
                  else
                  {
                      if (colorsPerThread == 4)
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 4, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 4, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                      else
                      {
                          //gpuFuncSetCacheConfig(img_acts_mediumcolor<2, 2, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_mediumcolor<2, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, blockX, blockY);
                      }
                  }
              }
          }
          else
          {
              if (imgsPerThread == 8)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 1, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 1, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 2, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<8, 3, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<8, 3, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
              else if (imgsPerThread == 4)
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 1, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 1, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 2, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<4, 3, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<4, 3, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
              else
              {
                  if (checkCaseBounds)
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, true, true, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, true, true, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
                  else
                  {
                      if (numFilterColors == 1)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 1, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 1, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 2)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 2, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 2, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                      else if (numFilterColors == 3)
                      {
                          //gpuFuncSetCacheConfig(img_acts_color<2, 3, true, false, false>, gpuFuncCachePreferShared);
                          img_acts_color<2, 3, true, false, false> (hidActs, filters, targets,
                                                              numModulesY, numModulesX, numImages, numFilters, filterSize, imgSizeY, imgSizeX, paddingStart, moduleStride, scaleTargets, scaleOutput, blockX, blockY, numFilterColors);
                      }
                  }
              }
          }
      }
  }
}


