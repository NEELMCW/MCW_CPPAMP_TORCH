/* 
 * This code has been adapted from Alex Krizhevsky's GPU library
 * (http://code.google.com/p/gpu-convnet/), and was originally
 * licensed under a BSD license.
 */

#ifndef SPATIAL_POOL_BPROP_CU
#define	SPATIAL_POOL_BPROP_CU

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * maxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
void kLocalMaxUndo(Concurrency::array_view<float,1> &avImages,
                              Concurrency::array_view<float,1> &avMaxGrads, Concurrency::array_view<float,1> &avMaxActs,
                              Concurrency::array_view<float,1> &avTargets, int imgSize, int numFilters,
                              int numImages, int subsX, int startX, int strideX, int outputsX,
                              float scaleTargets, float scaleOutputs, int blockX, int blockY) 
{
  Concurrency::extent<3> grdExt(1, blockY * 4, blockX * 32);
  Concurrency::tiled_extent<1, 4, 32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 4, 32> tidx) restrict(amp) 
  {
    tile_static float shImgs[B_Y*filtersPerThread][B_X*imgsPerThread];
    float* imgs = avImages.data();
    float* maxGrads = avMaxGrads.data();
    float* maxActs = avMaxActs.data();
    float* target = avTargets.data();

    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = tidx.tile[2] / numImgBlocks;
    const int blockPxY = tidx.tile[1] / (numFilters/(B_Y*filtersPerThread));

    const int blockImgIdx = (tidx.tile[2] % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (tidx.tile[1] % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;

    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);

    const int imgIdx = blockImgIdx + tidx.local[2];

    imgs += ((blockFilterIdx + tidx.local[1]) * imgPixels + blockPx) * numImages + imgIdx;
    maxGrads += ((blockFilterIdx + tidx.local[1]) * numOutputs) * numImages 
                 + imgIdx;
    maxActs += ((blockFilterIdx + tidx.local[1]) * numOutputs) * numImages 
                + imgIdx;

    target += ((blockFilterIdx + tidx.local[1]) * imgPixels + blockPx) * numImages + imgIdx;

    float prod[filtersPerThread][imgsPerThread];
    for (int f = 0; f < filtersPerThread; f++) {
      for (int i = 0; i < imgsPerThread; i++) {
        prod[f][i] = 0;
      }
    }

    if (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX 
        && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          for (int f = 0; f < filtersPerThread; f++) {
            shImgs[tidx.local[1] + B_Y * f][tidx.local[2] + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
          }
        }
      }
      for (int my = startOutputY; my < endOutputY; my++) {
        for (int mx = startOutputX; mx < endOutputX; mx++) {
          const int outputIdx = my * outputsX + mx;
          for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
              for (int f = 0; f < filtersPerThread; f++) {
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
    if (!add) {
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          for (int f = 0; f < filtersPerThread; f++) {
            target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
          }
        }
      }
    } else {
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          for (int f = 0; f < filtersPerThread; f++) {
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
  Concurrency::array_view<float,1>&images, Concurrency::array_view<float,1>&maxgrads,
  Concurrency::array_view<float,1>&maxacts, Concurrency::array_view<float,1>&targets,
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
  assert(imgSize * imgSize == imgPixels); /// TODO SQUARE !

  int subsX = filterSizeX;
  assert(filterSizeX == filterSizeY);
  
  int startX = paddingStart;
  int strideX = moduleStride;

  int outputsX = numModulesX;

  assert(numModulesY == numModulesX);  /// TODO SQUARE !

  assert(numFilters % 16 == 0);
  assert(strideX <= subsX);

  int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
  int checkCaseBounds = numImages % (32*imgsPerThread) != 0;

  int blockX, blockY;
  blockX = DIVUP(numImages,32*imgsPerThread) * imgSize;
  blockY = (numFilters / (4 * 2)) * imgSize;

  if (imgsPerThread == 4) {
    if (checkCaseBounds) {
      if (scaleTargets == 0 && scaleOutput == 1) {
    kLocalMaxUndo<4, 32, 4, 2, false, true>(images, maxgrads, maxacts, targets,
                                     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                     blockX, blockY);
      } else {
    kLocalMaxUndo<4, 32, 4, 2, true, true>(images, maxgrads, maxacts, targets,
                                    imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                    blockX, blockY);
      }
    } else {
      if (scaleTargets == 0 && scaleOutput == 1) {
    kLocalMaxUndo<4, 32, 4, 2, false, false>(images, maxgrads, maxacts, targets,
                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                      blockX, blockY);
      } else {
    kLocalMaxUndo<4, 32, 4, 2, true, false>(images, maxgrads, maxacts, targets,
                                     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                     blockX, blockY);
      }
    }
  } else if (imgsPerThread == 2) {
    if (checkCaseBounds) {
      if (scaleTargets == 0 && scaleOutput == 1) {
    kLocalMaxUndo<4, 32, 2, 2, false, true>(images, maxgrads, maxacts, targets,
                                     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                     blockX, blockY);
      } else {
    kLocalMaxUndo<4, 32, 2, 2, true, true>(images, maxgrads, maxacts, targets,
                                    imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                    blockX, blockY);
      }
    } else {
      if (scaleTargets == 0 && scaleOutput == 1) {
    kLocalMaxUndo<4, 32, 2, 2, false, false>(images, maxgrads, maxacts, targets,
                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                      blockX, blockY);
      } else {
    kLocalMaxUndo<4, 32, 2, 2, true, false>(images, maxgrads, maxacts, targets,
                                     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                     blockX, blockY);
      }
    }
  } else {
    if (checkCaseBounds) {
      if (scaleTargets == 0 && scaleOutput == 1) {
    kLocalMaxUndo<4, 32, 1, 2, false, true>(images, maxgrads, maxacts, targets,
                                     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                     blockX, blockY);
      } else {
    kLocalMaxUndo<4, 32, 1, 2, true, true>(images, maxgrads, maxacts, targets,
                                    imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                    blockX, blockY);
      }
    } else {
      if (scaleTargets == 0 && scaleOutput == 1) {
    kLocalMaxUndo<4, 32, 1, 2, false, false>(images, maxgrads, maxacts, targets,
                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                      blockX, blockY);
      } else {
    kLocalMaxUndo<4, 32, 1, 2, true, false>(images, maxgrads, maxacts, targets,
                                     imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput,
                                     blockX, blockY);
      }
    }
  }
}

#endif	/* SPATIAL_POOL_BPROP_CU */
