/* 
 * This code has been adapted from Alex Krizhevsky's GPU library
 * (http://code.google.com/p/cuda-convnet/), and was originally
 * licensed under a BSD license.
 */

#define LO16(x)     ((x) & 0x0000FFFF)
#define HI16(x)     ((x) >> 16)

/*
 * Each block computes weight gradients for B_Y * pixelsPerThread pixels and B_X filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X, module batch of partialSum
 * blockIdx.y determines pixel batch of B_Y * pixelsPerThread
 *
 * Number of filters must be divisible by B_X
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * numModules must be divisible by partialSum
 *
 * After adding pixelsPerThread, register usage went from 20 to 23 (when pixelsPerThread = 1)...
 * so the compiler is messing up here somehow. It's unable to optimize that case away.
 */
template <int B_Y, int B_X, int pixelsPerThread, int preloadCases, int numColors, bool scale, bool checkCaseBounds>
void conv_weight_acts_c(float* images, float* hidActs, float* targets,
                                   const int numImages, const int numFilters,
                                   const int numModulesY, const int numModulesX,
                                   const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride, const int imgStride,
                                   const int partialSum,
                                   const float scaleTargets, const float scaleOutputs) {
/*    __shared__ float shImages[pixelsPerThread * B_Y * numColors][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidActs

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int filterBlocksPerModule = numFilters / B_X;
    const int outputModuleIdx = blockIdx.x / filterBlocksPerModule;
    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = B_X * (blockIdx.x % filterBlocksPerModule);

//    const int moduleStride = (imgSize - filterSize + 1) / numModulesX; 
    const int numModules = numModulesY * numModulesX;

    const int blockPixelOffset = blockIdx.y * B_Y * pixelsPerThread;

    images += loadX;
    hidActs += moduleIdx * numImages
            + blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
    
    targets += (outputModuleIdx * numFilters) * filterPixels * numColors
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    float* shImgLoad = &shImages[loadY][loadX];
    float* shHidActLoad = &shHidActs[loadY][loadX];

    float prod[numColors][pixelsPerThread];
    #pragma unroll
    for (int c = 0; c < numColors; c++) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            prod[c][p] = 0;
        }
    }
    
    __shared__ int pxDivs[B_Y*pixelsPerThread];
    if (tidx < B_Y * pixelsPerThread) {
        pxDivs[tidx] = (((blockPixelOffset + tidx) / filterSize) << 16) + ((blockPixelOffset + tidx) % filterSize);
    }
    __syncthreads();
    for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
        const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
            if (loadY < B_Y * pixelsPerThread) {
    //            #pragma unroll
                for (int y = 0; y < B_Y * pixelsPerThread; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_Y * pixelsPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y * pixelsPerThread) {
                        const int pxIdx = loadY + y; // pixel idx in filter

                        if (pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                            const int pxY = imgLoadModPosY + HI16(pxDivs[pxIdx]); // pixel x,y coords in image
                            const int pxX = imgLoadModPosX + LO16(pxDivs[pxIdx]);
                            if (pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX) {
                                const int pixIdx = (pxY * imgSizeX + pxX) * imgStride;
                                #pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < numColors; c++) {
                                    shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < numColors; c++) {
                                shImgLoad[(y + c * pixelsPerThread * B_Y) * preloadCases] = 0;
                            }
                        }
                    }
                }
            }
            if (loadY < B_X && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                #pragma unroll
                for (int y = 0; y < B_X; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if (B_X % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X) {
                        shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules];
                    }
                }
            }

            __syncthreads();
            #pragma unroll
            for (int p = 0; p < pixelsPerThread; p++) {
                #pragma unroll
                for (int i = 0; i < preloadCases; i++) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        prod[c][p] += shImages[threadIdx.y + p * B_Y + c * pixelsPerThread * B_Y][i] * shHidActs[threadIdx.x][i];
                    }
                }
            }
            __syncthreads();
        }
        hidActs += numImages;
    }
    
    if (scale) {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    targets[p * B_Y * numFilters + c * filterPixels * numFilters] = scaleTargets * targets[p * B_Y * numFilters + c * filterPixels * numFilters] + scaleOutputs * prod[c][p];
                }
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < pixelsPerThread; p++) {
            if (blockPixelOffset + p * B_Y + threadIdx.y < filterPixels) {
                #pragma unroll
                for (int c = 0; c < numColors; c++) {
                    targets[p * B_Y * numFilters + c * filterPixels * numFilters] = scaleOutputs * prod[c][p];
                }
            }
        }
    }
*/
}

/*
 * Each block computes weight gradients for B_Y pixels and B_X * filtersPerThread filters
 * threadIdx.x determines filter
 * threadIdx.y determines pixel in filter
 *
 * blockIdx.x determines filter batch of B_X * filtersPerThread, module batch of partialSum
 * blockIdx.y determines pixel, color batch of B_Y * colorsPerThread
 *      In essence, blockIdx.y.x = 0...numFilterColors / colorsPerThread
 *                  blockIdx.y.y = 0...DIVUP(numPixels, B_Y)
 * ============
 * CONSTRAINTS:
 * ============
 * numFilters/numGroups must be divisible by B_X * filtersPerThread
 * numImgColors/numGroups must be divisible by colorsPerThread
 * numFilters must be divisible by numGroups
 * numImgColors must be divisible by numGroups
 * Number of images (cases) should be divisible by preloadCases if checkCaseBounds is false.
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * targets:     (numModulesY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)
 *
 * B_Y * B_X should be divisible by preloadCases.
 * preloadCases one of 16, 32.
 * B_X one of 4, 8, 16, 32
 * B_Y arbitrary (satisfying divisibility constraints)
 * 
 * This routine is especially fast when numFilters >= 32. That's when it should be used.
 */
template <int B_Y, int B_X, int filtersPerThread, int colorsPerThread, int preloadCases, bool scale, bool checkCaseBounds>
void conv_weight_acts_mc_mf(float* images, float* hidActs, float* targets,
                                       const int numImages, const int numFilters,
                                       const int numModulesY, const int numModulesX,
                                       const int imgSizeY, const int imgSizeX, const int filterSize,
                                       const int paddingStart, const int moduleStride, const int imgStride,
                                       const int numImgColors, const int numGroups, const int partialSum,
                                       const float scaleTargets, const float scaleOutputs) {
/*    __shared__ float shImages[colorsPerThread * B_Y][preloadCases]; // preload preloadCases cases of B_Y * pixelsPerThread pixels
    __shared__ float shHidActs[filtersPerThread * B_X][preloadCases + 1]; // preload preloadCases cases of B_X hidacts

    const int tidx = B_X * threadIdx.y + threadIdx.x;
    const int loadY = tidx / preloadCases, loadX = tidx % preloadCases;

    const int filterPixels = filterSize * filterSize;
    const int imgPixels = imgSizeY * imgSizeX;

    const int numFilterBlocks = numFilters / (B_X * filtersPerThread);
    const int outputModuleIdx = blockIdx.x / numFilterBlocks;
    const int moduleIdx = partialSum * outputModuleIdx;
    const int blockFilterIdx = filtersPerThread * B_X * (blockIdx.x % numFilterBlocks);
    const int numModules = numModulesY * numModulesX;
    
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;
    const int numFilterColors = numImgColors / numGroups;
    
    const int blockPixelOffset = (blockIdx.y / (numFilterColors/colorsPerThread)) * B_Y;
    const int filterColorIdx = (blockIdx.y % (numFilterColors/colorsPerThread)) * colorsPerThread;
    const int imgColorIdx = filterColorIdx + blockGroupIdx * numFilterColors;

    images += imgColorIdx * imgPixels * imgStride + loadX;

    hidActs += moduleIdx * numImages
            + blockFilterIdx * numImages * numModules
            + loadY * numImages * numModules
            + loadX;
    
    targets += outputModuleIdx * numFilters * filterPixels * numFilterColors
            + filterColorIdx * filterPixels * numFilters
            + blockPixelOffset * numFilters
            + blockFilterIdx
            + threadIdx.y * numFilters + threadIdx.x;

    float* shHidActLoad = &shHidActs[loadY][loadX];
    float* shImgLoad = &shImages[loadY][loadX];
    float prod[colorsPerThread][filtersPerThread];
    #pragma unroll
    for (int c = 0; c < colorsPerThread; c++) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
            prod[c][f] = 0;
        }
    }
    // This avoids doing a division in an inner loop
    __shared__ int pxDivs[B_Y];
    if (tidx < B_Y) {
        pxDivs[tidx] = (((blockPixelOffset + tidx) / filterSize) << 16) + (blockPixelOffset + tidx) % filterSize;
    }
    __syncthreads();
    for (int m = moduleIdx; m < moduleIdx + partialSum; m++) {
        const int imgLoadModPosY = paddingStart + (m / numModulesX) * moduleStride;
        const int imgLoadModPosX = paddingStart + (m % numModulesX) * moduleStride;
        for (int caseIdx = 0; caseIdx < numImages; caseIdx += preloadCases) {
            if (loadY < B_Y) {
    //            #pragma unroll
                for (int y = 0; y < B_Y; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if (B_Y % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_Y) {
                        const int pxIdx = loadY + y; // pixel idx in filter

                        if (pxIdx + blockPixelOffset < filterPixels && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                            const int pxY = imgLoadModPosY + HI16(pxDivs[pxIdx]);//pxIdx / filterSize; // pixel x,y coords in image
                            const int pxX = imgLoadModPosX + LO16(pxDivs[pxIdx]);
                            if (pxY >= 0 && pxY < imgSizeY && pxX >= 0 && pxX < imgSizeX) {
                                const int pixIdx = (pxY * imgSizeX + pxX) * imgStride; // pixel idx in image
                                #pragma unroll
                                for (int c = 0; c < colorsPerThread; c++) {
                                    shImgLoad[(y + c * B_Y) * preloadCases] = images[caseIdx + c * imgPixels * imgStride + pixIdx];
                                }
                            } else {
                                #pragma unroll
                                for (int c = 0; c < colorsPerThread; c++) {
                                    shImgLoad[(y + c * B_Y) * preloadCases] = 0;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorsPerThread; c++) {
                                shImgLoad[(y + c * B_Y) * preloadCases] = 0;
                            }
                        }
                    }
                }
            }
            if (loadY < B_X * filtersPerThread && (!checkCaseBounds || caseIdx + loadX < numImages)) {
                #pragma unroll
                for (int y = 0; y < B_X * filtersPerThread; y += (B_X * B_Y) / preloadCases) {
                    // Make sure number of rows in the array is divisible by number of rows filled per iteration
                    if ((B_X * filtersPerThread) % (B_X * B_Y / preloadCases) == 0 || y + loadY < B_X * filtersPerThread) {
                        shHidActLoad[y * (preloadCases + 1)] = hidActs[caseIdx + y * numImages * numModules];
                    }
                }
            }

            __syncthreads();

            #pragma unroll
            for (int c = 0; c < colorsPerThread; c++) {
                #pragma unroll
                for (int i = 0; i < preloadCases; i++) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[c][f] += shImages[threadIdx.y + c * B_Y][i] * shHidActs[threadIdx.x + f * B_X][i];
                    }
                }
            }
            __syncthreads();
        }
        hidActs += numImages;
    }
    if (blockPixelOffset + threadIdx.y < filterPixels) {
        if (scale) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * filterPixels * numFilters + f * B_X] = scaleTargets * targets[c * filterPixels * numFilters + f * B_X] + scaleOutputs * prod[c][f];
                }
            }
        } else {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for (int c = 0; c < colorsPerThread; c++) {
                    targets[c * filterPixels * numFilters + f * B_X] = scaleOutputs * prod[c][f];
                }
            }
        }
    }
*/
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages), with stride given
 * hidActs:     (numFilters, numModules, numImages)
 *
 * targets:     (numModuleY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters)
 * 
 * TODO: you can get a slight speed boost for local non-convolutional units by writing special
 * routines for partialSum = 1. But I dunno if the code duplication is worth it...
 * 
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128. 
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast. 
 */
void spatialConv_accGradParameters(
    // raw pointers:
    float *images, float *hidActs, float *targets,
    // input dim:
    int numImgColors, int imgSizeY, int imgSizeX, int numImages,
    // output dim:
    int numFilters, int numModulesY, int numModulesX, 
    // filter size:
    int filterSizeY, int filterSizeX,
    // input params:
    int paddingStart, int moduleStride,
    // output params:
    float scaleTargets, float scaleOutput,
    int partialSum
)
{
    int numGroups = 1;
    int imgStride = numImages;
    int numFilterColors = numImgColors / numGroups;
    int numModules = numModulesY * numModulesX;
    int numFiltersPerGroup = numFilters / numGroups;
    int filterSize = filterSizeX;
    int filterPixels = filterSize * filterSize;
    
    assert(filterSizeX == filterSizeY);
    assert(imgSizeY == imgSizeX);
    assert(numImgColors % numGroups == 0);
    assert(numFilters % (16*numGroups) == 0);
    assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 4 == 0)));
    assert(numGroups == 1 || numFilterColors % 4 == 0);

    partialSum = partialSum == 0 ? numModules : partialSum;

    assert(numModules % partialSum == 0);

    assert(paddingStart <= 0);
    assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    assert(moduleStride <= filterSize);
    

    //dim3 blocks, threads;
    int bx, by;
    int pixelsPerThread, filtersPerThread, colorsPerThread;
    // Worth playing with these parameters to find best values for your problem.
    // These values work relatively well, but not optimal for all problems.
    if (numFilterColors > 3) {
        filtersPerThread = numFiltersPerGroup % 32 == 0 ? 2 : 1;
        colorsPerThread = numFilterColors % 8 == 0 ? 8 : 4;
        by = numFiltersPerGroup % 64 == 0 ? 4 : 8;
        bx = numFiltersPerGroup % 64 == 0 ? 32 : 16;
        //blocks = dim3((numModules/partialSum)*(numFilters/(bx*filtersPerThread)), DIVUP(filterPixels, by) * (numFilterColors / colorsPerThread));
    } else {
        assert(numGroups == 1); // Just for sanity
        pixelsPerThread = numFilters % 32 == 0 ? (numImgColors == 1 ? 8 : 5) : (numImgColors == 1 ? 5 : 2);
        by = numFilters % 32 == 0 ? 4 : 8; // by == 4 seems to work best
        bx = numFilters % 32 == 0 ? 32 : 16; 
       // blocks = dim3((numModules/partialSum)*(numFilters/bx), DIVUP(filterPixels, by*pixelsPerThread));
    }
    assert((by * bx) % 32 == 0);
    //threads = dim3(bx, by);
    bool checkCaseBounds = numImages % 32 != 0;
    
    if (numFilterColors > 3) {
        if (scaleTargets == 0) { // do not scale
            if (numFiltersPerGroup % 64 == 0) {
                if (numFilterColors % 8 == 0) {
                    if (checkCaseBounds) {
                       // cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<4,32,2,8,32,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                         //                                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                       // cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<4,32,2,8,32,false, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                          //                                                             paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (checkCaseBounds) {
                       // cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<4,32,2,4,32,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                         //                                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                       // cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<4,32,2,4,32,false, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                          //                                                             paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numFiltersPerGroup % 32 == 0) {
                if (numFilterColors % 8 == 0) {
                    if (checkCaseBounds) {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,2,8,32,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,2,8,32,false, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (checkCaseBounds) {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,2,4,32,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                       // cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,2,4,32,false, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else {
                if (numFilterColors % 8 == 0) {
                    if (checkCaseBounds) {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,1,8,32,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,1,8,32,false, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (checkCaseBounds) {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,1,4,32,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                             //                                                          paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,1,4,32,false, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                }
            }
        } else {

            if (numFiltersPerGroup % 64 == 0) {
                if (numFilterColors % 8 == 0) {
                    if (checkCaseBounds) {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<4,32,2,8,32,true, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<4,32,2,8,32,true, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (checkCaseBounds) {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, true>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_mc_mf<4,32,2,4,32,true, true><<<blocks, threads>>>(images, hidActs, targets,
                        //                                                               numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                          //                                                             paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<4,32,2,4,32,true, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numFiltersPerGroup % 32 == 0) {
                if (numFilterColors % 8 == 0) {
                    if (checkCaseBounds) {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, true>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_mc_mf<8,16,2,8,32,true, true><<<blocks, threads>>>(images, hidActs, targets,
                        //                                                               numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                          //                                                             paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,2,8,32,true, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (checkCaseBounds) {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,2,4,32,true, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,2,4,32,true, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else {
                if (numFilterColors % 8 == 0) {
                    if (checkCaseBounds) {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,1,8,32,true, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_mc_mf<8,16,1,8,32,true, false><<<blocks, threads>>>(images, hidActs, targets,
                         //                                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                           //                                                            paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (checkCaseBounds) {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, true>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_mc_mf<8,16,1,4,32,true, true><<<blocks, threads>>>(images, hidActs, targets,
                        //                                                               numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                          //                                                             paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    } else {
                 //       cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, false>, cudaFuncCachePreferShared);
                   //     conv_weight_acts_mc_mf<8,16,1,4,32,true, false><<<blocks, threads>>>(images, hidActs, targets,
                     //                                                                  numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize,
                       //                                                                paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
                    }
                }
            }
        }
    } else { // numColors in 1,2,3
        if (scaleTargets == 0) { // do not scale
            if (numFilterColors == 1) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
              //          cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, false, true>, cudaFuncCachePreferShared);
                //        conv_weight_acts_c<4,32,8,32,1,false, true><<<blocks, threads>>>(images, hidActs, targets,
                  //                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, false, true>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<8,16,5,32,1,false, true><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, false, false>, cudaFuncCachePreferShared);
                        //conv_weight_acts_c<4,32,8,32,1,false, false><<<blocks, threads>>>(images, hidActs, targets,
                          //                                      numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, false, false>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<8,16,5,32,1,false, false><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numFilterColors == 2) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                 //       cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, false, true>, cudaFuncCachePreferShared);
                   //     conv_weight_acts_c<4,32,5,32,2,false, true><<<blocks, threads>>>(images, hidActs, targets,
                     //                                           numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_c<8,16,2,32,2,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                       numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, false, false>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<4,32,5,32,2,false, false><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, false, false>, cudaFuncCachePreferShared);
                       // conv_weight_acts_c<8,16,2,32,2,false, false><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numFilterColors == 3) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, false, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_c<4,32,5,32,3,false, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                       numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, false, true>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<8,16,2,32,3,false, true><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                   //     cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, false, false>, cudaFuncCachePreferShared);
                     //   conv_weight_acts_c<4,32,5,32,3,false, false><<<blocks, threads>>>(images, hidActs, targets,
                       //                                         numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, false, false>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<8,16,2,32,3,false, false><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                }
            }

        } else { // do scale
            if (numFilterColors == 1) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                //        cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, true, true>, cudaFuncCachePreferShared);
                  //      conv_weight_acts_c<4,32,8,32,1,true, true><<<blocks, threads>>>(images, hidActs, targets,
                    //                                            numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                     //   cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, true, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_c<8,16,5,32,1,true, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                       numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, true, false>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<4,32,8,32,1,true, false><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, true, false>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<8,16,5,32,1,true, false><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numFilterColors == 2) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
              //          cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, true, true>, cudaFuncCachePreferShared);
                //        conv_weight_acts_c<4,32,5,32,2,true, true><<<blocks, threads>>>(images, hidActs, targets,
                  //                                              numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                   //     cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, true, true>, cudaFuncCachePreferShared);
                    //    conv_weight_acts_c<8,16,2,32,2,true, true><<<blocks, threads>>>(images, hidActs, targets,
                      //                                          numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                    //    cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, true, false>, cudaFuncCachePreferShared);
                      //  conv_weight_acts_c<4,32,5,32,2,true, false><<<blocks, threads>>>(images, hidActs, targets,
                        //                                        numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, true, false>, cudaFuncCachePreferShared);
                        //conv_weight_acts_c<8,16,2,32,2,true, false><<<blocks, threads>>>(images, hidActs, targets,
                          //                                      numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                }
            } else if (numFilterColors == 3) {
                if (checkCaseBounds) {
                    if (numFilters % 32 == 0) {
                   //     cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, true, true>, cudaFuncCachePreferShared);
                     //   conv_weight_acts_c<4,32,5,32,3,true, true><<<blocks, threads>>>(images, hidActs, targets,
                       //                                         numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, true, true>, cudaFuncCachePreferShared);
                       // conv_weight_acts_c<8,16,2,32,3,true, true><<<blocks, threads>>>(images, hidActs, targets,
                         //                                       numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                } else {
                    if (numFilters % 32 == 0) {
                 //       cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, true, false>, cudaFuncCachePreferShared);
                   //     conv_weight_acts_c<4,32,5,32,3,true, false><<<blocks, threads>>>(images, hidActs, targets,
                     //                                           numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    } else {
                      //  cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, true, false>, cudaFuncCachePreferShared);
                        //conv_weight_acts_c<8,16,2,32,3,true, false><<<blocks, threads>>>(images, hidActs, targets,
                          //                                      numImages, numFilters, numModulesY, numModulesX, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
                    }
                }
            }
        }
    }
    
    // check for errors
}
