#ifndef TH_GPU_TENSOR_CONV_INC
#define TH_GPU_TENSOR_CONV_INC

#include "THCTensor.h"

THC_API void THGPUTensor_conv2Dmv(THGPUTensor *output, float beta, THGPUTensor *input,
                                  THGPUTensor *kernel, long srow, long scol, const char *type);

THC_API void THGPUTensor_conv2Dmm(THGPUTensor *output, float beta, THGPUTensor *input,
                                  THGPUTensor *kernel, long srow, long scol, const char *type);

THC_API void THGPUTensor_conv2DRevger(THGPUTensor *output, float beta, float alpha,
                                      THGPUTensor *input, THGPUTensor *kernel,
                                      long srow, long scol);

THC_API void THGPUTensor_conv2DRevgerm(THGPUTensor *output, float beta, float alpha,
                                       THGPUTensor *input, THGPUTensor *kernel,
                                       long srow, long scol);

THC_API void THGPUTensor_conv2Dmap(THGPUTensor *output, THGPUTensor *input,
                                   THGPUTensor *kernel, long stride_x, long stride_y,
                                   THGPUTensor *table, long fanin);

#endif
