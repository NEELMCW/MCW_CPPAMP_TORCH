#include "THCBlas.h"

#define GPU_NUM_THREADS 256

// GPU: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
}

// WARNING: this module is incomplete - and just meant for reference for now.

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)

void imt2col_kernel(int n, Concurrency::array_view<float> & avData_im, long imOffset,
                    int inOffset, int height, int width, int ksize_h, int ksize_w, int pad_h,
                    int pad_w, int stride_h, int stride_w, int height_col, int width_col,
                    Concurrency::array_view<float,1> &avData_col, long colOffset)
{
  avData_im.discard_data();
  unsigned grdSz = (n + 255) & ~255;
  Concurrency::extent<1> grdExt(grdSz);
  Concurrency::tiled_extent<256> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp)
  {
    float dataCol = 0;
    float dataIm = inOffset;
    for (int i = tidx.global[0]; i < (n); i += t_ext[0])
    {
      int w_out = i % width_col;
      i /= width_col;
      int h_out = i % height_col;
      int channel_in = i / height_col;
      int channel_out = channel_in * ksize_h * ksize_w;
      int h_in = h_out * stride_h - pad_h;
      int w_in = w_out * stride_w - pad_w;
      dataCol += (channel_out * height_col + h_out) * width_col + w_out;
      dataIm += (channel_in * height + h_in) * width + w_in;
      for (int p = 0; p < ksize_h; ++p)
      {
        for (int j = 0; j < ksize_w; ++j)
        {
          int h = h_in + p;
          int w = w_in + j;
          avData_col[colOffset + dataCol] = (h >= 0 && w >= 0 && h < height && w < width) ? avData_im[imOffset + dataIm + p * width + j] : 0;
          dataCol += height_col * width_col;
        }
      }
    }
  });
}

void imt2col(Concurrency::array_view<float> &data_im, long imOffset,
             int inOffset, int channels, int height, int width,int ksize_h,
             int ksize_w, int pad_h, int pad_w, int stride_h, int stride_w,
             Concurrency::array_view<float,1> &data_col, long colOffset)
{
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  // Launch
  imt2col_kernel(num_kernels, data_im, imOffset, inOffset, height, width,
                 ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
                 height_col, width_col, data_col, colOffset);
}

static int gpunn_SpatialConvolutionMM_BHWD_updateOutput(lua_State *L)
{
  // Input
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THGPUTensor *weight = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *bias = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.GPUTensor");
  THGPUTensor *columns = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.GPUTensor");
  THGPUTensor *ones = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;


  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THGPUTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THGPUTensor_resize2d(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth)
  {
    // Resize plane and fill with ones...
    THGPUTensor_resize2d(ones, outputHeight, outputWidth);
    THGPUTensor_fill(ones, 1);
  }

  // Helpers
  PREPARE_AV(columns, avData_col);
  PREPARE_AV(input, avData_im);

  long m_ = nOutputPlane;
  long n_ = outputHeight * outputWidth;
  long k_ = 1;
  long m = weight->size[0];
  long n = columns->size[1];
  long k = weight->size[1];

  PREPARE_AV(ones, avData_ones);
  PREPARE_AV(bias, avData_bias);
  PREPARE_AV(output, avData_output);
  PREPARE_AV(weight, avData_weight);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++)
  {
    // Matrix mulitply per output:
    avData_ones->discard_data();
    avData_bias->discard_data();
    avData_output->discard_data();

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THGPUBlas_gemm_opt('t', 'n', n_, m_, k_, 1,
                       *avData_ones, ones->storageOffset, k_,
                       *avData_bias, bias->storageOffset, k_, 0,
                       *avData_output, output->storageOffset + output->stride[0] * elt, n_);

    avData_im->discard_data();
    avData_col->discard_data();

    // Extract columns:
    imt2col(*avData_im, input->storageOffset, input->stride[0] * elt,
            nInputPlane, inputHeight, inputWidth, kH, kW, padding,
            padding, dH, dW, *avData_col, columns->storageOffset);

    avData_col->discard_data();
    avData_weight->discard_data();
    avData_output->discard_data();

    // M,N,K are dims of matrix A and B
    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THGPUBlas_gemm_opt('n', 'n', n, m, k, 1,
                       *avData_col, columns->storageOffset, n,
                       *avData_weight, weight->storageOffset, k, 1,
                       *avData_output, output->storageOffset + output->stride[0] * elt, n);
  }
  return 1;
}

static int gpunn_SpatialConvolutionMM_BHWD_updateGradInput(lua_State *L)
{
    // implementation in progress
    return 1;
}

static int gpunn_SpatialConvolutionMM_BHWD_accGradParameters(lua_State *L)
{
    // implementation in progress
    return 0;
}

static const struct luaL_Reg gpunn_SpatialConvolutionMM_BHWD__ [] = {
    {"SpatialConvolutionMM_BHWD_updateOutput", gpunn_SpatialConvolutionMM_BHWD_updateOutput},
    {"SpatialConvolutionMM_BHWD_updateGradInput", gpunn_SpatialConvolutionMM_BHWD_updateGradInput},
    {"SpatialConvolutionMM_BHWD_accGradParameters", gpunn_SpatialConvolutionMM_BHWD_accGradParameters},
    {NULL, NULL}
};

static void gpunn_SpatialConvolutionMM_BHWD_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.GPUTensor");
    luaT_registeratname(L, gpunn_SpatialConvolutionMM_BHWD__, "nn");
    lua_pop(L,1);
}
