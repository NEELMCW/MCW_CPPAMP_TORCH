// WARNING: this module is incomplete - and just meant for reference for now.

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)

#include "../gputorch/lib/THC/THCBlas.h"
void imt2col_kernel(const int n, THGPUTensor* data_im,
                    const int height, const int width, const int ksize, const int pad,
                    const int stride, const int channels,
                    const int height_col, const int width_col,
                    const int bidx, const int batch,
                    THGPUTensor* data_col, unsigned int elt)
{
  Concurrency::array_view<float,1> avData_im(THGPUTensor_nElement(data_im), THGPUTensor_data(data_im));
  Concurrency::array_view<float,1> avData_col(THGPUTensor_nElement(data_col), THGPUTensor_data(data_col));
  //unsigned grdSz = (n + 255) & ~255;
  unsigned int grdSz = (n + 255)/256;
  Concurrency::extent<1> grdExt(grdSz * 256);
  Concurrency::tiled_extent<256> t_ext(grdExt);
  //std::cout<<"imt2col"<<std::endl;
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp)
  {
    int data_col=0;
    int data_im=0;
    data_im = elt * height * width * channels;
    for (int i = tidx.global[0]; i < (n); i += t_ext[0])
    {
      // float *data_col = avData_col.data();
      // float *data_im = avData_im.data();
      int w_out = i % width_col;
      i /= width_col;
      int h_out = i % height_col;
      int channel_in = i / height_col;
      int channel_out = channel_in * ksize * ksize;
      int h_in = h_out * stride - pad;
      int w_in = w_out * stride - pad;
      data_col += ((channel_out * batch + bidx) * height_col + h_out) * width_col + w_out;
      data_im += ((bidx * height + h_in) * width + w_in) * channels + channel_in;
      for (int p = 0; p < ksize; ++p)
      {
        for (int j = 0; j < ksize; ++j)
        {
          int h = h_in + p;
          int w = w_in + j;
          avData_col[data_col] = (h >= 0 && w >= 0 && h < height && w < width) ?
                                 avData_im[data_im + ((p * width + j) * channels)] : 0;
          data_col += batch * height_col * width_col;
        }
      }
    }
  });
  avData_col.synchronize();
}

void imt2col(THGPUTensor* data_im, unsigned int elt, const int channels,
        const int height, const int width, const int ksize, const int pad,
        const int stride, const int batch, THGPUTensor* data_col)
{
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  // Launch
  for (int bidx = 0; bidx < batch; bidx++) {
    imt2col_kernel(num_kernels, data_im, height, width, ksize,
                   pad, stride, channels,height_col, width_col,
                   bidx, batch, data_col, elt);
  }
}

static int gpunn_SpatialConvolutionMM_BHWD_updateOutput(lua_State *L) {
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
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int dimw = 1;
  int dimh = 0;
  if (input->nDimension == 4) {
    dimw++;
    dimh++;
  }
  long inputWidth   = input->size[dimw];
  long inputHeight  = input->size[dimh];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  luaL_argcheck(L, kW == kH, 1, "filters must be square (kW == kH)");
  luaL_argcheck(L, dW == dH, 1, "stride must be square (dW == dH)");

  if (input->nDimension == 3) {

      // implementation in progress...

  } else {
    // Batch size + input planes
    long batchSize = input->size[0];
    luaL_argcheck(L, batchSize == 1 || batchSize % 4 == 0, 1, "batch size should be a multiple of 4 or equal to 1");
    luaL_argcheck(L, nOutputPlane % 8 == 0, 1, "nOutputPlane should be a multiple of 8");

    // Step batch (inner loop)
    // This variable defines how many samples are processed in //, in the inner loop
    int stepBatchSize = 1;
    if (batchSize % 4 == 0) {
      stepBatchSize = 4;
    }

    // Resize output
    THGPUTensor_resize4d(output, batchSize, outputHeight, outputWidth, nOutputPlane);

    // Resize temporary columns
    THGPUTensor_resize2d(columns, kH*kW*nInputPlane, stepBatchSize*outputHeight*outputWidth);

    // Add bias first
    // TODO: replace this by more efficient, custom kernel
    long k;
    THGPUTensor *outputPlane = THGPUTensor_new();
    for(k=0; k<nOutputPlane; k++) {
      THGPUTensor_select(outputPlane, output, 3, k);
      THGPUTensor_fill(outputPlane, THGPUTensor_get1d(bias, k));
    }
    THGPUTensor_free(outputPlane);

    // Helper
    THGPUTensor *output_n = THGPUTensor_new();

    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt += stepBatchSize) {
        // Extract columns:
        // To Do by  Neelakandan
      imt2col(
          input, elt,
          nInputPlane, inputHeight, inputWidth, kW, padding, dW, stepBatchSize,
          columns
      );

      // Matrix mulitply per output:
      THGPUTensor_narrow(output_n, output, 0, elt, stepBatchSize);

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/gpu/cublas/#cublas-lt-t-gt-gemm)
      long m = weight->size[0];
      long n = columns->size[1];
      long k = weight->size[1];

      // Do GEMM_BHWD (note: this is a bit confusing because gemm assumes column-major matrices)
      THFloatBlas_gemm(
          't', 't',
          m, n, k,
          1,
          THGPUTensor_data(weight), k,
          THGPUTensor_data(columns), n,
          1,
          THGPUTensor_data(output_n), m
      );
    }

    // Free
    THGPUTensor_free(output_n);
  }

  // return output
  return 1;
}

static int gpunn_SpatialConvolutionMM_BHWD_updateGradInput(lua_State *L) {
  // implementation in progress
  return 1;
}

static int gpunn_SpatialConvolutionMM_BHWD_accGradParameters(lua_State *L) {
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
