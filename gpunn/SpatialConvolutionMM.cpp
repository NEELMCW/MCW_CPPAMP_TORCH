// GPU: grid stride l oping

#define GPU_KERNEL_LOOP(i, n) \
for (int i = tidx.tile_dim0 * tidx.tile[0] + tidx.local[0]; i < (n); i += t_ext[0])\

#include "amp_math.h"
#include "THBlas.h"
#include "../gputorch/lib/THC/THCBlas.h"
#include "THCGeneral.h"

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)

void im2col_kernel(const int n, Concurrency::array_view<float,1> &avData_im, const int inp_stride, const int elt, const int height, const int width, const int ksize_h,
                   const int ksize_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                   const int height_col, const int width_col, Concurrency::array_view<float,1> &avData_col)
{
  unsigned grdSz = (n + 255) & ~255;
  Concurrency::extent<1> grdExt(grdSz);
  Concurrency::tiled_extent<256> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp)
  {
    float dataCol = 0;
    float dataIm = 0;
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
          avData_col[dataCol] = (h >= 0 && w >= 0 && h < height && w < width) ? avData_im[ dataIm + p * width + j + elt * inp_stride] : 0;
          dataCol += height_col * width_col;
        }
      }
    }
  });
}

void im2col(Concurrency::array_view<float,1> &avData_im, const int inp_stride, const int elt, const int channels, const int height, const int width,
            const int ksize_h, const int ksize_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, Concurrency::array_view<float,1> &avData_col)
{
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  // Launch
  im2col_kernel(num_kernels, avData_im, inp_stride, elt, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, height_col, width_col, avData_col);
}

void col2im_kernel(const int n, Concurrency::array_view<float,1> &avData_col, const int height, const int width, const int channels,
                   const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h,
                   const int stride_w, const int height_col, const int width_col, Concurrency::array_view<float,1> &avData_im, const int inp_stride, const int elt)
{
  unsigned grdSz = (n + 255) & ~255;
  Concurrency::extent<1> grdExt(grdSz);
  Concurrency::tiled_extent<256> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp)
  {
    for (int i = tidx.global[0]; i < (n); i += t_ext[0])
    {
      float val = 0.0;
      int w = i % width + pad_w;
      int h = (i / width) % height + pad_h;
      int c = i / (width * height);
      // compute the start and end of the output
      int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
      int w_col_end = Concurrency::fast_math::fmin(w / stride_w + 1, width_col);
      int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
      int h_col_end = Concurrency::fast_math::fmin(h / stride_h + 1, height_col);
      // equivalent implementation
      int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
      int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
      int coeff_w_col = (1 - stride_w * height_col * width_col);
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) 
      {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) 
        {
          val += avData_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
        }
      }
      avData_im[i + elt * inp_stride] = val;
    }

  });
}

void col2im(Concurrency::array_view<float,1> &avData_col, const int channels, const int height, const int width,
            const int patch_h, const int patch_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, Concurrency::array_view<float,1> &avData_im, const int inp_stride, const int elt)
{
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
   col2im_kernel(num_kernels, avData_col, height, width, channels, patch_h, patch_w,
                 pad_h, pad_w, stride_h, stride_w, height_col, width_col, avData_im, inp_stride, elt);
}

static int gpunn_SpatialConvolutionMM_updateOutput(lua_State *L) {
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

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THGPUTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;


  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THGPUTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THGPUTensor_resize2d(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
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

  /* Prepare OpenCL memory objects and place matrices inside them. */
  void* bufA = THGPUBlas_clCreateBuffer(n_, k_ ,THGPUTensor_data(ones));
  void* bufB = THGPUBlas_clCreateBuffer(k_, m_ ,THGPUTensor_data(bias));
  void* bufC = THGPUBlas_clCreateBuffer(n_, m_*batchSize ,output->storage->data);
  //cl_mem* bufC_base = new cl_mem[batchSize];
  // FIXME: Need ensure the pre-allocation won't run out of memory based on 'batchSize'
  //for (int i = 0; i < batchSize; i++)
   // bufC_base[i] = static_cast<cl_mem>(THGPUBlas_clCreateBuffer(n_, m_ ,output->storage->data + output->stride[0] * i));



  void* buf_Column = THGPUBlas_clCreateBuffer(n, k ,THGPUTensor_data(columns));
  void* buf_Weight = THGPUBlas_clCreateBuffer(k, m ,THGPUTensor_data(weight));
  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/gpu/cublas/#cublas-lt-t-gt-gemm)

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THGPUBlas_gemm_opt(
        't', 'n',
        n_, m_, k_,
        1,
        THGPUTensor_data(ones), k_,
        THGPUTensor_data(bias), k_,
        0, output->stride[0] * elt,
        output->storage->data + output->stride[0] * elt, n_,
        bufA, bufB, bufC
    );
    // Extract columns:
    im2col(
        *avData_im, input->stride[0], elt,
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        *avData_col
    );
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/gpu/cublas/#cublas-lt-t-gt-gemm)

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THGPUBlas_gemm_opt(
        'n', 'n',
        n, m, k,
        1,
        THGPUTensor_data(columns), n,
        THGPUTensor_data(weight), k,
        1,output->stride[0] * elt,
        output->storage->data + output->stride[0] * elt, n,
        buf_Column, buf_Weight, bufC
    );
  }

  clReleaseMemObject(static_cast<cl_mem>(bufC));
  clReleaseMemObject(static_cast<cl_mem>(bufB));
  clReleaseMemObject(static_cast<cl_mem>(bufA));
  clReleaseMemObject(static_cast<cl_mem>(buf_Column));
  clReleaseMemObject(static_cast<cl_mem>(buf_Weight));

  // Resize output
  if (batch == 0) {
    THGPUTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
    THGPUTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
  }
  // return output
  return 1;
}

static int gpunn_SpatialConvolutionMM_updateGradInput(lua_State *L) {
  // Inputs
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THGPUTensor *weight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *gradColumns = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THGPUTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
    THGPUTensor_resize4d(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THGPUTensor_resize4d(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THGPUTensor_resize2d(gradColumns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  PREPARE_AV(gradColumns, avData_col);
  PREPARE_AV(gradInput, avData_im);
  long m = weight->size[1];
  long n = gradColumns->size[1];
  long k = weight->size[0];

  /* Prepare OpenCL memory objects and place matrices inside them. */
  void* buf_Column = THGPUBlas_clCreateBuffer(n, m ,THGPUTensor_data(gradColumns));
  void* buf_Weight = THGPUBlas_clCreateBuffer(k, m ,THGPUTensor_data(weight));

 // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/gpu/cublas/#cublas-lt-t-gt-gemm)

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THGPUBlas_gemm_opt(
        'n', 't',
        n, m, k,
        1,
        gradOutput->storage->data + gradOutput->stride[0] * elt, n,
        THGPUTensor_data(weight), m,
        0, 0,
        THGPUTensor_data(gradColumns), n,
        NULL, buf_Weight, buf_Column
    );

    // Unpack columns back into input:
    col2im(
        *avData_col,
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        *avData_im, gradInput->stride[0], elt
    );
  }

  // Free
  clReleaseMemObject(static_cast<cl_mem>(buf_Column));
  clReleaseMemObject(static_cast<cl_mem>(buf_Weight));

  // Resize output
  if (batch == 0) {
    THGPUTensor_resize3d(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THGPUTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
    THGPUTensor_resize3d(gradInput, nInputPlane, inputHeight, inputWidth);
  }

  // Return gradInput
  return 1;
}

static int gpunn_SpatialConvolutionMM_accGradParameters(lua_State *L) {
  // Inputs
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  float scale = luaL_optnumber(L, 4, 1);

  THGPUTensor *gradWeight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.GPUTensor");
  THGPUTensor *gradBias = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.GPUTensor");
  THGPUTensor *columns = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.GPUTensor");
  THGPUTensor *ones = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THGPUTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
    THGPUTensor_resize4d(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THGPUTensor_resize2d(ones, outputHeight, outputWidth);
    THGPUTensor_fill(ones, 1);
  }

  // Resize temporary columns
  THGPUTensor_resize2d(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THGPUTensor *gradOutput_n = THGPUTensor_new();

  long m = gradWeight->size[0];
  long n = gradWeight->size[1];
  long k = columns->size[1];
  long m_ = nOutputPlane;
  long k_ = outputHeight * outputWidth;
  // char trans = 't', see this in the loop body

  void* buf_Column = THGPUBlas_clCreateBuffer(n, k ,THGPUTensor_data(columns));
  void* buf_Output = THGPUBlas_clCreateBuffer(k, m ,THGPUTensor_data(gradOutput_n));
  void* buf_Weight = THGPUBlas_clCreateBuffer(n, m ,THGPUTensor_data(gradWeight));
  void* bufX = THGPUBlas_clCreateBuffer(k_, 1 ,THGPUTensor_data(ones));
  void* bufY = THGPUBlas_clCreateBuffer(m_, 1 ,THGPUTensor_data(gradBias));

  PREPARE_AV(columns, avData_col);
  PREPARE_AV(input, avData_im);
  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THGPUTensor_select(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
        *avData_im, input->stride[0], elt,
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        *avData_col
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/gpu/cublas/#cublas-lt-t-gt-gemm)

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THGPUBlas_gemm_opt(
        't', 'n',
        n, m, k,
        scale,
        THGPUTensor_data(columns), k,
        THGPUTensor_data(gradOutput_n), k,
        1, 0,
        THGPUTensor_data(gradWeight), n,
        buf_Column, buf_Output, buf_Weight
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/gpu/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    THGPUBlas_gemv_opt(
        't',
        k_, m_,
        scale,
        THGPUTensor_data(gradOutput_n), k_,
        THGPUTensor_data(ones), 1,
        1,
        THGPUTensor_data(gradBias), 1,
        NULL, bufX, bufY
    );
  }

  clReleaseMemObject(static_cast<cl_mem>(buf_Output));
  clReleaseMemObject(static_cast<cl_mem>(buf_Column));
  clReleaseMemObject(static_cast<cl_mem>(buf_Weight));
  clReleaseMemObject(static_cast<cl_mem>(bufY));
  clReleaseMemObject(static_cast<cl_mem>(bufX));
  // Free
  THGPUTensor_free(gradOutput_n);

  // Resize
  if (batch == 0) {
    THGPUTensor_resize3d(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THGPUTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
  }
  // Return nothing
  return 0;
}

static const struct luaL_Reg gpunn_SpatialConvolutionMM__ [] = {
  {"SpatialConvolutionMM_updateOutput", gpunn_SpatialConvolutionMM_updateOutput},
  {"SpatialConvolutionMM_updateGradInput", gpunn_SpatialConvolutionMM_updateGradInput},
  {"SpatialConvolutionMM_accGradParameters", gpunn_SpatialConvolutionMM_accGradParameters},
  {NULL, NULL}
};

static void gpunn_SpatialConvolutionMM_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialConvolutionMM__, "nn");
  lua_pop(L,1);
}
