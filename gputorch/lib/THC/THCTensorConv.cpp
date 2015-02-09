#include "THCTensorConv.h"
#include "THCGeneral.h"
#include "THCTensorMath.h"
#include "common.h"
#include <stdio.h>

#define CUDA_SHARED_MEM_SIZE (8*1024-32)

#define FOR_KERNEL_SPECIALIZED_DIMENSION(ROWS, COLUMNS, KERNEL) \
  if ((ROWS) == (COLUMNS)) {                                    \
    switch ((ROWS)) {                                           \
      case 3: { KERNEL(3); break; }                             \
      case 4: { KERNEL(4); break; }                             \
      case 5: { KERNEL(5); break; }                             \
      case 6: { KERNEL(6); break; }                             \
      case 7: { KERNEL(7); break; }                             \
      case 8: { KERNEL(8); break; }                             \
      case 9: { KERNEL(9); break; }                             \
      case 10: { KERNEL(10); break; }                           \
      case 11: { KERNEL(11); break; }                           \
      case 12: { KERNEL(12); break; }                           \
      case 13: { KERNEL(13); break; }                           \
      default: { KERNEL(0); break; }                            \
    }                                                           \
  } else {                                                      \
    KERNEL(0);                                                  \
  }
/*
 * API-compatible with THRealTensor_conv2Dmv
 * 3D input, 4D kernel, 3D output
 * matrix vector product like: y <- Ax + beta*y
 */
template <bool swapkernel, int T_kernel_h, int T_kernel_w>
void THGPUTensor_kernel_conv2generic(Concurrency::array_view<float, 1> &input_data, Concurrency::array_view<float, 1> &kernel_data, Concurrency::array_view<float, 1> &output_data,
                                     int input_n, int input_h, int input_w,
                                     int kernel_n, int kernel_h, int kernel_w,
                                     long stride_h, long stride_w, int nOutputPlane, int yblocks)
{
  Concurrency::extent<3> copyExt(1, yblocks * 16, nOutputPlane * 16);
  Concurrency::tiled_extent<1,16,16> t_ext(copyExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,16,16> tidx) restrict(amp)
  {
    // output dimensions
    int output_h = (input_h - kernel_h) / stride_h + 1;
    int output_w = (input_w - kernel_w) / stride_w + 1;

    // xcorr or conv
    int koffset = swapkernel ? kernel_w*kernel_h-1 : 0;

    // nb outputs
    int output_n = kernel_n / input_n;

    // generate offsets according to block/thread ids
    int xx_start = tidx.local[2];
    int xx_end = output_w;
    int xx_step = t_ext.tile_dim2;

    int yy_start = tidx.global[1];
    int yy_end = output_h;
    int yy_step = copyExt[1];// t_ext[1];

    int oo_start = tidx.tile[2];
    int oo_end = oo_start+1;

    int ii_start = (oo_start / output_n) * input_n;
    int ii_end = ii_start + input_n;

    // nb threads, unique thread id
    int tid = t_ext.tile_dim2 * t_ext.tile_dim1 * tidx.local[0] + t_ext.tile_dim2 * tidx.local[1] + tidx.local[2];
    int nthreads = t_ext.tile_dim2 * t_ext.tile_dim1 * t_ext.tile_dim0;

    // iterators
    int oo, ii, xx, yy, kx, ky, kk;

    // do the kernels fit in shared mem ?
    if (input_n*kernel_w*kernel_h <= CUDA_SHARED_MEM_SIZE) {

    // put the kernel in shared memory
    tile_static float shared_kernel[CUDA_SHARED_MEM_SIZE];

    // first thread of each block does the copy
    for (kk = tid; kk < kernel_w*kernel_h*input_n; kk += nthreads) {
      shared_kernel[kk] = kernel_data[input_n*kernel_w*kernel_h*(oo_start % output_n) + kk];
    }
    tidx.barrier.wait();

    // templated kernel size
    if ((T_kernel_w > 0) && (T_kernel_h > 0)) {
      // unrolled convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for(ii = ii_start; ii < ii_end; ii++) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input_data.data();
              input_p += (ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w);
              float *output_p = output_data.data();
              output_p += (oo*output_h*output_w + yy*output_w + xx);
              float *kernel_p = shared_kernel + (ii % input_n)*kernel_w*kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
                for(ky = 0; ky < T_kernel_h; ky++) {
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
                for(ky = 0; ky < T_kernel_h; ky++) {
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    } else {
      // default convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for(ii = ii_start; ii < ii_end; ii++) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input_data.data();
              input_p += (ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w);
              float *output_p = output_data.data();
              output_p += (oo*output_h*output_w + yy*output_w + xx);
              float *kernel_p = shared_kernel + (ii % input_n) * kernel_w * kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
                for(ky = 0; ky < kernel_h; ky++) {
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
                for(ky = 0; ky < kernel_h; ky++) {
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    }

  } else { // not enough shared mem for kernels, simply stream them

    // convolution loop
    for(oo = oo_start; oo < oo_end; oo++) {
      for(ii = ii_start; ii < ii_end; ii++) {
        for(yy = yy_start; yy < yy_end; yy+=yy_step) {
          for(xx = xx_start; xx < xx_end; xx+=xx_step) {
            // Dot product in two dimensions... (between input image and the mask)
            float *input_p = input_data.data();
            input_p += (ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w);
            float *output_p = output_data.data();
            output_p += (oo*output_h*output_w + yy*output_w + xx);
            float *kernel_p = kernel_data.data();
            kernel_p += (((oo % output_n) * input_n + (ii % input_n))*kernel_w*kernel_h + koffset);
            float sum = 0;
            if (swapkernel) {
              for(ky = 0; ky < kernel_h; ky++) {
                for(kx = 0; kx < kernel_w; kx++) {
                  sum += input_p[kx]*(*kernel_p--);
                }
                input_p += input_w;
              }
            } else {
              for(ky = 0; ky < kernel_h; ky++) {
                for(kx = 0; kx < kernel_w; kx++) {
                  sum += input_p[kx]*(*kernel_p++);
                }
                input_p += input_w;
              }
            }
            *output_p += sum;
          }
        }
      }
    }
  }
  });
}


void THGPUTensor_conv2Dmv(THGPUTensor *output, float beta, THGPUTensor *t_,
                                  THGPUTensor *k_, long srow, long scol, const char *type)
{
  int nInputPlane, nInputRows, nInputCols;
  int nKernelRows, nKernelCols;
  int nOutputPlane, nOutputRows, nOutputCols;
  THGPUTensor *input;
  THGPUTensor *kernel;
 
  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(type[0] == 'v' || type[0] == 'f', 7, "type of convolution can 'v' or 'f'");
  THArgCheck(type[1] == 'c' || type[1] == 'x', 7, "type of convolution can 'x' or 'c'");


  input = THGPUTensor_newContiguous(t_);
  if (!(k_->stride[3] == 1) || !(k_->stride[2] == k_->size[3])) {
    kernel = THGPUTensor_newContiguous(k_);
  } else {
    kernel = k_;
  }

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelRows  = kernel->size[2];
  nKernelCols  = kernel->size[3];
  nOutputPlane = kernel->size[0];

  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2,
              "conv2Dmv : Input image is smaller than kernel");

  if (*type == 'F') {
    // output dims
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;

    // use temp buffer
    THGPUTensor *inputP = NULL;
    int firstcall = 1;
    if (firstcall) {
      inputP = THGPUTensor_new();
      firstcall = 0;
    }

    // create a zero-padded input
    long nInputRowsPadded = (nOutputRows - 1) * srow + nKernelRows;
    long nInputColsPadded = (nOutputCols - 1) * scol + nKernelCols;
    THGPUTensor_resize3d(inputP, nInputPlane, nInputRowsPadded, nInputColsPadded);
    THGPUTensor_zero(inputP);

    THGPUTensor *centered = THGPUTensor_new();
    THGPUTensor_narrow(centered, inputP, 2, nKernelCols-1, nInputCols);
    THGPUTensor_narrow(centered, NULL, 1, nKernelRows-1, nInputRows);
    THGPUTensor_copy(centered, input);
    THGPUTensor_free(centered);

    // remap input to newly created tensor
    input = inputP;
    nInputRows = nInputRowsPadded;
    nInputCols = nInputColsPadded;

  } else { // 'v'
    // output dims
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  long nelem = THGPUTensor_nElement(output);
  THGPUTensor_resize3d(output, nOutputPlane, nOutputRows, nOutputCols);

  if (beta == 0 || nelem != THGPUTensor_nElement(output)) {
    THGPUTensor_zero(output);
  } else if (beta != 1) {
    THGPUTensor_mul(output,output, beta);
  }

  int yblocks = (int)(16L / nOutputPlane);
  yblocks = yblocks < 1 ? 1 : yblocks;

  PREPARE_AV(input, pavInput);
  PREPARE_AV(kernel, pavKernel);
  PREPARE_AV(output, pavOutput);

  // convolution: xcorr2 or conv2
  if (type[1] == 'X') {
#define X_CONV_KERNEL(dim)                                              \
    THGPUTensor_kernel_conv2generic <false, (dim), (dim)>(              \
        *pavInput, *pavKernel, *pavOutput,                              \
        nInputPlane, nInputRows, nInputCols,                            \
        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,             \
        srow, scol, nOutputPlane, yblocks);                             \

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelRows, nKernelCols, X_CONV_KERNEL);
#undef X_CONV_KERNEL
  } else { // 'c'
#define C_CONV_KERNEL(dim)                                              \
    THGPUTensor_kernel_conv2generic <true, (dim), (dim)>(               \
        *pavInput, *pavKernel, *pavOutput,                              \
        nInputPlane, nInputRows, nInputCols,                            \
        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,             \
        srow, scol, nOutputPlane, yblocks);                             \

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelRows, nKernelCols, C_CONV_KERNEL);
#undef C_CONV_KERNEL
  }
  THGPUTensor_free(input);
}

/*

 * API-compatible with THRealTensor_conv2Dmm

 * 4D input, 4D kernel, 4D output

 * matrix vector product like: y <- Ax + beta*y

 */

void THGPUTensor_conv2Dmm(THGPUTensor *output, float beta, THGPUTensor *t_,
                          THGPUTensor *k_, long srow, long scol, const char *type)
{
  int nbatch, nInputPlane, nInputRows, nInputCols;
  int nKernelRows, nKernelCols;
  int nOutputPlane, nOutputRows, nOutputCols;

  THGPUTensor *input;
  THGPUTensor *kernel;

  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(type[0] == 'v' || type[0] == 'f', 7, "type of convolution can 'v' or 'f'");
  THArgCheck(type[1] == 'c' || type[1] == 'x', 7, "type of convolution can 'x' or 'c'");

  input = THGPUTensor_newContiguous(t_);
  kernel = THGPUTensor_newContiguous(k_);

  nbatch      = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  nKernelRows  = kernel->size[2];
  nKernelCols  = kernel->size[3];
  nOutputPlane = kernel->size[0];

  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");
  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2,
              "conv2Dmm : Input image is smaller than kernel");

  if (*type == 'F') {
    // output dims
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;

    // use temp buffer
    static THGPUTensor *inputP;
    static int firstcall = 1;

    if (firstcall) {
      inputP = THGPUTensor_new();
      firstcall = 0;
    }

    // create a zero-padded input
    long nInputRowsPadded = (nOutputRows - 1) * srow + nKernelRows;
    long nInputColsPadded = (nOutputCols - 1) * scol + nKernelCols;

    THGPUTensor_resize4d(inputP, nbatch, nInputPlane, nInputRowsPadded, nInputColsPadded);
    THGPUTensor_zero(inputP);
    THGPUTensor *centered = THGPUTensor_new();
    THGPUTensor_narrow(centered, inputP, 3, nKernelCols-1, nInputCols);
    THGPUTensor_narrow(centered, NULL, 2, nKernelRows-1, nInputRows);
    THGPUTensor_copy(centered, input);
    //THGPUTensor_free(centered);
    // remap input to newly created tensor
    THGPUTensor_free(input);

    input = inputP;
    nInputRows = nInputRowsPadded;
    nInputCols = nInputColsPadded;
  } else { // 'v'
    // output dims
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  long nelem = THGPUTensor_nElement(output);
  THGPUTensor_resize4d(output, nbatch, nOutputPlane, nOutputRows, nOutputCols);

  if (beta == 0 || nelem != THGPUTensor_nElement(output)) {
    THGPUTensor_zero(output);
  } else if (beta != 1) {
    THGPUTensor_mul(output,output, beta);
  }

  // gpu blocks & threads:
  int yblocks = (int)(16L / nOutputPlane);
  yblocks = yblocks < 1 ? 1 : yblocks;

  PREPARE_AV(input, pavInput);
  PREPARE_AV(kernel, pavKernel);
  PREPARE_AV(output, pavOutput);

  // convolution: xcorr2 or conv2
  if (type[1] == 'X') {
#define X_CONV_KERNEL(dim)                                              \
    THGPUTensor_kernel_conv2generic <false, (dim), (dim)>(              \
        *pavInput, *pavKernel, *pavOutput,                              \
        nInputPlane, nInputRows, nInputCols,                            \
        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,             \
        srow, scol, nOutputPlane, yblocks);

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelCols, nKernelRows, X_CONV_KERNEL);

#undef X_CONV_KERNEL
  } else { // 'c'
#define C_CONV_KERNEL(dim)                                              \
    THGPUTensor_kernel_conv2generic <true, (dim), (dim)>(               \
        *pavInput, *pavKernel, *pavOutput,                              \
        nInputPlane, nInputRows, nInputCols,                            \
        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,             \
        srow, scol, nOutputPlane, yblocks);                             \

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelCols, nKernelRows, C_CONV_KERNEL);

#undef C_CONV_KERNEL
  }
  // clean

  if (*type != 'F') THGPUTensor_free(input);
  THGPUTensor_free(kernel);
}


void THGPUTensor_kernel_conv2genericrev(THGPUTensor *input, THGPUTensor *kernel, THGPUTensor *output,
                                int input_n, int input_h, int input_w,
                                int kernel_n, int kernel_h, int kernel_w,
                                float alpha, int stride_h, int stride_w,
                                int nKernelPlane, int nInputPlane, int nOutputRows,
                                int cst, int subbatch, int sl)
{
  int ip_stride =  input->stride[0]*sl;
  int k_stride = kernel->stride[0]*sl;

  Concurrency::extent<3> copyExt(1,nInputPlane*16,nKernelPlane*16);
  Concurrency::tiled_extent<1,16,16> t_ext(copyExt);
  Concurrency::array_view<float,1>input_data(Concurrency::extent<1>(input->storage->size),THGPUTensor_data(input));
  input_data.discard_data();
  Concurrency::array_view<float,1>kernel_data(Concurrency::extent<1>(kernel->storage->size),THGPUTensor_data(kernel));
  kernel_data.discard_data();
  Concurrency::array_view<float,1>output_o(Concurrency::extent<1>(output->storage->size),THGPUTensor_data(output));
  output_o.discard_data();

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,16,16> tidx) restrict(amp)
  {
    // output dimensions
    int output_h = input_h - (kernel_h - 1) * stride_h;
    int output_w = input_w - (kernel_w - 1) * stride_w;

    // this thread only processes one output, defined by the block Ids
    int kk = tidx.tile[2];
    int ii = tidx.tile[1];

    // batch id
    int batch = tidx.local[0];

    // kernel id
    int kid = tidx.local[2];
    int nkids = t_ext.tile_dim2;

    // thread ID
    int tid = kid + batch*nkids;
    int nthreads = nkids * t_ext.tile_dim0;

    // one thread only sees one output 
    float *output_data = output_o.data();
    output_data += (kk * input_n + ii) * output_h*output_w;

    // put the output in shared memory
    tile_static float shared_output[CUDA_SHARED_MEM_SIZE];

    // generate tid outputs in shared memory
    float *output_s = shared_output + tid*output_w*output_h;
 
    // convolution loop
    int xx, yy, kx, ky;
    yy = tidx.local[1];

    float *output_p = output_s + yy * output_w;

    for(xx=0; xx<output_w; xx++) {
      // Dot product in two dimensions... (between input image and kernel)
      float *input_p = input_data.data();
      input_p += ((ii + batch*input_n)*input_h*input_w + yy*stride_h*input_w + xx*stride_w) + ip_stride;
      float *kernel_p = kernel_data.data();
      kernel_p += ((kk + batch*kernel_n)*kernel_w*kernel_h) + k_stride;

      float sum = 0;

      for(ky=0; ky<kernel_h; ky++) {
        for(kx=kid; kx<kernel_w; kx+=nkids) {
          sum += input_p[kx]*kernel_p[kx];
        }
        input_p += input_w;
        kernel_p += kernel_w;
      }
      *(output_p++) = sum;
    }
    //tidx.barrier.wait();

    // reduce and write back
    if (yy == 0) {
      // reduce outputs
      for (int k=1; k<nthreads; k++) {
        for (int i=tid; i<output_w*output_h; i+=nthreads) {
          shared_output[i] += shared_output[k*output_h*output_w + i];
        }
      }
      //tidx.barrier.wait();

      // add existing output, and write back
      for (int i=tid; i<output_w*output_h; i+=nthreads) {
        output_data[i] += alpha*shared_output[i];
      }
    }
  });
}

/*

 * API-compatible with THRealTensor_conv2DRevger

 * 3D input, 3D kernel, 4D output

 * like rank1 update

 * A <- xx' + beta*A

 * for sr,sc=1 this is equivalent to xcorr2Dger, but otherwise it is useful for

 * calculating derivatives wrt a kernel that is applied with stride sr,sc != 1

 */

void THGPUTensor_conv2DRevger(THGPUTensor *output, float beta, float alpha,
                              THGPUTensor *input, THGPUTensor *kernel,
                              long srow, long scol)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputRows, nOutputCols;

  THArgCheck(input->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(kernel->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THGPUTensor_newContiguous(input);
  kernel = THGPUTensor_newContiguous(kernel);

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2,
             "conv2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  long nelem = THGPUTensor_nElement(output);
  THGPUTensor_resize4d(output, nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THGPUTensor_nElement(output)) {
    THGPUTensor_zero(output);
  } else if (beta != 1) {
    THGPUTensor_mul(output,output, beta);
  }

  // auto compute nb of blocks and threads
  int cst = 0, subbatch = 0, sl = 0;

  // compute rev conv
  THGPUTensor_kernel_conv2genericrev (input, kernel, output,
                                       nInputPlane, nInputRows, nInputCols,
                                       nKernelPlane, nKernelRows, nKernelCols,
                                       alpha, srow, scol, nKernelPlane, nInputPlane, 
                                       nOutputRows, cst, subbatch, sl);
  // clean
  THGPUTensor_free(input);
  THGPUTensor_free(kernel);
}



/*

 * API-compatible with THRealTensor_conv2DRevgerm

 * 4D input, 4D kernel, 4D output

 * conv2DRevgerm is doing the same thing as conv2DRevger, but with batch inputs

 */

void THGPUTensor_conv2DRevgerm(THGPUTensor *output, float beta, float alpha,
                                       THGPUTensor *input, THGPUTensor *kernel,
                                       long srow, long scol)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelPlane, nKernelRows, nKernelCols;
  long nOutputRows, nOutputCols;
  long nbatch;

  THArgCheck(input->nDimension == 4 , 3, "input: 3D Tensor expected");
  THArgCheck(kernel->nDimension == 4 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THGPUTensor_newContiguous(input);
  kernel = THGPUTensor_newContiguous(kernel);

  nbatch      = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  nKernelPlane = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2,
             "conv2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  long nelem = THGPUTensor_nElement(output);
  THGPUTensor_resize4d(output, nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THGPUTensor_nElement(output)) {
    THGPUTensor_zero(output);
  } else if (beta != 1) {
    THGPUTensor_mul(output,output, beta);
  }

  // kernel is called multiple times
  // (the arbitrary split below is just here to make sure we dont go over 256 threads)
  for (int sl=0; sl<nbatch; sl+=6) {
    // auto compute nb of blocks and threads
    int subbatch = 6;
    if (sl+subbatch > nbatch) subbatch = nbatch - sl;

    int cst = 256 / (subbatch * nOutputRows);
    // compute rev conv

    THGPUTensor_kernel_conv2genericrev (input + input->stride[0]*sl,
                                           kernel + kernel->stride[0]*sl, 
                                           output,
                                           nInputPlane, nInputRows, nInputCols,
                                           nKernelPlane, nKernelRows, nKernelCols,
                                           alpha, srow, scol, nKernelPlane, nInputPlane,
                                           nOutputRows, cst, subbatch, sl);
  }

  // clean
  THGPUTensor_free(input);
  THGPUTensor_free(kernel);
}

///////////////////////////////////

///// ConvolutionMap

/*

 * Description:

 *   base conv2D routine: 3D input, 3D output, 4D kernel

 *

 *   - all chunks of data should be contiguous

 *   - the swapkernel flag can be used to generate a conv2 instead of xcorr2

 *   - the templated kernel size is useful to generate code that's 2x faster

 *     but can be set to 0 to allow arbitrary kernel sizes

 *   ---- the table should have the first dim with the outputs, each output 

 *   ---- should have a fanin set of inputs contiguously

 */

template <bool swapkernel, int T_kernel_h, int T_kernel_w>
void THGPUTensor_kernel_conv2mapgeneric(Concurrency::array_view<float, 1> &input_data, Concurrency::array_view<float, 1> &kernel_data, Concurrency::array_view<float, 1> &output_data,
                                         int input_n, int input_h, int input_w,
                                         int kernel_n, int kernel_h, int kernel_w,
                                         int stride_w, int stride_h,
                                         Concurrency::array_view<float, 1> &table_data, int fanin,
                                         int nOutputPlane, int block_height)
{
  Concurrency::extent<3> copyExt(1,nOutputPlane*16,block_height*16);
  Concurrency::tiled_extent<1,16,16> t_ext(copyExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,16,16> tidx) restrict(amp)
  {
    // output dimensions
    int output_h = (input_h - kernel_h) / stride_h + 1;
    int output_w = (input_w - kernel_w) / stride_w + 1;
    // xcorr or conv
    int koffset = swapkernel ? kernel_w*kernel_h-1 : 0;
    // nb outputs
    // int output_n = kernel_n / fanin;
    // generate offsets according to block/thread ids
    int xx_start = tidx.local[2];
    int xx_end = output_w;
    int xx_step = t_ext.tile_dim2;

    int yy_start = tidx.global[1];
    int yy_end = output_h;
    int yy_step = t_ext[1];
    int oo_start = tidx.tile[2];
    int oo_end = oo_start+1;

    int table_start = oo_start * (fanin * 2);
    int table_end = table_start + (fanin * 2);
    // nb threads, unique thread id
    int tid = t_ext.tile_dim2 * t_ext.tile_dim1 * tidx.local[0] + t_ext.tile_dim2 * tidx.local[1] + tidx.local[2];
    int nthreads = t_ext.tile_dim2 * t_ext.tile_dim1 * t_ext.tile_dim0;

    // iterators
    int oo, ii, xx, yy, kx, ky, kk;
    // do the kernels fit in shared mem ?

    if (kernel_w*kernel_h*kernel_n <= CUDA_SHARED_MEM_SIZE) { 
    // put the kernel in shared memory
    tile_static float shared_kernel[CUDA_SHARED_MEM_SIZE];

    // first thread of each block does the copy
    for (kk = tid; kk < kernel_w*kernel_h*kernel_n; kk += nthreads) {
        shared_kernel[kk] = kernel_data[kk];
    }
    tidx.barrier.wait();

    // templated kernel size
    if ((T_kernel_w > 0) && (T_kernel_h > 0)) {
      // unrolled convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
      for (ii = table_start; ii < table_end; ii = ii + 2) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
                // Dot product in two dimensions... (between input image and the mask)
                float *input_p = input_data.data();
                input_p += ((long)table_data[ii]-1)*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
                float *output_p = output_data.data();
                output_p += oo*output_h*output_w + yy*output_w + xx;
                float *kernel_p = shared_kernel 
                + ((long)table_data[ii + 1]-1) *kernel_w*kernel_h + koffset;
                float sum = 0;
                if (swapkernel) {
                  for(ky = 0; ky < T_kernel_h; ky++) {
                      for(kx = 0; kx < T_kernel_w; kx++) {
                        sum += input_p[kx]*(*kernel_p--);
                      }
                      input_p += input_w;
                  }
                } else {
                  for(ky = 0; ky < T_kernel_h; ky++) {
                      for(kx = 0; kx < T_kernel_w; kx++) {
                        sum += input_p[kx]*(*kernel_p++);
                      }
                      input_p += input_w;
                  }
                }
                *output_p += sum;
            }
          }
      }
      }
    } else {
        // default convolution loop
        for(oo = oo_start; oo < oo_end; oo++) {
        for (ii = table_start; ii < table_end; ii++) {
            for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
                // Dot product in two dims (between input image and the mask)
                float *input_p = input_data.data();
                input_p += ((long)table_data[ii]-1)*input_h*input_w 
                + yy*stride_h*input_w + xx*stride_w;
                float *output_p = output_data.data();
                output_p += oo*output_h*output_w + yy*output_w + xx;
                float *kernel_p = shared_kernel 
                + ((long)table_data[ii + 1]-1) *kernel_w*kernel_h + koffset;
                float sum = 0;
                if (swapkernel) {
                for(ky = 0; ky < kernel_h; ky++) {
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                    }
                    input_p += input_w;
                  }
                } else {
                for(ky = 0; ky < kernel_h; ky++) {
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                    }
                    input_p += input_w;
                  }
                }
                *output_p += sum;
            }
            }
        }
        }
    }
    } else { // not enough shared mem for kernels, simply stream them
    // convolution loop
    for(oo = oo_start; oo < oo_end; oo++) {
        for (ii = table_start; ii < table_end; ii = ii + 2) {
        for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
            // Dot product in two dimensions... (between input image and the mask)
            float *input_p = input_data.data();
            input_p += ((long)table_data[ii]-1)*input_h*input_w 
                + yy*stride_h*input_w + xx*stride_w;
            float *output_p = output_data.data();
            output_p += oo*output_h*output_w + yy*output_w + xx;
            float *kernel_p = kernel_data.data();
            kernel_p += ((long)table_data[ii + 1]-1) *kernel_w*kernel_h + koffset;
            float sum = 0;
            if (swapkernel) {
                for(ky = 0; ky < kernel_h; ky++) {
                for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                }
                input_p += input_w;
                }
            } else {
                for(ky = 0; ky < kernel_h; ky++) {
                for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                }
                input_p += input_w;
                }
            }
            *output_p += sum;
            }
        }
        }
    }
    }
  });
}

/*

 * API-compatible with THRealTensor_conv2Dmv

 * 3D input, 4D kernel, 3D output

 * matrix vector product like: y <- Ax + beta*y

 */

void THGPUTensor_conv2Dmap(THGPUTensor *output, THGPUTensor *input,
                                   THGPUTensor *kernel, long stride_x, long stride_y,
                                   THGPUTensor *table, long fanin)
{
  long nInputPlane, nInputRows, nInputCols;
  long nKernelRows, nKernelCols;
  long nOutputPlane, nOutputRows, nOutputCols;

  THArgCheck(kernel->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(stride_x >= 1, 5, "Stride should be a positive integer");
  THArgCheck(stride_y >= 1, 6, "Stride should be a positive integer");

  input = THGPUTensor_newContiguous(input);
  kernel = THGPUTensor_newContiguous(kernel);
  table = THGPUTensor_newContiguous(table);

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelRows  = kernel->size[1];
  nKernelCols  = kernel->size[2];
  nOutputPlane = kernel->size[0] / fanin;
  // THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols), 2,
              "conv2Dmap : Input image is smaller than kernel");

  // output dims
  nOutputRows = (nInputRows - nKernelRows) / stride_y + 1;
  nOutputCols = (nInputCols - nKernelCols) / stride_x + 1;

  // long nelem = THCudaTensor_nElement(state, output);
  THGPUTensor_resize3d(output, nOutputPlane, nOutputRows, nOutputCols);


  // set the number of blocks and threads
  #if 0
  int nthreads_x = 32;
  int nthreads_y = 8;
  #endif
  int block_height = (int)(16L / nOutputPlane);
  if (block_height < 1)
    block_height = 1;

  PREPARE_AV(input, pavInput);
  PREPARE_AV(kernel, pavKernel);
  PREPARE_AV(output, pavOutput);
  PREPARE_AV(table, pavTable);

#define GENERIC_MAP_KERNEL(dim)                                     \
  THGPUTensor_kernel_conv2mapgeneric <false, (dim), (dim)>(         \
      *pavInput, *pavKernel, *pavOutput, nInputPlane, nInputRows,   \
      nInputCols, nOutputPlane*fanin, nKernelRows, nKernelCols,     \
      stride_x, stride_y, *pavTable, fanin, nOutputPlane, block_height);

  FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelCols, nKernelRows, GENERIC_MAP_KERNEL);
#undef GENERIC_MAP_KERNEL
  // clean
  THGPUTensor_free(input);
  THGPUTensor_free(kernel);
  THGPUTensor_free(table);
}
