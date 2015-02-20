#include "THCTensorCopy.h"
#include "THCGeneral.h"
#include "THGeneral.h"
#include "THCTensor.h"

#include <iostream>
#include "common.h"
#include "amp_math.h"
#include "copyHelpers.h"

using namespace std;

// Maximum number of dimensions allowed for gputorch
#define MAX_DIMS 25

/* specific methods */

void THGPUTensor_copyFloat(THGPUTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THGPUTensor_nElement(self) == THFloatTensor_nElement(src), 2, "sizes do not match");

  {
    THGPUTensor *selfc = THGPUTensor_newContiguous(self);
    src = THFloatTensor_newContiguous(src);
    float* selfc_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(selfc->storage->data));

    THGPUCheck(gpuMemcpy(selfc_ptr, selfc->storageOffset * sizeof(float),
                         src->storage->data + src->storageOffset, 0,
                         THGPUTensor_nElement(self) * sizeof(float),
                         gpuMemcpyHostToDevice));

    THFloatTensor_free(src);
    THGPUTensor_freeCopyTo(selfc, self);
  }
}

/* everything comes down to copy to a tensor of floats */
#define IMPLEMENT_TH_GPU_TENSOR_COPY(TYPEC)                                                           \
void THGPUTensor_copy##TYPEC(THGPUTensor *self, struct TH##TYPEC##Tensor *src)                        \
{                                                                                                     \
  THArgCheck(THGPUTensor_nElement(self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
                                                                                                      \
  {                                                                                                   \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);                                           \
    THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);                                      \
                                                                                                      \
    THFloatTensor_copy##TYPEC(srcf, src);                                                             \
    THGPUTensor_copyFloat(self, srcf);                                                                \
                                                                                                      \
    THLongStorage_free(size);                                                                         \
    THFloatTensor_free(srcf);                                                                         \
  }                                                                                                   \
}

IMPLEMENT_TH_GPU_TENSOR_COPY(Byte)
IMPLEMENT_TH_GPU_TENSOR_COPY(Char)
IMPLEMENT_TH_GPU_TENSOR_COPY(Short)
IMPLEMENT_TH_GPU_TENSOR_COPY(Int)
IMPLEMENT_TH_GPU_TENSOR_COPY(Long)
IMPLEMENT_TH_GPU_TENSOR_COPY(Double)

/* copyGPU */

void THFloatTensor_copyGPU(THFloatTensor *self, struct THGPUTensor *src)
{
  THArgCheck(THFloatTensor_nElement(self) == THGPUTensor_nElement(src), 2, "sizes do not match"); 

  {
    THFloatTensor *selfc = THFloatTensor_newContiguous(self);
    src = THGPUTensor_newContiguous(src);
    float* src_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(src->storage->data));
    THGPUCheck(gpuMemcpy(selfc->storage->data + selfc->storageOffset, 0,
                       src_ptr, src->storageOffset * sizeof(float),
                       THGPUTensor_nElement(src) * sizeof(float),
                       gpuMemcpyDeviceToHost));

    THGPUTensor_free(src);
    THFloatTensor_freeCopyTo(selfc, self);
  }
}

#define IMPLEMENT_TH_GPU_TENSOR_COPY_TO(TYPEC)                                                        \
void TH##TYPEC##Tensor_copyGPU(TH##TYPEC##Tensor *self, struct THGPUTensor *src)                      \
{                                                                                                     \
  THArgCheck(TH##TYPEC##Tensor_nElement(self) == THGPUTensor_nElement(src), 2, "sizes do not match"); \
                                                                                                      \
  {                                                                                                   \
    THLongStorage *size = THGPUTensor_newSizeOf(src);                                                 \
    THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);                                      \
                                                                                                      \
    THFloatTensor_copyGPU(srcf, src);                                                                 \
    TH##TYPEC##Tensor_copyFloat(self, srcf);                                                          \
                                                                                                      \
    THLongStorage_free(size);                                                                         \
    THFloatTensor_free(srcf);                                                                         \
  }                                                                                                   \
}

IMPLEMENT_TH_GPU_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_GPU_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_GPU_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_GPU_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_GPU_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_GPU_TENSOR_COPY_TO(Double)

void THGPUTensor_copyGPU(THGPUTensor *self, THGPUTensor *src)
{
  THGPUTensor_copy(self, src);
}

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

// Copy self->size to device and remove all dims of size=1
static void THGPUTensor_computesz(THGPUTensor *self, Concurrency::array_view<long,1> **sz_,
                                  Concurrency::array_view<long> **st_, int *dim_, long *innermostdim)
{
  long *szh, *sth;
  int i, j, dim;
  long last_sz;

  dim = 0;
  // how many dims with size > 1 ?
  for (i = self->nDimension - 1; i >= 0; i--)
  {
    if (self->size[i] != 1)
      dim++;
  }

  if (dim == 0) THError("Error: using non-contiguous code-path for tensor with all singleton dimensions");  
  Concurrency::extent<1> nDim(dim);
  *sz_ = new Concurrency::array_view<long, 1>(nDim);
  *st_ = new Concurrency::array_view<long, 1>(nDim);
  szh = (long*)THAlloc(sizeof(long)*dim);
  sth = (long*)THAlloc(sizeof(long)*dim);

  j = dim - 1;
  for (i = self->nDimension - 1; i >= 0; i--)
  {
    // ignore dimensions of size 1 to prevent copy bug
    if (self->size[i] != 1)
    {
      sth[j] = self->stride[i];
      if(j == dim - 1) 
      {
        szh[j] = 1;
        *innermostdim = self->size[i];
      }
      else
        szh[j] = szh[j+1] * last_sz; //this makes no sense to me (should be size[i])
      j--;
      last_sz = self->size[i];
    }
  }

  long* sz_ptr = static_cast<long*>(Concurrency::getAllocator().device_data((*sz_)->data()));
  THGPUCheck(gpuMemcpy(sz_ptr, 0, szh, 0, dim * sizeof(long), gpuMemcpyHostToDevice));
  long* st_ptr = static_cast<long*>(Concurrency::getAllocator().device_data((*st_)->data()));
  THGPUCheck(gpuMemcpy(st_ptr, 0, sth, 0, dim * sizeof(long), gpuMemcpyHostToDevice));

  THFree(szh);
  THFree(sth);

  *dim_ = dim;
}

void THGPUTensor_kernel_copy(Concurrency::array_view<float>& av_dst, long dstOffset,
                             Concurrency::array_view<float>& av_src, long srcOffset, 
                             Concurrency::array_view<long, 1> &av_dst_sz,
                             Concurrency::array_view<long, 1> &av_dst_st, int dst_dim, 
                             Concurrency::array_view<long, 1> &av_src_sz, Concurrency::array_view<long, 1> &av_src_st,
                             int src_dim, long n_elem, long innerdim, int nblockx, int nblocky,
                             int nblockz)
{
  Concurrency::extent<3> copyExt(nblockz, nblocky *16, nblockx * 16);
  Concurrency::tiled_extent<1, 16, 16> t_ext(copyExt);

  //Copy Kernel
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 16, 16> tidx) restrict(amp)
  {
    #if 0
    long x = t_ext.tile_dim2;
    long y = t_ext.tile_dim1;
    #endif
    long k = (tidx.tile[0] * (t_ext[2] / t_ext.tile_dim2) * (t_ext[1] / t_ext.tile_dim1) + tidx.tile[1] * (t_ext[2] / t_ext.tile_dim2) + tidx.tile[2] ) * t_ext.tile_dim1 + tidx.local[1];
    //long i_start = threadIdx.x * src_st[src_dim-1];
    long i_start = tidx.local[2] * av_src_st[Concurrency::index<1>(src_dim - 1)];
    //long i_step = blockDim.x * src_st[src_dim-1];
    long i_step = t_ext.tile_dim2 * av_src_st[Concurrency::index<1>(src_dim - 1)]; 
    //long o_start = threadIdx.x * dst_st[dst_dim-1];
    long o_start = tidx.local[2] * av_dst_st[Concurrency::index<1>(src_dim - 1)];
    //long o_step = blockDim.x * dst_st[dst_dim-1];
    long o_step = t_ext.tile_dim2 * av_dst_st[Concurrency::index<1>(src_dim - 1)];
    long o_end = innerdim * av_dst_st[Concurrency::index<1>(src_dim - 1)];
    if (((k + 1) * innerdim) <= n_elem) // too safe
    {
      long dst_idx = 0;
      long dst_rest = k * innerdim;
      for (int dim = 0; dim < dst_dim; dim++)
      {
        dst_idx += (dst_rest / av_dst_sz[Concurrency::index<1>(dim)]) * av_dst_st[Concurrency::index<1>(dim)];
        dst_rest = dst_rest % av_dst_sz[Concurrency::index<1>(dim)];
      }
      long src_idx = 0;
      long src_rest = k * innerdim;
      for (int dim = 0; dim < src_dim; dim++)
      {
        src_idx += (src_rest / av_src_sz[Concurrency::index<1>(dim)]) * av_src_st[Concurrency::index<1>(dim)];
        src_rest = src_rest % av_src_sz[Concurrency::index<1>(dim)];
      }
      for (int i = i_start, o = o_start; o < o_end; i += i_step, o += o_step)
      {
        av_dst[Concurrency::index<1>(dstOffset + dst_idx + o)] = av_src[Concurrency::index<1>(srcOffset + src_idx + i)];
      }
    }
  });
}

THC_API void THGPUTensor_copy(THGPUTensor *self, THGPUTensor *src)
{
  // Avoid unnecessary copy
  if (self == src)
    return;

  long totalElements = THGPUTensor_nElement(self);

  THArgCheck(totalElements == THGPUTensor_nElement(src), 2,
             "sizes do not match");

  THArgCheck(THGPUTensor_nDimension(self) <= MAX_DIMS, 2,
             "Copy only supported for <= 25 dimensions");
  THArgCheck(THGPUTensor_nDimension(src) <= MAX_DIMS, 3,
             "Copy only supported for <= 25 dimensions");

  if (THGPUTensor_nDimension(self) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }
  if((THGPUTensor_isContiguous(self) && THGPUTensor_isContiguous(src)) ||
    (totalElements == 1))
  {
    float* self_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(self->storage->data));
    float* src_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(src->storage->data));
    // TODO: Async copy
    THGPUCheck(gpuMemcpy(self_ptr, self->storageOffset * sizeof(float),
               src_ptr, src->storageOffset * sizeof(float),
               totalElements * sizeof(float),
               gpuMemcpyDeviceToDevice));
  }
  else
  {
    Concurrency::array_view<long, 1> *d_self_sz, *d_self_st, *d_src_sz, *d_src_st;
    int self_dim, src_dim;
    long size = THGPUTensor_nElement(self);
    long innermostdim;

    // Data is valid only in device side of d_src_sz, d_src_st, d_self_sz, d_self_st
    THGPUTensor_computesz(src, &d_src_sz, &d_src_st, &src_dim, &innermostdim);
    THGPUTensor_computesz(self, &d_self_sz, &d_self_st, &self_dim, &innermostdim);

    int nblocks = ceil((float)size / (16 * innermostdim ));

    // if nblocks greater than 65535 then we need to open a second dimension
    #define __MAX_NUM_BLOCKS_PER_GRID_DIM__ 65535

    // The configuration below can deal with Tensors 
    // of size up to 65535 * 65535 * 65535 * 16 elements.
    int nblocks_x = (nblocks > __MAX_NUM_BLOCKS_PER_GRID_DIM__) ? __MAX_NUM_BLOCKS_PER_GRID_DIM__ : nblocks;
    int number_blocks_dim_x = DIVUP(nblocks, nblocks_x);
    int nblocks_y = (number_blocks_dim_x > __MAX_NUM_BLOCKS_PER_GRID_DIM__) ? __MAX_NUM_BLOCKS_PER_GRID_DIM__ : number_blocks_dim_x;
    int number_blocks_dim_y = DIVUP(nblocks, nblocks_x * nblocks_y);
    int nblocks_z = number_blocks_dim_y;

    PREPARE_AV(self, avSelf);
    PREPARE_AV(src, avSrc);
    d_self_sz->discard_data();
    d_self_st->discard_data();
    d_src_sz->discard_data();
    d_src_st->discard_data();
    THGPUTensor_kernel_copy(*avSelf, self->storageOffset, *avSrc, src->storageOffset,
                           *d_self_sz, *d_self_st, self_dim,
                           *d_src_sz, *d_src_st, src_dim,
                           size, innermostdim, nblocks_x, nblocks_y, nblocks_z);

    delete d_self_st; 
    delete d_self_sz;
    delete d_src_st;
    delete d_src_sz;
  }
}
