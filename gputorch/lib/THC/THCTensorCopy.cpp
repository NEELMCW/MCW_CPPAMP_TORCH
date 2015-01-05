#include "THCGeneral.h"
#include "THGeneral.h"
#include "THCTensor.h"
#include <iostream>
#include "common.h"
#include "amp_math.h"


using namespace std;

/* specific methods */

void THGPUTensor_copyFloat(THGPUTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THGPUTensor_nElement(self) == THFloatTensor_nElement(src), 2, "sizes do not match");

  {
    THGPUTensor *selfc = THGPUTensor_newContiguous(self);
    src = THFloatTensor_newContiguous(src);
    #if 0
    Concurrency::array<float> arrSrc(Concurrency::extent<1>(src->storage->size), src->storage->data);
    Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->storage->size), self->storage->data);
    Concurrency::copy(arrSrc, avSelfCopy);
    #else
    // Hui: Re-implement without any array ctor and by reusing preallocated array_view
    // Note that, providing the following code snippets in lua
    //    local x_cpu    = x:float()
    //    local x_gpu   = x_cpu:gpu()
    //
    //  The call graph of it is draftly listed as belows
    //  (1) THGPUStorage_new        (0 GPU memory allocation, since default size is set to 0 now)
    //  (2) THGPUStorage_resize     (1 GPU memory allocation)
    //  (3) THGPUTensor_copyFloat (now, 0 GPU memory operation)
    //  (4) THGPUTensor_free         (1 GPU memory de-allocation)
    Concurrency::array_view<float, 1> *avSelfCopy= static_cast<Concurrency::array_view<float, 1> *>(self->storage->allocatorContext);
    Concurrency::copy(src->storage->data, src->storage->data+src->storage->size, *avSelfCopy);
    #endif
    THFloatTensor_free(src);
    THGPUTensor_freeCopyTo(selfc, self);
  }
}

/* everything comes down to copy to a tensor of floats */
#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                                                          \
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

IMPLEMENT_TH_CUDA_TENSOR_COPY(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Double)

/* copyGPU */

void THFloatTensor_copyGPU(THFloatTensor *self, struct THGPUTensor *src)
{
  THArgCheck(THFloatTensor_nElement(self) == THGPUTensor_nElement(src), 2, "sizes do not match"); 

  {
    THFloatTensor *selfc = THFloatTensor_newContiguous(self);
    src = THGPUTensor_newContiguous(src);
    #if 0
    Concurrency::array<float> arrSrc(Concurrency::extent<1>(self->storage->size), src->storage->data);
    Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->storage->size), self->storage->data);
    Concurrency::copy(arrSrc, avSelfCopy);
    #else
    Concurrency::array_view<float, 1> *avSrc= static_cast<Concurrency::array_view<float, 1> *>(src->storage->allocatorContext);
    Concurrency::copy(*avSrc, self->storage->data);
    #endif

    THGPUTensor_free(src);
    THFloatTensor_freeCopyTo(selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(TYPEC)                                                       \
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

IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Double)

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

  Concurrency::copy(szh, **sz_);
  Concurrency::copy(sth, **st_);
  THFree(szh);
  THFree(sth);

  *dim_ = dim;
}

void THGPUTensor_kernel_copy(THGPUTensor *self, THGPUTensor *src, Concurrency::array_view<long, 1> *dst_sz,
                             Concurrency::array_view<long, 1> *dst_st, int dst_dim, 
                             Concurrency::array_view<long, 1> *src_sz, Concurrency::array_view<long, 1> *src_st,
                             int src_dim, long n_elem, long innerdim, int nblockx, int nblocky,
                             int nblockz)
{
  Concurrency::extent<3> copyExt(nblockz, nblocky *16, nblockx * 16);
  Concurrency::tiled_extent<1, 16, 16> t_ext(copyExt);
  Concurrency::array_view<long, 1> av_src_st(*src_st);
  Concurrency::array_view<long, 1> av_src_sz(*src_sz);
  Concurrency::array_view<long, 1> av_dst_st(*dst_st);
  Concurrency::array_view<long, 1> av_dst_sz(*dst_sz);
  Concurrency::array_view<float, 1> av_dst(self->storage->size, THGPUTensor_data(self));
  Concurrency::array_view<float, 1> av_src(src->storage->size, THGPUTensor_data(src));
  //Copy Kernel
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 16, 16> tidx) restrict(amp)
  {
    long x = t_ext.tile_dim2;
    long y = t_ext.tile_dim1;
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
        av_dst[Concurrency::index<1>(dst_idx + o)] = av_src[Concurrency::index<1>(src_idx + i)];
      }
    }
  });
}

THC_API void THGPUTensor_copy(THGPUTensor *self, THGPUTensor *src)
{
  THArgCheck(THGPUTensor_nElement(self) == THGPUTensor_nElement(src), 2, "sizes do not match");

  if (THGPUTensor_nDimension(self) == 0) return;

  if(THGPUTensor_isContiguous(self) && THGPUTensor_isContiguous(src))
  {
    DECLARE_BOLT_DEVICE_VECTOR_2(src, srcVec, self, desVec);
    bolt::amp::copy(srcVec.begin(),srcVec.end(),desVec.begin());
  }
  else
  {
    Concurrency::array_view<long, 1> *d_self_sz, *d_self_st, *d_src_sz, *d_src_st;
    int self_dim, src_dim;
    long size = THGPUTensor_nElement(self);
    long innermostdim;

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

    THGPUTensor_kernel_copy(self, src,
                           d_self_sz, d_self_st, self_dim,
                           d_src_sz, d_src_st, src_dim,
                          size, innermostdim, nblocks_x, nblocks_y, nblocks_z);

    delete d_self_st; 
    delete d_self_sz;
    delete d_src_st;
    delete d_src_sz;
  }
}
