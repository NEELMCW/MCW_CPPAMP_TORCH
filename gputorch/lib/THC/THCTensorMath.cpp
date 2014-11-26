#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"
#include "amp_math.h"
#include "THBlas.h"
#include "THCBlas.h"
#include<algorithm>
#include<utility>
#include<numeric>
#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"

#define NB_THREADS_PER_BLOCK 256


void THCudaTensor_fill(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  bolt::amp::device_vector<float> Dself(self->storage->data,self->storage->data + self->storage->size);
  bolt::amp::fill(Dself.begin(),Dself.end(),value);
  THCudaTensor_copy(self_, self);
  if (self != self_)
  {
    THCudaStorage_free(self->storage);
    THCudaTensor_free(self);
  }
}

void THCudaTensor_zero(THCudaTensor *self_)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  bolt::amp::device_vector<float> Dself(self->storage->data,self->storage->data + self->storage->size);
  bolt::amp::fill(Dself.begin(),Dself.end(),0);
  THCudaTensor_copy(self_, self);
  if (self != self_)
  {
    THCudaStorage_free(self->storage);
    THCudaTensor_free(self);
  }
}


void THCudaTensor_zeros(THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_zero(r_);
}

void THCudaTensor_ones(THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_fill(r_, 1);
}

void THCudaTensor_reshape(THCudaTensor *r_, THCudaTensor *t, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_copy(r_, t);
}

long THCudaTensor_numel(THCudaTensor *t)
{
  return THCudaTensor_nElement(t);
}


struct addvalue_functor
{
  const float value;

  addvalue_functor(float value_) restrict(cpu,amp) : value(value_) {}

  float operator()(const float& x) const restrict(cpu,amp) 
  {
    return (x+value);
  }
};

void THCudaTensor_add(THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THCudaTensor_resizeAs(self_, src_);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaTensor *src = THCudaTensor_newContiguous(src_);
  bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> Dsrc_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), addvalue_functor(value));
  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

struct mulvalue_functor
{
  const float value;
  mulvalue_functor(float value_)restrict(cpu,amp) : value(value_) {}
  float operator()(const float& x) const restrict(cpu,amp)
  {
    return (x*value);
  }
};

void THCudaTensor_mul(THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THCudaTensor_resizeAs(self_, src_);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaTensor *src = THCudaTensor_newContiguous(src_);
  bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> Dsrc_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), mulvalue_functor(value));
  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

struct divvalue_functor
{
  const float value;
  divvalue_functor(float value_)restrict(amp,cpu) : value(value_) {}
  float operator()(const float& x) const restrict(amp,cpu)
  {
    return (x/value);
  }
};

void THCudaTensor_div(THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THCudaTensor_resizeAs(self_, src_);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  THCudaTensor *src = THCudaTensor_newContiguous(src_);
  bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> Dsrc_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), divvalue_functor(value));
  long size = THCudaTensor_nElement(self);
  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}


void THCudaTensor_cadd(THCudaTensor *self_, THCudaTensor* src1, float value, THCudaTensor *src2)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);

    if (self_ != src1) {
      src1 = THCudaTensor_newContiguous(src1);
      THCudaTensor_copy(self, src1);
      THCudaTensor_free(src1);
    }

    src2 = THCudaTensor_newContiguous(src2);

    THCudaBlas_axpy(THCudaTensor_nElement(self), value, THCudaTensor_data(src2), 1, THCudaTensor_data(self), 1);

    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_cmul(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);
  
    bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
    bolt::amp::device_vector<float> Dsrc1_data(THCudaTensor_data(src1), THCudaTensor_data(src1)+THCudaTensor_nElement(src1));
    bolt::amp::device_vector<float> Dsrc2_data(THCudaTensor_data(src2), THCudaTensor_data(src2)+THCudaTensor_nElement(src2));

    bolt::amp::transform(Dsrc2_data.begin(), Dsrc2_data.end(), Dsrc1_data.begin(), Dself_data.begin(), bolt::amp::multiplies<float>());

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_cdiv(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);

    bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
    bolt::amp::device_vector<float> Dsrc1_data(THCudaTensor_data(src1), THCudaTensor_data(src1)+THCudaTensor_nElement(src1));
    bolt::amp::device_vector<float> Dsrc2_data(THCudaTensor_data(src2), THCudaTensor_data(src2)+THCudaTensor_nElement(src2));

    std::transform(Dsrc1_data.begin(), Dsrc1_data.end(), Dsrc2_data.begin(), Dself_data.begin(), bolt::amp::divides<float>());


    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
  
}

void THCudaTensor_kernel_addcmul(float *data, float value, float *src1, float *src2, long size, const int nThreadPerBlock, int nBlockPerRow, int nBlockPerColumn)
{
    Concurrency::array_view<float,1> src1Data(Concurrency::extent<1>(size),src1);
    Concurrency::array_view<float,1> src2Data(Concurrency::extent<1>(size),src2);
    Concurrency::array_view<float,1> Data(Concurrency::extent<1>(size),data);
    const int nthreads = 256;
    nBlockPerRow = (nBlockPerRow + (nthreads -1)) & ~(nthreads -1);
    Concurrency::extent<2> gridExt(nBlockPerColumn,nBlockPerRow);
   
    Concurrency::tiled_extent<1,nthreads> t_ext(gridExt);
 
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,nthreads>tidx) restrict(amp)
    {
        long k = (((tidx.tile[0] * t_ext[1]/t_ext.tile_dim1) + tidx.tile[1]) * t_ext.tile_dim1) + tidx.local[1];
        if(k < size)
        {
            Data[Concurrency::index<1>(k)] += value*src1Data[Concurrency::index<1>(k)]*src2Data[Concurrency::index<1>(k)];
        }
      
    });
    Data.synchronize();
    src1Data.synchronize();
    src2Data.synchronize();
}


void THCudaTensor_addcmul(THCudaTensor *self_, THCudaTensor* t, float value, THCudaTensor *src1, THCudaTensor *src2)
{

  if(self_ != t)
  {
    THCudaTensor_resizeAs(self_, t);
    THCudaTensor_copy(self_, t);
  }

    THCudaTensor_resizeAs(self_, src1);
    THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");
  
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    THCudaTensor *temp1 = src1;
    THCudaTensor *temp2 = src2;
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);

    int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
    THCudaGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);

    THCudaTensor_kernel_addcmul(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size, nThreadPerBlock, nBlockPerRow,nBlockPerColumn);

    THCudaTensor_copy(self_, self);
    if (src1 != temp1)
    {
        THCudaStorage_free(src1->storage);
        THCudaTensor_free(src1);
    }
    if (src2 != temp2)
    {
        THCudaStorage_free(src2->storage);
        THCudaTensor_free(src2);
    }
    if (self != self_)
    {
        THCudaStorage_free(self->storage);
        THCudaTensor_free(self);
    }
}

void THCudaTensor_kernel_addcdiv(float *data, float value, float *src1, float *src2, long size, const int nThreadPerBlock, int nBlockPerRow, int nBlockPerColumn)
{

  
    Concurrency::array_view<float,1> src1Data(Concurrency::extent<1>(size),src1);
    Concurrency::array_view<float,1> src2Data(Concurrency::extent<1>(size),src2);
    Concurrency::array_view<float,1> Data(Concurrency::extent<1>(size),data);

    const int nthreads = 256;
    nBlockPerRow = (nBlockPerRow + (nthreads -1)) & ~(nthreads -1);
    Concurrency::extent<2> gridExt(nBlockPerColumn,nBlockPerRow);
    Concurrency::tiled_extent<1,nthreads> t_ext(gridExt);

    bolt::amp::device_vector<float> divided_data(size);
    bolt::amp::device_vector<float> Dsrc1_data(src1, src1+ size);
    bolt::amp::device_vector<float> Dsrc2_data(src2, src2+ size);

    bolt::amp::transform(Dsrc1_data.begin(), Dsrc1_data.end(), Dsrc2_data.begin(), divided_data.begin(), bolt::amp::divides<float>());

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,nthreads>tidx) restrict(amp)
    {
        long k = (((tidx.tile[0] * t_ext[1]/t_ext.tile_dim1) + tidx.tile[1]) * t_ext.tile_dim1) + tidx.local[1];
        if(k < size)
        {
            Data[Concurrency::index<1>(k)] += (float) value * divided_data[k]; 
        }
      
    });
    Data.synchronize();
}


void THCudaTensor_addcdiv(THCudaTensor *self_, THCudaTensor *t, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  if(self_ != t)
  {
    THCudaTensor_resizeAs(self_, t);
    THCudaTensor_copy(self_, t);
  }
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");

        THCudaTensor *self = THCudaTensor_newContiguous(self_);
        THCudaTensor *temp1 = src1;
        THCudaTensor *temp2 = src2;
        long size = THCudaTensor_nElement(self);
        src1 = THCudaTensor_newContiguous(src1);
        src2 = THCudaTensor_newContiguous(src2);

        int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
        THCudaGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);

        THCudaTensor_kernel_addcdiv(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size, nThreadPerBlock, nBlockPerRow,nBlockPerColumn);

        THCudaTensor_copy(self_, self);
        if (src1 != temp1)
        {
            THCudaStorage_free(src1->storage);
            THCudaTensor_free(src1);
        }
        if (src2 != temp2)
        {
            THCudaStorage_free(src2->storage);
            THCudaTensor_free(src2);
        }
        if (self != self_)
        {
            THCudaStorage_free(self->storage);
            THCudaTensor_free(self);
        }
}

float THCudaTensor_dot(THCudaTensor *self, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self) == THCudaTensor_nElement(src), 2, "size do not match");
  {
    self = THCudaTensor_newContiguous(self);
    src = THCudaTensor_newContiguous(src);
    float result = THCudaBlas_dot(THCudaTensor_nElement(self),
                                  THCudaTensor_data(self), 1,
                                  THCudaTensor_data(src), 1);
    THCudaTensor_free(src);
    THCudaTensor_free(self);
    return result;
  }
}

float THCudaTensor_minall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self),THCudaTensor_data(self) + THCudaTensor_nElement(self));
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THCudaTensor_nElement(self), (float)(THInf), bolt::amp::minimum<float>());
  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_maxall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self),THCudaTensor_data(self) + THCudaTensor_nElement(self));
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THCudaTensor_nElement(self), (float)(-THInf), bolt::amp::maximum<float>());
  THCudaTensor_free(self);
  return result;
}



float THCudaTensor_sumall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self),THCudaTensor_data(self) + THCudaTensor_nElement(self));
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THCudaTensor_nElement(self), (float)(0), bolt::amp::plus<float>());
  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_prodall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self),THCudaTensor_data(self) + THCudaTensor_nElement(self));
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THCudaTensor_nElement(self), (float)(1), bolt::amp::multiplies<float>());
  THCudaTensor_free(self);
  return result;
}


struct dim4 {
    unsigned arr[4];

    dim4(unsigned init=0) {
        for(unsigned i=0; i<4; i++) { arr[i] = init; }
    }

    unsigned& operator[](const unsigned& idx) { return arr[idx]; }
};



/* Reduce one of the outer dimensions of a tensor
 *
 * For an n-d tensor (n <= 4) where the reduction is *not* along the innermost
 * dimension:
 *
 * - block.x and grid.x make up the innermost dimension;
 * - The reduced dimension is looped over inside a block; and
 * - grid.y and grid.z are the remaining two dimensions (if any).
 * - block.y and block.z are not used as we're limited to 512 or 1024 threads
 *   in the block.
 *
 * For sizes/strides, index 3 is the reduced dimension, while the remaining
 * indices are for the remaining dimensions with index 0 the innermost dimension.
 *
 * Reduction along the innermost dimension is handled in a separate kernel.
 */

template<class BinaryFunction, class UnaryFunction>
void THCudaTensor_kernel_transformReduceOuterDim(THCudaTensor *tgt, THCudaTensor *src_,
                                                unsigned int tgtSz, unsigned int srcSz,
                                                unsigned int src_stride[], unsigned int tgt_stride[],
                                                unsigned int size[], UnaryFunction unary_op,
                                                float init, BinaryFunction binary_op,
                                                unsigned int gridConf[])
{
  const size_t reduce = 3;
  Concurrency::array_view<float, 1> avTgt(THCudaTensor_nElement(tgt), THCudaTensor_data(tgt));
  Concurrency::array_view<float, 1> avSrc(THCudaTensor_nElement(src_), THCudaTensor_data(src_));
  Concurrency::array_view<unsigned int, 1> avSrc_stride(4, src_stride);
  Concurrency::array_view<unsigned int, 1> avTgt_stride(4, tgt_stride);
  Concurrency::array_view<unsigned int, 1> avSize(4, size);
  gridConf[0] = (gridConf[0] + 255) & ~255;
  Concurrency::extent<3> grdExt(gridConf[2], gridConf[1], gridConf[0]);
  Concurrency::tiled_extent<1, 1, 256> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 1, 256> tidx) restrict(amp)
  {
    for (unsigned z = tidx.tile[0]; z < avSize[2] ; z += t_ext[0]/tidx.tile_dim0)
    {
      for (unsigned y = tidx.tile[1]; y < avSize[1] ; y += t_ext[1]/tidx.tile_dim1)  
      {
        for (unsigned col = tidx.global[2]; col < avSize[0]; col += t_ext[2]) 
        {
          float acc = init;
          for (unsigned i = 0; i < avSize[reduce]; i++)
          {
            acc = binary_op(acc, unary_op(avSrc[z * avSrc_stride[2] + y * avSrc_stride[1] + col + i * avSrc_stride[reduce]]));
          }
          avTgt[z * avTgt_stride[2] + y * avTgt_stride[1] + col] = float(acc);
        }
      }
    }
  });
  avTgt.synchronize();
}

template<class BinaryFunction, class UnaryFunction>
void THCudaTensor_transformReduceOuterDim(THCudaTensor *tgt, THCudaTensor *src, long rdim,
                                         UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  const size_t reduce = 3;
  unsigned int src_stride[4] = { 0, 0, 0, 0 };
  unsigned int tgt_stride[4] = { 0, 0, 0, 0 };
  unsigned int size[4] = { 1, 1, 1, 1 };
  unsigned int gridConfig[3];
  unsigned ndim = THCudaTensor_nDimension(src);
  for (unsigned idim = 0, o = ndim - 2; idim < ndim; idim++) 
  {
    unsigned odim = idim == rdim ? reduce : o--;
    src_stride[odim] = THCudaTensor_stride(src, idim);
    tgt_stride[odim] = THCudaTensor_stride(tgt, idim);
    size[odim] = THCudaTensor_size(src, idim);
  }
  const unsigned nThreadPerBlock = 256;
  unsigned nBlockPerColumn = (size[0] + nThreadPerBlock - 1) / nThreadPerBlock;
  //dim3 threads(nThreadPerBlock);
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  gridConfig[0] = std::min(maxGridDim, nBlockPerColumn);
  gridConfig[1] = std::min(maxGridDim, size[1]);
  gridConfig[2] = std::min(maxGridDim, size[2]);
  THCudaTensor_kernel_transformReduceOuterDim(tgt,
                                             src, THCudaTensor_nElement(src), THCudaTensor_nElement(tgt),
                                             src_stride, tgt_stride, size, unary_op, init, binary_op,gridConfig);
}




/* Reduce the innermost dimension of a tensor
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */

template<class UnaryFunction, class BinaryFunction>
void THCudaTensor_kernel_transformReduceInnermostDim(THCudaTensor *tgt, THCudaTensor *src_,unsigned int tgtSz,
                                                    unsigned int srcSz, unsigned int src_stride[],
                                                    unsigned int tgt_stride[], unsigned int size[4],
                                                    UnaryFunction unary_op, float init,
                                                    BinaryFunction binary_op, unsigned int gridConf[])
{
  Concurrency::array_view<float, 1> avTgt(THCudaTensor_nElement(tgt), THCudaTensor_data(tgt));
  Concurrency::array_view<float, 1> avSrc(THCudaTensor_nElement(src_), THCudaTensor_data(src_));
  Concurrency::array_view<unsigned int, 1> avSrc_stride(4, src_stride);
  Concurrency::array_view<unsigned int, 1> avTgt_stride(4, tgt_stride);
  Concurrency::array_view<unsigned int, 1> avSize(4, size);
  Concurrency::extent<3> grdExt(gridConf[2], gridConf[1] * 16, gridConf[0] * 32);
  Concurrency::tiled_extent<1, 16, 32> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 16, 32> tidx) restrict(amp)
  {
    tile_static float sbuf[16][32]; // 8kB
    for (unsigned z = tidx.tile[0]; z < avSize[3] ; z += t_ext[0])
    {
      for (unsigned x = tidx.tile[2]; x < avSize[2] ; x += t_ext[2])
      {
        for (unsigned bRow = tidx.tile_origin[1]; bRow < avSize[1]; bRow += grdExt[1]) 
        {
          float acc = init;
          unsigned row = bRow + tidx.local[1];
          //float *src = src_ + z * src_stride[3] + x * src_stride[2] + row * src_stride[1];
          bool reducing = tidx.local[2] < t_ext.tile_dim1 && bRow + tidx.local[2] < avSize[1] && tidx.local[1] == 0;
          for (unsigned bCol = 0; bCol < avSize[0]; bCol += t_ext.tile_dim2) 
          {
            sbuf[tidx.local[1]][tidx.local[2]] = init;
            unsigned col = bCol + tidx.local[2];
            if(row < avSize[1] && col < avSize[0]) 
            {
              sbuf[tidx.local[1]][tidx.local[2]] = unary_op(avSrc[z * avSrc_stride[3] + x * avSrc_stride[2] + row * avSrc_stride[1] + col]);
            }
            tidx.barrier.wait();
            float* line = &sbuf[tidx.local[1]][0];
            for (unsigned s = 16; s > 1; s >>= 1) 
            {
              if (row < avSize[1] && tidx.local[2] < s) 
              {
                line[tidx.local[2]] = binary_op(line[tidx.local[2]], line[tidx.local[2] + s]);
              }
              tidx.barrier.wait();
            }
            if (reducing)
            {
              sbuf[tidx.local[2]][0] = binary_op(sbuf[tidx.local[2]][0], sbuf[tidx.local[2]][1]);
              acc = binary_op(acc, sbuf[tidx.local[2]][0]);
            }
            tidx.barrier.wait();
          }
          if (reducing)
          {
            unsigned row = bRow + tidx.local[2];
            unsigned tgt_offset = z * avTgt_stride[3] + x * avTgt_stride[2];
            avTgt[tgt_offset + row] = acc;
          }
        }
      }
    }
  });
  avTgt.synchronize();
}

template<class UnaryFunction, class BinaryFunction>
void THCudaTensor_transformReduceInnermostDim(THCudaTensor *tgt, THCudaTensor *src, 
                                             UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  unsigned int src_stride[4] = { 0, 0, 0, 0 };
  unsigned int tgt_stride[4] = { 0, 0, 0, 0 };
  unsigned int size[4] = { 1, 1, 1, 1 };
  unsigned int gridConfig[3];
  unsigned ndim = THCudaTensor_nDimension(src);
  for (unsigned dim = 0; dim < ndim; dim++) 
  {
    unsigned odim = ndim - 1 - dim;
    src_stride[odim] = THCudaTensor_stride(src, dim);
    tgt_stride[odim] = THCudaTensor_stride(tgt, dim);
    size[odim] = THCudaTensor_size(src, dim);
  }
  //std::cout<<"InnerDim kernel"<<std::endl;
  unsigned nBlockPerRow = (size[1] + 16 - 1) / 16;
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  gridConfig[0]= std::min(maxGridDim, size[2]);
  gridConfig[1]= std::min(maxGridDim, nBlockPerRow);
  gridConfig[2] = std::min(maxGridDim, size[3]);
  THCudaTensor_kernel_transformReduceInnermostDim(tgt, src,
                                                 THCudaTensor_nElement(tgt), THCudaTensor_nElement(src),
                                                 src_stride, tgt_stride, size, unary_op, init,
                                                 binary_op, gridConfig);
}



template<class UnaryFunction, class BinaryFunction>
void THCudaTensor_transformReduceDim(THCudaTensor *self_, THCudaTensor *src,
        long dimension, UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(src), 3, "dimension out of range");
  THArgCheck(THCudaTensor_nDimension(src) <= 4, 2, "too many dimensions (>4)");

  THLongStorage *dim = THCudaTensor_newSizeOf(src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(self_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);

  if(dimension == THCudaTensor_nDimension(src)-1) {
    THCudaTensor_transformReduceInnermostDim(self, src, unary_op, init, binary_op);
  } else {
    THCudaTensor_transformReduceOuterDim(self, src, dimension, unary_op, init, binary_op);
  }

  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}



template<class BinaryFunction>
void THCudaTensor_reduceDim(THCudaTensor *self_, THCudaTensor *src, long dimension, float init, BinaryFunction binary_op)
{
  THCudaTensor_transformReduceDim(self_, src, dimension, bolt::amp::identity<float>(), init, binary_op);
}


void THCudaTensor_sum(THCudaTensor *self, THCudaTensor *src, long dimension)
{
  return THCudaTensor_reduceDim(self, src, dimension, 0.0f, bolt::amp::plus<float>());
}

void THCudaTensor_prod(THCudaTensor *self, THCudaTensor *src, long dimension)
{
  return THCudaTensor_reduceDim(self, src, dimension, 1.0f, bolt::amp::multiplies<float>());
}

void THCudaTensor_max(THCudaTensor *self, THCudaTensor *indices, THCudaTensor *src, long dimension)
{
  const float minfloat32 = -3.402823466e+38f;
  return THCudaTensor_reduceDim(self, src, dimension, minfloat32, bolt::amp::maximum<float>());
}


void THCudaTensor_min(THCudaTensor *self, THCudaTensor* indices, THCudaTensor *src, long dimension)
{
  const float maxfloat32 = 3.402823466e+38f;
  return THCudaTensor_reduceDim(self, src, dimension, maxfloat32, bolt::amp::minimum<float>());
}

void THCudaTensor_addmv(THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(r_, t);
    THCudaTensor_copy(r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THCudaBlas_gemv('n', mat->size[0], mat->size[1],
                    alpha, THCudaTensor_data(mat), mat->stride[1],
                    THCudaTensor_data(vec), vec->stride[0],
                  beta, THCudaTensor_data(r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THCudaBlas_gemv('t',  mat->size[1], mat->size[0],
                  alpha, THCudaTensor_data(mat), mat->stride[0],
                  THCudaTensor_data(vec), vec->stride[0],
                  beta, THCudaTensor_data(r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cmat = THCudaTensor_newContiguous(mat);

    THCudaBlas_gemv('t',  mat->size[1], mat->size[0],
                  alpha, THCudaTensor_data(cmat), cmat->stride[0],
                  THCudaTensor_data(vec), vec->stride[0],
                  beta, THCudaTensor_data(r_), r_->stride[0]);

    THCudaTensor_free(cmat);
  }
}

void THCudaTensor_addmm(THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *m1, THCudaTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THCudaTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if(t != r_)
  {
    THCudaTensor_resizeAs(r_, t);
    THCudaTensor_copy(r_, t);
  }

  /* r_ */
  if(r_->stride[0] == 1)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1)
  {
    THCudaTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    r__ = THCudaTensor_newWithSize2d(r_->size[1], r_->size[0]);
    THCudaTensor_copy(r__, r_);
    THCudaTensor_transpose(r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THCudaTensor_newContiguous(m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THCudaTensor_newContiguous(m2);
  }

  /* do the operation */
  THCudaBlas_gemm(transpose_m1,
                transpose_m2,
                r__->size[(transpose_r == 'n' ? 0 : 1)],
                r__->size[(transpose_r == 'n' ? 1 : 0)],
                m1_->size[(transpose_r == 'n' ? 1 : 0)],
                alpha,
                THCudaTensor_data(m1_),
                (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                THCudaTensor_data(m2_),
                (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                beta,
                THCudaTensor_data(r__),
                r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THCudaTensor_free(m1_);

  if(m2_ != m2)
    THCudaTensor_free(m2_);

  if(r__ != r_)
    THCudaTensor_freeCopyTo(r__, r_);
}

void THCudaTensor_addr(THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(r_, t);
    THCudaTensor_copy(r_, t);
  }

  if(beta != 1)
    THCudaTensor_mul(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THCudaBlas_ger(vec1->size[0], vec2->size[0],
                 alpha, THCudaTensor_data(vec1), vec1->stride[0],
                 THCudaTensor_data(vec2), vec2->stride[0],
                 THCudaTensor_data(r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THCudaBlas_ger(vec2->size[0], vec1->size[0],
                 alpha, THCudaTensor_data(vec2), vec2->stride[0],
                 THCudaTensor_data(vec1), vec1->stride[0],
                 THCudaTensor_data(r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cr = THCudaTensor_newClone(r_);

    THCudaBlas_ger(vec2->size[0], vec1->size[0],
                 alpha, THCudaTensor_data(vec2), vec2->stride[0],
                 THCudaTensor_data(vec1), vec1->stride[0],
                 THCudaTensor_data(cr), cr->stride[0]);

    THCudaTensor_freeCopyTo(cr, r_);
  }
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                        \
  struct NAME##_functor                                                      \
  {                                                                          \
    float operator()(const float& x) const               \
    {                                                                        \
      return CFUNC(x);                                                       \
    }                                                                        \
  };                                                                         \
                                                                             \
  void THCudaTensor_##NAME(THCudaTensor *self_, THCudaTensor *src)           \
  {                                                                          \
    THCudaTensor_resizeAs(self_, src);                                       \
    THCudaTensor *self = THCudaTensor_newContiguous(self_);                  \
    src = THCudaTensor_newContiguous(src);                                   \
    long size = THCudaTensor_nElement(self);                                 \
                                                                             \
    std::vector<float> self_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));\
    std::vector<float> src_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));\
    std::transform(src_data.begin(), src_data.end(), self_data.begin(),NAME##_functor());\
    std::copy(self_data.begin(), self_data.end(),self->storage->data);\
                                                                             \
    THCudaTensor_free(src);                                                  \
    THCudaTensor_freeCopyTo(self, self_);                                    \
  }

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, fabs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, roundf)

struct pow_functor
{
  const float value;

  pow_functor(float value_) restrict(amp,cpu) : value(value_) {}

  float operator()(const float& x) const restrict(amp,cpu)
  {
      return Concurrency::fast_math::pow(x, value);
      return 0;
  }
};

void THCudaTensor_pow(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_resizeAs(self_, src);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);
  bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> Dsrc_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), pow_functor(value));
  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

struct atan2_functor
{
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::atan2f(x, y);
  }
};
void THCudaTensor_atan2(THCudaTensor *self_, THCudaTensor *tx, THCudaTensor *ty)
{
  THCudaTensor_resizeAs(self_, tx);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  tx = THCudaTensor_newContiguous(tx);
  ty = THCudaTensor_newContiguous(ty);

  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> tx_data(THCudaTensor_data(tx), THCudaTensor_data(tx)+THCudaTensor_nElement(tx));
  bolt::amp::device_vector<float> ty_data(THCudaTensor_data(ty), THCudaTensor_data(ty)+THCudaTensor_nElement(ty));
  bolt::amp::transform(tx_data.begin(), tx_data.end(), ty_data.begin(), self_data.begin(), atan2_functor());

  THCudaTensor_free(tx);
  THCudaTensor_free(ty);
  THCudaTensor_freeCopyTo(self, self_);
}

struct clamp_functor
{
  const float min_value;
  const float max_value;

  clamp_functor(float min_value_, float max_value_) restrict(amp,cpu): min_value(min_value_), max_value(max_value_) {}

  float operator()(const float& x) const restrict(amp,cpu)
  {
    if (x < min_value) {
      return min_value;
    }
    if (x > max_value) {
      return max_value;
    }
    return x;
  }
};

void THCudaTensor_clamp(THCudaTensor *self_, THCudaTensor *src, float min_value,
  float max_value)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 2, "sizes do not match");
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);
  bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> Dsrc_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), clamp_functor(min_value,max_value));
  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}


struct sign_functor
{
    float operator()(const float &v) const restrict(amp,cpu) {
    return (v > 0) - (v < 0);
  }
};


void THCudaTensor_sign(THCudaTensor *self_, THCudaTensor *src)
{
  THCudaTensor_resizeAs(self_, src);
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  src = THCudaTensor_newContiguous(src);
  bolt::amp::device_vector<float> Dself_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> Dsrc_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), sign_functor());
  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}

float THCudaTensor_meanall(THCudaTensor *self)
{
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THCudaTensor_sumall(self)/THCudaTensor_nElement(self);
}

void
THCudaTensor_mean(THCudaTensor *self, THCudaTensor *src, long dim)
{
  THCudaTensor_sum(self, src, dim);
  THCudaTensor_div(self, self, THCudaTensor_size(src, dim));
}

struct square_functor
{
  const float mean;

  square_functor(float mean_) restrict(amp,cpu): mean(mean_) {}

  float operator()(const float& x) const restrict(amp,cpu)
  {
    return (x-mean)*(x-mean);
  }
};

float THCudaTensor_varall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);

  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  float mean = THCudaTensor_meanall(self);

  bolt::amp::device_vector<float> diff(self_data.size());
  bolt::amp::transform(self_data.begin(), self_data.end(), diff.begin(),std::bind2nd(bolt::amp::minus<float>(), mean));
   
  float result = bolt::amp::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

  result = result/(THCudaTensor_nElement(self)-1);

  THCudaTensor_free(self);
  return result;
  return 0;
}

float THCudaTensor_stdall(THCudaTensor *self)
{
  return sqrt(THCudaTensor_varall(self));
  return 0;
}



template<class Op>
void THCudaTensor_logicalValue(THCudaTensor *self_, THCudaTensor *src, Op op)
{
  THCudaTensor_resizeAs(self_, src);
  
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  src = THCudaTensor_newContiguous(src);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> src_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  bolt::amp::transform(src_data.begin(), src_data.end(), self_data.begin(), op);
  THCudaTensor_free(src);
  THCudaTensor_freeCopyTo(self, self_);
}


struct partial_less_functor
{
  const float rhs;
  partial_less_functor(float rhs) restrict(amp,cpu): rhs(rhs) {}
  float operator()(const float &lhs) const restrict(amp,cpu) {return lhs < rhs;}
};


void THCudaTensor_ltValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_less_functor(value));
}


struct partial_greater_functor
{
  const float rhs;
  partial_greater_functor(float rhs) restrict(amp,cpu) : rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu) {return lhs > rhs;}
};


void THCudaTensor_gtValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_greater_functor(value));
}


struct partial_less_equal_functor
{
  const float rhs;
  partial_less_equal_functor(float rhs) restrict(amp,cpu): rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu) {return lhs <= rhs;}
};


void THCudaTensor_leValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_less_equal_functor(value));
}


struct partial_greater_equal_functor
{
  const float rhs;
  partial_greater_equal_functor(float rhs) restrict(amp,cpu) : rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu) {return lhs >= rhs;}
};


void THCudaTensor_geValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_greater_equal_functor(value));
}


struct partial_equal_functor
{
  const float rhs;
  partial_equal_functor(float rhs) restrict(amp,cpu): rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu){return lhs == rhs;}
};


void THCudaTensor_eqValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_equal_functor(value));
}


struct partial_not_equal_functor
{
  const float rhs;
  partial_not_equal_functor(float rhs) restrict(cpu,amp) : rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(cpu,amp) {return lhs != rhs;}
};


void THCudaTensor_neValue(THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(self_, src, partial_not_equal_functor(value));
}


template<class Op>
void THCudaTensor_logicalTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2, Op op)
{
  THCudaTensor_resizeAs(self_, src1);
  THArgCheck(THCudaTensor_nElement(src1) == THCudaTensor_nElement(src2), 3, "size do not match");

  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  src1 = THCudaTensor_newContiguous(src1);
  src2 = THCudaTensor_newContiguous(src2);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> src1_data(THCudaTensor_data(src1), THCudaTensor_data(src1)+THCudaTensor_nElement(src1));
  bolt::amp::device_vector<float> src2_data(THCudaTensor_data(src2), THCudaTensor_data(src2)+THCudaTensor_nElement(src2));
  bolt::amp::transform(src1_data.begin(), src1_data.end(), src2_data.begin(), self_data.begin(), op);

  THCudaTensor_free(src1);
  THCudaTensor_free(src2);
  THCudaTensor_freeCopyTo(self, self_);
}


void THCudaTensor_ltTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, bolt::amp::less<float>());
}


void THCudaTensor_gtTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, bolt::amp::greater<float>());
}


void THCudaTensor_leTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, bolt::amp::less_equal<float>());
}


void THCudaTensor_geTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, bolt::amp::greater_equal<float>());
}


void THCudaTensor_eqTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, bolt::amp::equal_to<float>());
}


void THCudaTensor_neTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(self_, src1, src2, bolt::amp::not_equal_to<float>());
}


struct norm_functor
{
  const float exponent;

  norm_functor(float exponent_)restrict(cpu,amp) : exponent(exponent_) {}

  float operator()(const float& x) const restrict(cpu,amp)
  {
    return Concurrency::fast_math::pow(Concurrency::fast_math::fabs(x), exponent);
  }
};


float THCudaTensor_normall(THCudaTensor *self, float value)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self),THCudaTensor_data(self)+THCudaTensor_nElement(self));

  float result;
  if(value == 0.0f) {
    result = bolt::amp::transform_reduce(self_data.begin(), self_data.begin()+size, partial_not_equal_functor(0.0f), (float)0, bolt::amp::plus<float>());

  } else {
    result = bolt::amp::transform_reduce(self_data.begin(), self_data.begin()+size, norm_functor(value), (float)0, bolt::amp::plus<float>());
    result = pow(result, (float)1.0/value);
  }

  THCudaTensor_free(self);
  return result;
}

void THCudaTensor_norm(THCudaTensor* self, THCudaTensor* src, float value, long dimension)
{
  if(value == 0.0f) {
    THCudaTensor_transformReduceDim(self, src, dimension, partial_not_equal_functor(0.0f), (float)0, bolt::amp::plus<float>());
  } else {
    THCudaTensor_transformReduceDim(self, src, dimension, norm_functor(value), (float)0, bolt::amp::plus<float>());
    THCudaTensor_pow(self, self, 1/value);
  }
}

void THCudaTensor_kernel_renorm(THCudaTensor *data, const float value, const long size, const float maxnorm, long gridSz)
{
    Concurrency::array_view<float,1> avData(THCudaTensor_nElement(data),THCudaTensor_data(data));
    //gridSz = (gridSz + 31 ) & ~31;
    Concurrency::extent<1> grdExt(gridSz);
    Concurrency::tiled_extent<32> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<32>tidx) restrict(amp)
    {
        tile_static float buffer[32];
        unsigned long tx = tidx.local[0];
        long bx = tidx.tile[0];
        long step = t_ext.tile_dim0;
        float* dat = avData.data();
        float *row = dat + size*bx;
        buffer[tx] = 0;
        // get norm of axis
        for (long i=tx; i<size; i+=step)
        {
            buffer[tx] += Concurrency::fast_math::pow(Concurrency::fast_math::fabs(row[i]), value);
        }
        // add (reduce)
        for (unsigned int stride = t_ext.tile_dim0 >> 1; stride > 0; stride >>= 1)
        {
            tidx.barrier.wait();
            if (tx < stride)
                buffer[tx] += buffer[tx+stride];
        }
        // clip norms
        tidx.barrier.wait();
        float norm = Concurrency::fast_math::pow(buffer[0], 1.0/value);
        if (norm > maxnorm)
        {
            norm = maxnorm / (norm + 1e-7);
            // renormalize
            for (long i=tx; i<size; i+=step)
            {
                row[i] *= norm;
            }
        }
    });
    avData.synchronize();
}

void THCudaTensor_renorm(THCudaTensor* self, THCudaTensor* src, float value, long dimension, float maxnorm)
{
  THCudaTensor *self_;
  THCudaTensor *src_ = THCudaTensor_newTranspose(src, dimension, 0);
  THCudaTensor *data = THCudaTensor_newClone(src_);
  long size = THCudaTensor_nElement(data)/data->size[0];
  
  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(src), 3, "invalid dimension");
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THCudaTensor_nDimension(src) > 1, 1, "need at least 2 dimensions");
  long gridSize = data->size[0] * 32;

  THCudaTensor_kernel_renorm(data, value, size, maxnorm, gridSize);

  THCudaTensor_free(src_);
  self_ = THCudaTensor_newTranspose(data, dimension, 0);
  THCudaTensor_resizeAs(self, self_);
  THCudaTensor_freeCopyTo(self_, self);
  THCudaTensor_free(data);
}

struct dist_functor
{
  const float exponent;

  dist_functor(float exponent_) restrict(amp,cpu) : exponent(exponent_) {}

  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::pow(Concurrency::fast_math::fabs(x-y), exponent);
  }
};

float THCudaTensor_dist(THCudaTensor *self, THCudaTensor *src, float value)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  src = THCudaTensor_newContiguous(src);
  bolt::amp::device_vector<float> self_data(THCudaTensor_data(self), THCudaTensor_data(self)+THCudaTensor_nElement(self));
  bolt::amp::device_vector<float> src_data(THCudaTensor_data(src), THCudaTensor_data(src)+THCudaTensor_nElement(src));
  float result = bolt::amp::inner_product(self_data.begin(), self_data.end(), src_data.begin(), (float) 0,bolt::amp::plus<float>(), dist_functor(value));

  THCudaTensor_free(src);
  THCudaTensor_free(self);
  
  return pow(result, (float)1.0/value);
}


void THCudaTensor_rand(THCudaRNGState* rng_state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_uniform(rng_state, r_, 0, 1);
}
void THCudaTensor_randn(THCudaRNGState* rng_state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_normal(rng_state, r_, 0, 1);
}

void THCudaTensor_kernel_indexFill(
   float *tensor, Concurrency::array<long>* stride, float *index, long src_nDim, 
   int dim, long idx_size, long tensor_size, long size_dim, float val, long nblockx
)
{
    Concurrency::array_view<float,1> srcTensor(Concurrency::extent<1>(tensor_size),tensor);
    Concurrency::array_view<long,1> srcStride(*stride);
    Concurrency::array_view<float,1> indx(Concurrency::extent<1>(idx_size),index);
    Concurrency::extent<2> gridExt(16,nblockx*16);
    Concurrency::tiled_extent<16,16> t_ext(gridExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16,16>tidx) restrict(amp)
    { 
        //int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        int thread_idx = tidx.tile[1] * t_ext.tile_dim1 * t_ext.tile_dim0 + tidx.local[0] * t_ext.tile_dim1 + tidx.local[1];

        long flat_size = tensor_size / idx_size; 
  
        if (thread_idx < flat_size)
        {
            long coeff = 0;
            for (int i=0; i<idx_size; i++)
            {
                int leftover = thread_idx;
                int srcIdx = 0;
                for (int d=0; d<src_nDim; d++)
                {
                    if (d < dim)
                    {
                        coeff = leftover / (srcStride[Concurrency::index<1>(d)] / size_dim);
                        leftover -= coeff * (srcStride[Concurrency::index<1>(d)] / size_dim);
                        srcIdx += coeff * srcStride[Concurrency::index<1>(d)];
                    }
                    else if (d > dim)
                    {
                        coeff = leftover / srcStride[Concurrency::index<1>(d)];
                        leftover -= coeff * srcStride[Concurrency::index<1>(d)];
                        srcIdx += coeff * srcStride[Concurrency::index<1>(d)];
                    } 
                }
                srcTensor[Concurrency::index<1>(srcIdx + (int)((indx[Concurrency::index<1>(i)])-1)*srcStride[Concurrency::index<1>(dim)])] = val;        
            }
        }
    });
    srcTensor.synchronize();
}	

void THCudaTensor_kernel_indexCopy(
   float *res, float *src, Concurrency::array<long,1>* res_stride, float *index, long res_size, 
   long res_nDim, int dim, long idx_size, long src_size, long size_dim, long nblockx)
{
    Concurrency::array_view<float,1> resTensor(Concurrency::extent<1>(res_size),res);
    Concurrency::array_view<float,1> srcTensor(Concurrency::extent<1>(src_size),src);
    Concurrency::array_view<long,1> resStride(*res_stride);
    Concurrency::array_view<float,1> indx(Concurrency::extent<1>(idx_size),index);

    Concurrency::extent<2> gridExt(16,nblockx*16);
    Concurrency::tiled_extent<16,16> t_ext(gridExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16,16>tidx) restrict(amp)
    { 
        //int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        int thread_idx = tidx.tile[1] * t_ext.tile_dim1 * t_ext.tile_dim0 + tidx.local[0] * t_ext.tile_dim1 + tidx.local[1];

        long flat_size = src_size / idx_size; 
  
        if (thread_idx < flat_size)
        {
            long coeff = 0;
            for (int i=0; i<idx_size; i++)
            {
                int leftover = thread_idx;
                int targetIdx = 0;
                int resIdx = 0;
                for (int d=0; d<res_nDim; d++)
                {
                    if (d < dim)
                    {
                        long stride_d = (resStride[Concurrency::index<1>(d)]) / size_dim;
                        coeff = leftover / stride_d;
                        leftover -= coeff * stride_d;
                        targetIdx += coeff * stride_d * idx_size;
                        resIdx += coeff * (resStride[Concurrency::index<1>(d)]);
                    }
                    else if (d > dim)
                    {
                        coeff = leftover / (resStride[Concurrency::index<1>(d)]);
                        leftover -= coeff * (resStride[Concurrency::index<1>(d)]);
                        targetIdx += coeff * (resStride[Concurrency::index<1>(d)]);
                        resIdx += coeff * (resStride[Concurrency::index<1>(d)]);
                    } 
                }
                resTensor[Concurrency::index<1>(resIdx + ((int)(indx[Concurrency::index<1>(i)])-1)*(resStride[Concurrency::index<1>(dim)]))] = srcTensor[Concurrency::index<1>(targetIdx +(int) i*(resStride[Concurrency::index<1>(dim)]))];
            }
        }
     });
     resTensor.synchronize();
}	

void THCudaTensor_indexCopy(THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src)
{
      THCudaTensor *indices_;
      Concurrency::array<long,1> *stride_;
      long nIndex = indices->size[0];
      long nRes;
  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THCudaTensor_newContiguous(src);
  indices_ = THCudaTensor_newWithSize1d(nIndex);
  THCudaTensor_copyLong(indices_, indices);

  nRes = THCudaTensor_nElement(res_);
  
      /*dim3 nthreads(16, 16);*/
      long nblockx = (long)(ceil((float)nRes / nIndex / (16*16)));
        stride_ =  new Concurrency::array<long,1>(Concurrency::extent<1>(res_->nDimension),res_->stride);

      THCudaTensor_kernel_indexCopy(
        THCudaTensor_data(res_), THCudaTensor_data(src), 
        stride_, THCudaTensor_data(indices_), nRes,
        res_->nDimension, dim, nIndex, 
        THCudaTensor_nElement(src), res_->size[dim],nblockx
      );

      delete stride_;
      THCudaStorage_free(indices_->storage);
}


void THCudaTensor_indexFill(THCudaTensor *res_, int dim, THLongTensor *indices, float val)
{
  THCudaTensor *indices_;
  Concurrency::array<long,1> *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < res_->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(res_->nDimension > 0, 2, "Source tensor is empty");
  
  indices_ = THCudaTensor_newWithSize1d(nIndex);
  THCudaTensor_copyLong(indices_, indices);
  
  nRes = THCudaTensor_nElement(res_) / res_->size[dim] * nIndex;
      long nblockx = (long)(ceil((float)nRes / nIndex / (16 * 16)));
      
      stride_ =  new Concurrency::array<long,1>(Concurrency::extent<1>(res_->nDimension),res_->stride);

  
      /*THCudaTensor_kernel_indexFill(
        THCudaTensor_data(res_), stride_, THCudaTensor_data(indices_), 
        res_->nDimension, dim, nIndex, nRes, res_->size[dim], val, nblockx
      );*/

     delete stride_;
     THCudaStorage_free(indices_->storage);
  

}

void THCudaTensor_kernel_indexSelect(
   float *tensor, float *src, Concurrency::array<long>* src_stride, float *index, 
   long src_nDim, int dim, long idx_size, long tensor_size, long src_size, long size_dim, long nblockx
)
{
    Concurrency::array_view<float,1> resTensor(Concurrency::extent<1>(tensor_size),tensor);
    Concurrency::array_view<float,1> srcTensor(Concurrency::extent<1>(src_size),src);
    Concurrency::array_view<long,1> srcStride(*src_stride);
    Concurrency::array_view<float,1> indx(Concurrency::extent<1>(idx_size),index);

    Concurrency::extent<2> gridExt(16,nblockx*16);
    Concurrency::tiled_extent<16,16> t_ext(gridExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16,16>tidx) restrict(amp)
    { 
        //int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        int thread_idx = tidx.tile[1] * t_ext.tile_dim1 * t_ext.tile_dim0 + tidx.local[0] * t_ext.tile_dim1 + tidx.local[1];

	    long flat_size = tensor_size / idx_size; 
  
	    if (thread_idx < flat_size)
	    {
		    long coeff = 0;
		    for (int i=0; i<idx_size; i++)
		    {
		        int leftover = thread_idx;
		        int targetIdx = 0;
		        int srcIdx = 0;
		        for (int d=0; d<src_nDim; d++)
		        {
			        if (d < dim)
			        {
			            long stride_d = srcStride[Concurrency::index<1>(d)] / size_dim;
			            coeff = leftover / stride_d;
			            leftover -= coeff * stride_d;
			            targetIdx += coeff * stride_d * idx_size;
			            srcIdx += coeff * srcStride[Concurrency::index<1>(d)];
			        }
			        else if (d > dim)
			        {
			            coeff = leftover / srcStride[Concurrency::index<1>(d)];
			            leftover -= coeff * srcStride[Concurrency::index<1>(d)];
			            targetIdx += coeff *srcStride[Concurrency::index<1>(d)];
			            srcIdx += coeff * srcStride[Concurrency::index<1>(d)];
			        } 
		        }
		        resTensor[Concurrency::index<1>(targetIdx + i*srcStride[Concurrency::index<1>(dim)])] = srcTensor[Concurrency::index<1>(srcIdx + ((int)(indx[Concurrency::index<1>(i)])-1)*srcStride[Concurrency::index<1>(dim)])];
		    }
	    }
	});
	resTensor.synchronize();
}
	


void THCudaTensor_indexSelect(THCudaTensor *res_, THCudaTensor *src, int dim, THLongTensor *indices)
{
  THLongStorage *newSize;
  THCudaTensor *indices_;
  Concurrency::array<long> *stride_;
  long nIndex = indices->size[0];
  long nRes;
  
  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  
  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize, src->size);
  newSize->data[dim] = nIndex;
  THCudaTensor_resize(res_, newSize, NULL);
  THLongStorage_free(newSize);
  
  indices_ = THCudaTensor_newWithSize1d(nIndex);
  THCudaTensor_copyLong(indices_, indices);

  nRes = THCudaTensor_nElement(res_);
      long nblockx = (long)(ceil((float)nRes / nIndex / (16*16)));
  
      stride_ =  new Concurrency::array<long,1>(Concurrency::extent<1>(res_->nDimension),res_->stride);
  
      THCudaTensor_kernel_indexSelect(
        THCudaTensor_data(res_), THCudaTensor_data(src), 
        stride_, THCudaTensor_data(indices_), 
        src->nDimension, dim, indices->size[0], nRes,THCudaTensor_nElement(src), src->size[dim],nblockx
      );
    
      //THCudaCheck(cudaFree(stride_));
      //THCudaTensor_free(indices_);
      delete stride_;
      THCudaStorage_free(indices_->storage);
      THCudaTensor_free(indices_);
      THLongStorage_free(newSize);
}
