#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"
#include "amp_math.h"
#include "THBlas.h"
#include "THCBlas.h"
#include<algorithm>
#include<utility>
#include<numeric>
#include "common.h"


#define NB_THREADS_PER_BLOCK 256


void THGPUTensor_fill(THGPUTensor *self_, float value)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  THGPUStorage_fill(self->storage,value);
  THGPUTensor_copy(self_, self);
  if (self != self_)
  {
    THGPUStorage_free(self->storage);
    THGPUTensor_free(self);
  }
}

void THGPUTensor_zero(THGPUTensor *self_)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  THGPUStorage_fill(self->storage,0);
  THGPUTensor_copy(self_, self);
  if (self != self_)
  {
    THGPUStorage_free(self->storage);
    THGPUTensor_free(self);
  }
}


void THGPUTensor_zeros(THGPUTensor *r_, THLongStorage *size)
{
  THGPUTensor_resize(r_, size, NULL);
  THGPUTensor_zero(r_);
}

void THGPUTensor_ones(THGPUTensor *r_, THLongStorage *size)
{
  THGPUTensor_resize(r_, size, NULL);
  THGPUTensor_fill(r_, 1);
}

void THGPUTensor_reshape(THGPUTensor *r_, THGPUTensor *t, THLongStorage *size)
{
  THGPUTensor_resize(r_, size, NULL);
  THGPUTensor_copy(r_, t);
}

long THGPUTensor_numel(THGPUTensor *t)
{
  return THGPUTensor_nElement(t);
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

void THGPUTensor_add(THGPUTensor *self_, THGPUTensor *src_, float value)
{
  THGPUTensor_resizeAs(self_, src_);
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  THGPUTensor *src = THGPUTensor_newContiguous(src_);
  DECLARE_BOLT_DEVICE_VECTOR_2(self, Dself_data, src, Dsrc_data);
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), addvalue_functor(value));
  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
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

void THGPUTensor_mul(THGPUTensor *self_, THGPUTensor *src_, float value)
{
  THGPUTensor_resizeAs(self_, src_);
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  THGPUTensor *src = THGPUTensor_newContiguous(src_);
  DECLARE_BOLT_DEVICE_VECTOR_2(src, Dsrc_data, self, Dself_data);
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), mulvalue_functor(value));
  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
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

void THGPUTensor_div(THGPUTensor *self_, THGPUTensor *src_, float value)
{
  THGPUTensor_resizeAs(self_, src_);
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  THGPUTensor *src = THGPUTensor_newContiguous(src_);
  DECLARE_BOLT_DEVICE_VECTOR_2(src, Dsrc_data, self, Dself_data);
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), divvalue_functor(value));
  long size = THGPUTensor_nElement(self);
  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
}


void THGPUTensor_cadd(THGPUTensor *self_, THGPUTensor* src1, float value, THGPUTensor *src2)
{
  THGPUTensor_resizeAs(self_, src1);
  THArgCheck(THGPUTensor_nElement(src1) == THGPUTensor_nElement(src2), 3, "size do not match");
  {
    THGPUTensor *self = THGPUTensor_newContiguous(self_);

    if (self_ != src1)
    {
      src1 = THGPUTensor_newContiguous(src1);
      THGPUTensor_copy(self, src1);
      THGPUTensor_free(src1);
    }

    src2 = THGPUTensor_newContiguous(src2);

    THGPUBlas_axpy(THGPUTensor_nElement(self), value, THGPUTensor_data(src2), 1, THGPUTensor_data(self), 1);

    THGPUTensor_free(src2);
    THGPUTensor_freeCopyTo(self, self_);
  }
}

void THGPUTensor_cmul(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_resizeAs(self_, src1);
  THArgCheck(THGPUTensor_nElement(src1) == THGPUTensor_nElement(src2), 3, "size do not match");
  {
    THGPUTensor *self = THGPUTensor_newContiguous(self_);
    long size = THGPUTensor_nElement(self);
    src1 = THGPUTensor_newContiguous(src1);
    src2 = THGPUTensor_newContiguous(src2);
    DECLARE_BOLT_DEVICE_VECTOR_3(src1, Dsrc1_data, src2, Dsrc2_data, self, Dself_data);
    bolt::amp::transform(Dsrc2_data.begin(), Dsrc2_data.end(), Dsrc1_data.begin(), Dself_data.begin(), bolt::amp::multiplies<float>());

    THGPUTensor_free(src1);
    THGPUTensor_free(src2);
    THGPUTensor_freeCopyTo(self, self_);
  }
}

void THGPUTensor_cdiv(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_resizeAs(self_, src1);
  THArgCheck(THGPUTensor_nElement(src1) == THGPUTensor_nElement(src2), 3, "size do not match");
  {
    THGPUTensor *self = THGPUTensor_newContiguous(self_);
    long size = THGPUTensor_nElement(self);
    src1 = THGPUTensor_newContiguous(src1);
    src2 = THGPUTensor_newContiguous(src2);
    DECLARE_BOLT_DEVICE_VECTOR_3(src1, Dsrc1_data, src2, Dsrc2_data, self, Dself_data);
    bolt::amp::transform(Dsrc1_data.begin(), Dsrc1_data.end(), Dsrc2_data.begin(), Dself_data.begin(), bolt::amp::divides<float>());


    THGPUTensor_free(src1);
    THGPUTensor_free(src2);
    THGPUTensor_freeCopyTo(self, self_);
  }
  
}

void THGPUTensor_kernel_addcmul( Concurrency::array_view<float,1> &Data, float value, Concurrency::array_view<float,1>&src1Data, Concurrency::array_view<float,1>&src2Data, long size, const int nThreadPerBlock, int nBlockPerRow, int nBlockPerColumn)
{
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
}


void THGPUTensor_addcmul(THGPUTensor *self_, THGPUTensor* t, float value, THGPUTensor *src1, THGPUTensor *src2)
{
  if(self_ != t)
  {
    THGPUTensor_resizeAs(self_, t);
    THGPUTensor_copy(self_, t);
  }

  THGPUTensor_resizeAs(self_, src1);
  THArgCheck(THGPUTensor_nElement(src1) == THGPUTensor_nElement(src2), 3, "size do not match");

  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  THGPUTensor *temp1 = src1;
  THGPUTensor *temp2 = src2;
  long size = THGPUTensor_nElement(self);
  src1 = THGPUTensor_newContiguous(src1);
  src2 = THGPUTensor_newContiguous(src2);

  int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
  THGPUGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);

  Concurrency::array_view<float,1> *pavData = static_cast<Concurrency::array_view<float,1> *>(self->storage->allocatorContext);
  Concurrency::array_view<float,1> *pavSrc1 = static_cast<Concurrency::array_view<float,1> *>(src1->storage->allocatorContext);
  Concurrency::array_view<float,1> *pavSrc2 = static_cast<Concurrency::array_view<float,1> *>(src2->storage->allocatorContext);

  THGPUTensor_kernel_addcmul(*pavData, value, *pavSrc1, *pavSrc2, size, nThreadPerBlock, nBlockPerRow,nBlockPerColumn);

  THGPUTensor_copy(self_, self);
  if (src1 != temp1)
  {
    THGPUStorage_free(src1->storage);
    THGPUTensor_free(src1);
  }
  if (src2 != temp2)
  {
    THGPUStorage_free(src2->storage);
    THGPUTensor_free(src2);
  }
  if (self != self_)
  {
    THGPUStorage_free(self->storage);
    THGPUTensor_free(self);
  }
}

void THGPUTensor_kernel_addcdiv(Concurrency::array_view<float, 1> &Data, float value, Concurrency::array_view<float, 1> &src1Data, Concurrency::array_view<float, 1> &src2Data, long size, const int nThreadPerBlock, int nBlockPerRow, int nBlockPerColumn)
{
  const int nthreads = 256;
  nBlockPerRow = (nBlockPerRow + (nthreads -1)) & ~(nthreads -1);
  Concurrency::extent<2> gridExt(nBlockPerColumn,nBlockPerRow);
  Concurrency::tiled_extent<1,nthreads> t_ext(gridExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,nthreads>tidx) restrict(amp)
  {
    long k = (((tidx.tile[0] * t_ext[1]/t_ext.tile_dim1) + tidx.tile[1]) * t_ext.tile_dim1) + tidx.local[1];
    if(k < size)
    {
      Data[Concurrency::index<1>(k)] += (float) value * (src1Data[k] / src2Data[k]); 
    }

  });
}


void THGPUTensor_addcdiv(THGPUTensor *self_, THGPUTensor *t, float value, THGPUTensor *src1, THGPUTensor *src2)
{
  if(self_ != t)
  {
    THGPUTensor_resizeAs(self_, t);
    THGPUTensor_copy(self_, t);
  }
  THGPUTensor_resizeAs(self_, src1);
  THArgCheck(THGPUTensor_nElement(src1) == THGPUTensor_nElement(src2), 3, "size do not match");
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  THGPUTensor *temp1 = src1;
  THGPUTensor *temp2 = src2;
  long size = THGPUTensor_nElement(self);
  src1 = THGPUTensor_newContiguous(src1);
  src2 = THGPUTensor_newContiguous(src2);

  int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
  THGPUGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);

  Concurrency::array_view<float, 1> *pavData = static_cast<Concurrency::array_view<float, 1> *>(self->storage->allocatorContext);
  Concurrency::array_view<float, 1> *pavSrc1 = static_cast<Concurrency::array_view<float, 1> *>(src1->storage->allocatorContext);
  Concurrency::array_view<float, 1> *pavSrc2 = static_cast<Concurrency::array_view<float, 1> *>(src2->storage->allocatorContext);

  THGPUTensor_kernel_addcdiv(*pavData, value, *pavSrc1, *pavSrc2, size, nThreadPerBlock, nBlockPerRow,nBlockPerColumn);

  THGPUTensor_copy(self_, self);
  if (src1 != temp1)
  {
    THGPUStorage_free(src1->storage);
    THGPUTensor_free(src1);
  }
  if (src2 != temp2)
  {
    THGPUStorage_free(src2->storage);
    THGPUTensor_free(src2);
  }
  if (self != self_)
  {
    THGPUStorage_free(self->storage);
    THGPUTensor_free(self);
  }
}

float THGPUTensor_dot(THGPUTensor *self, THGPUTensor *src)
{
  THArgCheck(THGPUTensor_nElement(self) == THGPUTensor_nElement(src), 2, "size do not match");
  {
    self = THGPUTensor_newContiguous(self);
    src = THGPUTensor_newContiguous(src);
    float result = THGPUBlas_dot(THGPUTensor_nElement(self),
                                  THGPUTensor_data(self), 1,
                                  THGPUTensor_data(src), 1);
    THGPUTensor_free(src);
    THGPUTensor_free(self);
    return result;
  }
}

float THGPUTensor_minall(THGPUTensor *self)
{
  self = THGPUTensor_newContiguous(self);
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(THInf), bolt::amp::minimum<float>());
  THGPUTensor_free(self);
  return result;
}

float THGPUTensor_maxall(THGPUTensor *self)
{
  self = THGPUTensor_newContiguous(self);
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(-THInf), bolt::amp::maximum<float>());
  THGPUTensor_free(self);
  return result;
}



float THGPUTensor_sumall(THGPUTensor *self)
{
  self = THGPUTensor_newContiguous(self);
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(0), bolt::amp::plus<float>());
  THGPUTensor_free(self);
  return result;
}

float THGPUTensor_prodall(THGPUTensor *self)
{
  self = THGPUTensor_newContiguous(self);
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  float result = bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(1), bolt::amp::multiplies<float>());
  THGPUTensor_free(self);
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
void THGPUTensor_kernel_transformReduceOuterDim(Concurrency::array_view<float, 1> &avTgt, Concurrency::array_view<float, 1> &avSrc,
                                                unsigned int tgtSz, unsigned int srcSz,
                                                Concurrency::array_view<unsigned int, 1> &avSrc_stride, Concurrency::array_view<unsigned int, 1> &avTgt_stride,
                                                Concurrency::array_view<unsigned int, 1> &avSize, UnaryFunction unary_op,
                                                float init, BinaryFunction binary_op,
                                                unsigned int gridConf[])
{
  const size_t reduce = 3;
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
}

template<class BinaryFunction, class UnaryFunction>
void THGPUTensor_transformReduceOuterDim(THGPUTensor *tgt, THGPUTensor *src, long rdim,
                                         UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  const size_t reduce = 3;
  unsigned int src_stride[4] = { 0, 0, 0, 0 };
  unsigned int tgt_stride[4] = { 0, 0, 0, 0 };
  unsigned int size[4] = { 1, 1, 1, 1 };
  unsigned int gridConfig[3];
  unsigned ndim = THGPUTensor_nDimension(src);
  for (unsigned idim = 0, o = ndim - 2; idim < ndim; idim++) 
  {
    unsigned odim = idim == rdim ? reduce : o--;
    src_stride[odim] = THGPUTensor_stride(src, idim);
    tgt_stride[odim] = THGPUTensor_stride(tgt, idim);
    size[odim] = THGPUTensor_size(src, idim);
  }
  const unsigned nThreadPerBlock = 256;
  unsigned nBlockPerColumn = (size[0] + nThreadPerBlock - 1) / nThreadPerBlock;
  //dim3 threads(nThreadPerBlock);
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  gridConfig[0] = Concurrency::fast_math::fmin(maxGridDim, nBlockPerColumn);
  gridConfig[1] = Concurrency::fast_math::fmin(maxGridDim, size[1]);
  gridConfig[2] = Concurrency::fast_math::fmin(maxGridDim, size[2]);

  Concurrency::array_view<float, 1> *pavTgt = static_cast<Concurrency::array_view<float, 1> *>(tgt->storage->allocatorContext);
  Concurrency::array_view<float, 1> *pavSrc = static_cast<Concurrency::array_view<float, 1> *>(src->storage->allocatorContext);
  Concurrency::array_view<unsigned int, 1> avSrc_stride(4, src_stride);
  Concurrency::array_view<unsigned int, 1> avTgt_stride(4, tgt_stride);
  Concurrency::array_view<unsigned int, 1> avSize(4, size);


  THGPUTensor_kernel_transformReduceOuterDim(*pavTgt,
                                             *pavSrc, THGPUTensor_nElement(src), THGPUTensor_nElement(tgt),
                                             avSrc_stride, avTgt_stride, avSize, unary_op, init, binary_op,gridConfig);
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
void THGPUTensor_kernel_transformReduceInnermostDim(Concurrency::array_view<float, 1> &avTgt, Concurrency::array_view<float, 1> &avSrc, unsigned int tgtSz,
                                                    unsigned int srcSz, Concurrency::array_view<unsigned int, 1> &avSrc_stride, Concurrency::array_view<unsigned int, 1> &avTgt_stride,
                                                    Concurrency::array_view<unsigned int, 1> &avSize,
                                                    UnaryFunction unary_op, float init,
                                                    BinaryFunction binary_op, unsigned int gridConf[])
{
  Concurrency::extent<3> grdExt(gridConf[2], gridConf[1] * 8, gridConf[0] *32);
  Concurrency::tiled_extent<1, 8, 32> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, 8, 32> tidx) restrict(amp)
  {
    tile_static float sbuf[16][32]; // 8kB
    for (unsigned z = tidx.tile[0]; z < avSize[3] ; z += t_ext[0]/tidx.tile_dim0)
    {
      for (unsigned x = tidx.tile[2]; x < avSize[2] ; x += t_ext[2]/tidx.tile_dim2)
      {
        for (unsigned bRow = tidx.tile[1] * tidx.tile_dim1; bRow < avSize[1]; bRow += t_ext[1]) 
        {
          float acc = init;
          unsigned row = bRow + tidx.local[1];
          //float *src = src_ + z * src_stride[3] + x * src_stride[2] + row * src_stride[1];
          bool reducing = tidx.local[2] < t_ext.tile_dim1 && bRow + tidx.local[2] < avSize[1] && tidx.local[1] == 0;
          for (unsigned bCol = 0; bCol < avSize[0]; bCol += t_ext.tile_dim2) 
          {
            sbuf[tidx.local[1]][tidx.local[2]] = init;
            unsigned col = bCol + tidx.local[2];
            if (row < avSize[1] && col < avSize[0]) 
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
}

template<class UnaryFunction, class BinaryFunction>
void THGPUTensor_transformReduceInnermostDim(THGPUTensor *tgt, THGPUTensor *src, 
                                             UnaryFunction unary_op, float init, BinaryFunction binary_op)
{
  unsigned int src_stride[4] = { 0, 0, 0, 0 };
  unsigned int tgt_stride[4] = { 0, 0, 0, 0 };
  unsigned int size[4] = { 1, 1, 1, 1 };
  unsigned int gridConfig[3];
  unsigned ndim = THGPUTensor_nDimension(src);
  for (unsigned dim = 0; dim < ndim; dim++) 
  {
    unsigned odim = ndim - 1 - dim;
    src_stride[odim] = THGPUTensor_stride(src, dim);
    tgt_stride[odim] = THGPUTensor_stride(tgt, dim);
    size[odim] = THGPUTensor_size(src, dim);
  }
  //std::cout<<"InnerDim kernel"<<std::endl;
  unsigned nBlockPerRow = (size[1] + 16 - 1) / 16;
  unsigned maxGridDim = 1024; // anything < 64k is fine. The choice has no impact on performance.
  gridConfig[0]= std::min(maxGridDim, size[2]);
  gridConfig[1]= std::min(maxGridDim, nBlockPerRow);
  gridConfig[2] = std::min(maxGridDim, size[3]);

  Concurrency::array_view<float, 1> *pavTgt = static_cast<Concurrency::array_view<float, 1> *>(tgt->storage->allocatorContext);
  Concurrency::array_view<float, 1> *pavSrc = static_cast<Concurrency::array_view<float, 1> *>(src->storage->allocatorContext);  
  Concurrency::array_view<unsigned int, 1> avSrc_stride(4, src_stride);
  Concurrency::array_view<unsigned int, 1> avTgt_stride(4, tgt_stride);
  Concurrency::array_view<unsigned int, 1> avSize(4, size);
  
  THGPUTensor_kernel_transformReduceInnermostDim(*pavTgt, *pavSrc,
                                                 THGPUTensor_nElement(tgt), THGPUTensor_nElement(src),
                                                 avSrc_stride, avTgt_stride, avSize, unary_op, init,
                                                 binary_op, gridConfig);
}



template<class UnaryFunction, class BinaryFunction>
void THGPUTensor_transformReduceDim(THGPUTensor *self_, THGPUTensor *src,
                                    long dimension, UnaryFunction unary_op,
                                    float init, BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 && dimension < THGPUTensor_nDimension(src), 3, "dimension out of range");
  THArgCheck(THGPUTensor_nDimension(src) <= 4, 2, "too many dimensions (>4)");

  THLongStorage *dim = THGPUTensor_newSizeOf(src);
  THLongStorage_set(dim, dimension, 1);
  THGPUTensor_resize(self_, dim, NULL);
  THLongStorage_free(dim);

  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  src = THGPUTensor_newContiguous(src);

  if (dimension == THGPUTensor_nDimension(src)-1)
  {
    THGPUTensor_transformReduceInnermostDim(self, src, unary_op, init, binary_op);
  } else
  {
    THGPUTensor_transformReduceOuterDim(self, src, dimension, unary_op, init, binary_op);
  }

  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
}



template<class BinaryFunction>
void THGPUTensor_reduceDim(THGPUTensor *self_, THGPUTensor *src, long dimension, float init, BinaryFunction binary_op)
{
  THGPUTensor_transformReduceDim(self_, src, dimension, bolt::amp::identity<float>(), init, binary_op);
}


void THGPUTensor_sum(THGPUTensor *self, THGPUTensor *src, long dimension)
{
  return THGPUTensor_reduceDim(self, src, dimension, 0.0f, bolt::amp::plus<float>());
}

void THGPUTensor_prod(THGPUTensor *self, THGPUTensor *src, long dimension)
{
  return THGPUTensor_reduceDim(self, src, dimension, 1.0f, bolt::amp::multiplies<float>());
}

void THGPUTensor_max(THGPUTensor *self, THGPUTensor *indices, THGPUTensor *src, long dimension)
{
  const float minfloat32 = -3.402823466e+38f;
  return THGPUTensor_reduceDim(self, src, dimension, minfloat32, bolt::amp::maximum<float>());
}


void THGPUTensor_min(THGPUTensor *self, THGPUTensor* indices, THGPUTensor *src, long dimension)
{
  const float maxfloat32 = 3.402823466e+38f;
  return THGPUTensor_reduceDim(self, src, dimension, maxfloat32, bolt::amp::minimum<float>());
}

void THGPUTensor_addmv(THGPUTensor *r_, float beta, THGPUTensor *t, float alpha, THGPUTensor *mat, THGPUTensor *vec)
{
  if ( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if ( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if (t->nDimension != 1)
    THError("size mismatch");

  if (t->size[0] != mat->size[0])
    THError("size mismatch");

  if (r_ != t)
  {
    THGPUTensor_resizeAs(r_, t);
    THGPUTensor_copy(r_, t);
  }

  if (mat->stride[0] == 1)
  {
    THGPUBlas_gemv('n', mat->size[0], mat->size[1],
                     alpha, THGPUTensor_data(mat), mat->stride[1],
                     THGPUTensor_data(vec), vec->stride[0],
                     beta, THGPUTensor_data(r_), r_->stride[0]);
  }
  else if (mat->stride[1] == 1)
  {
    THGPUBlas_gemv('t',  mat->size[1], mat->size[0],
                     alpha, THGPUTensor_data(mat), mat->stride[0],
                     THGPUTensor_data(vec), vec->stride[0],
                     beta, THGPUTensor_data(r_), r_->stride[0]);
  }
  else
  {
    THGPUTensor *cmat = THGPUTensor_newContiguous(mat);

    THGPUBlas_gemv('t',  mat->size[1], mat->size[0],
                     alpha, THGPUTensor_data(cmat), cmat->stride[0],
                     THGPUTensor_data(vec), vec->stride[0],
                     beta, THGPUTensor_data(r_), r_->stride[0]);

    THGPUTensor_free(cmat);
  }
}

void THGPUTensor_addmm(THGPUTensor *r_, float beta, THGPUTensor *t, float alpha, THGPUTensor *m1, THGPUTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THGPUTensor *r__, *m1_, *m2_;

  if ( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if (t->nDimension != 2)
    THError("size mismatch");

  if ( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if (t != r_)
  {
    THGPUTensor_resizeAs(r_, t);
    THGPUTensor_copy(r_, t);
  }

  /* r_ */
  if (r_->stride[0] == 1)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if (r_->stride[1] == 1)
  {
    THGPUTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    r__ = THGPUTensor_newWithSize2d(r_->size[1], r_->size[0]);
    THGPUTensor_copy(r__, r_);
    THGPUTensor_transpose(r__, NULL, 0, 1);
  }

  /* m1 */
  if (m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if (m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THGPUTensor_newContiguous(m1);
  }

  /* m2 */
  if (m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if (m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THGPUTensor_newContiguous(m2);
  }

  /* do the operation */
  THGPUBlas_gemm(transpose_m1,
                   transpose_m2,
                   r__->size[(transpose_r == 'n' ? 0 : 1)],
                   r__->size[(transpose_r == 'n' ? 1 : 0)],
                   m1_->size[(transpose_r == 'n' ? 1 : 0)],
                   alpha,
                   THGPUTensor_data(m1_),
                   (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   THGPUTensor_data(m2_),
                   (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                   beta,
                   THGPUTensor_data(r__),
                   r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if (m1_ != m1)
    THGPUTensor_free(m1_);

  if (m2_ != m2)
    THGPUTensor_free(m2_);

  if (r__ != r_)
    THGPUTensor_freeCopyTo(r__, r_);
}

void THGPUTensor_addr(THGPUTensor *r_, float beta, THGPUTensor *t, float alpha, THGPUTensor *vec1, THGPUTensor *vec2)
{
  if ( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if (t->nDimension != 2)
    THError("size mismatch");

  if ( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if (r_ != t)
  {
    THGPUTensor_resizeAs(r_, t);
    THGPUTensor_copy(r_, t);
  }

  if (beta != 1)
    THGPUTensor_mul(r_, r_, beta);

  if (r_->stride[0] == 1)
  {
    THGPUBlas_ger(vec1->size[0], vec2->size[0],
                    alpha, THGPUTensor_data(vec1), vec1->stride[0],
                    THGPUTensor_data(vec2), vec2->stride[0],
                    THGPUTensor_data(r_), r_->stride[1]);
  }
  else if (r_->stride[1] == 1)
  {
    THGPUBlas_ger(vec2->size[0], vec1->size[0],
                    alpha, THGPUTensor_data(vec2), vec2->stride[0],
                    THGPUTensor_data(vec1), vec1->stride[0],
                    THGPUTensor_data(r_), r_->stride[0]);
  }
  else
  {
    THGPUTensor *cr = THGPUTensor_newClone(r_);

    THGPUBlas_ger(vec2->size[0], vec1->size[0],
                    alpha, THGPUTensor_data(vec2), vec2->stride[0],
                    THGPUTensor_data(vec1), vec1->stride[0],
                    THGPUTensor_data(cr), cr->stride[0]);

    THGPUTensor_freeCopyTo(cr, r_);
  }
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                                                                   \
struct NAME##_functor                                                                                                   \
{                                                                                                                       \
  float operator()(const float& x) const                                                                                \
  {                                                                                                                     \
    return CFUNC(x);                                                                                                    \
  }                                                                                                                     \
};                                                                                                                      \
                                                                                                                        \
void THGPUTensor_##NAME(THGPUTensor *self_, THGPUTensor *src)                                                           \
{                                                                                                                       \
  THGPUTensor_resizeAs(self_, src);                                                                                     \
  THGPUTensor *self = THGPUTensor_newContiguous(self_);                                                                 \
  src = THGPUTensor_newContiguous(src);                                                                                 \
  long size = THGPUTensor_nElement(self);                                                                               \
                                                                                                                        \
  DECLARE_BOLT_DEVICE_VECTOR_2(src, src_data, self, self_data); \
  bolt::amp::transform(src_data.begin(), src_data.end(), self_data.begin(),NAME##_functor());                           \
                                                                                                                        \
  THGPUTensor_free(src);                                                                                                \
  THGPUTensor_freeCopyTo(self, self_);                                                                                  \
}

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, Concurrency::fast_math::log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, Concurrency::precise_math::log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, Concurrency::fast_math::exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, Concurrency::fast_math::cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, Concurrency::fast_math::acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, Concurrency::fast_math::cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, Concurrency::fast_math::sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, Concurrency::fast_math::asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, Concurrency::fast_math::sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, Concurrency::fast_math::tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, Concurrency::fast_math::atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, Concurrency::fast_math::tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, Concurrency::fast_math::sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, Concurrency::fast_math::ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, Concurrency::fast_math::floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, Concurrency::fast_math::fabs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, Concurrency::fast_math::roundf)

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

void THGPUTensor_pow(THGPUTensor *self_, THGPUTensor *src, float value)
{
  THGPUTensor_resizeAs(self_, src);
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  src = THGPUTensor_newContiguous(src);
  DECLARE_BOLT_DEVICE_VECTOR_2(src, Dsrc_data, self, Dself_data);
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), pow_functor(value));
  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
}

struct atan2_functor
{
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::atan2f(x, y);
  }
};
void THGPUTensor_atan2(THGPUTensor *self_, THGPUTensor *tx, THGPUTensor *ty)
{
  THGPUTensor_resizeAs(self_, tx);
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  tx = THGPUTensor_newContiguous(tx);
  ty = THGPUTensor_newContiguous(ty);
  DECLARE_BOLT_DEVICE_VECTOR_3(tx, tx_data, ty, ty_data, self, self_data);
  bolt::amp::transform(tx_data.begin(), tx_data.end(), ty_data.begin(), self_data.begin(), atan2_functor());

  THGPUTensor_free(tx);
  THGPUTensor_free(ty);
  THGPUTensor_freeCopyTo(self, self_);
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

void THGPUTensor_clamp(THGPUTensor *self_, THGPUTensor *src, float min_value,
                       float max_value)
{
  THArgCheck(THGPUTensor_nElement(self_) == THGPUTensor_nElement(src), 2, "sizes do not match");
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  src = THGPUTensor_newContiguous(src);
  DECLARE_BOLT_DEVICE_VECTOR_2(src, Dsrc_data, self, Dself_data);
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), clamp_functor(min_value,max_value));
  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
}


struct sign_functor
{
  float operator()(const float &v) const restrict(amp,cpu) {
    return (v > 0) - (v < 0);
  }
};


void THGPUTensor_sign(THGPUTensor *self_, THGPUTensor *src)
{
  THGPUTensor_resizeAs(self_, src);
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  src = THGPUTensor_newContiguous(src);
  DECLARE_BOLT_DEVICE_VECTOR_2(src, Dsrc_data, self, Dself_data);
  bolt::amp::transform(Dsrc_data.begin(), Dsrc_data.end(), Dself_data.begin(), sign_functor());
  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
}

float THGPUTensor_meanall(THGPUTensor *self)
{
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THGPUTensor_sumall(self)/THGPUTensor_nElement(self);
}

void
THGPUTensor_mean(THGPUTensor *self, THGPUTensor *src, long dim)
{
  THGPUTensor_sum(self, src, dim);
  THGPUTensor_div(self, self, THGPUTensor_size(src, dim));
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

float THGPUTensor_varall(THGPUTensor *self)
{
  self = THGPUTensor_newContiguous(self);
  long size = THGPUTensor_nElement(self);
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  float mean = THGPUTensor_meanall(self);

  bolt::amp::device_vector<float> diff(self_data.size());
  bolt::amp::transform(self_data.begin(), self_data.end(), diff.begin(),std::bind2nd(bolt::amp::minus<float>(), mean));
   
  float result = bolt::amp::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

  result = result/(THGPUTensor_nElement(self)-1);

  THGPUTensor_free(self);
  return result;
  return 0;
}

float THGPUTensor_stdall(THGPUTensor *self)
{
  return sqrt(THGPUTensor_varall(self));
  return 0;
}



template<class Op>
void THGPUTensor_logicalValue(THGPUTensor *self_, THGPUTensor *src, Op op)
{
  THGPUTensor_resizeAs(self_, src);
  
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  src = THGPUTensor_newContiguous(src);
  DECLARE_BOLT_DEVICE_VECTOR_2(src, src_data, self, self_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), self_data.begin(), op);
  THGPUTensor_free(src);
  THGPUTensor_freeCopyTo(self, self_);
}


struct partial_less_functor
{
  const float rhs;
  partial_less_functor(float rhs) restrict(amp,cpu): rhs(rhs) {}
  float operator()(const float &lhs) const restrict(amp,cpu) {return lhs < rhs;}
};


void THGPUTensor_ltValue(THGPUTensor *self_, THGPUTensor *src, float value)
{
  THGPUTensor_logicalValue(self_, src, partial_less_functor(value));
}


struct partial_greater_functor
{
  const float rhs;
  partial_greater_functor(float rhs) restrict(amp,cpu) : rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu) {return lhs > rhs;}
};


void THGPUTensor_gtValue(THGPUTensor *self_, THGPUTensor *src, float value)
{
  THGPUTensor_logicalValue(self_, src, partial_greater_functor(value));
}


struct partial_less_equal_functor
{
  const float rhs;
  partial_less_equal_functor(float rhs) restrict(amp,cpu): rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu) {return lhs <= rhs;}
};


void THGPUTensor_leValue(THGPUTensor *self_, THGPUTensor *src, float value)
{
  THGPUTensor_logicalValue(self_, src, partial_less_equal_functor(value));
}


struct partial_greater_equal_functor
{
  const float rhs;
  partial_greater_equal_functor(float rhs) restrict(amp,cpu) : rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu) {return lhs >= rhs;}
};


void THGPUTensor_geValue(THGPUTensor *self_, THGPUTensor *src, float value)
{
  THGPUTensor_logicalValue(self_, src, partial_greater_equal_functor(value));
}


struct partial_equal_functor
{
  const float rhs;
  partial_equal_functor(float rhs) restrict(amp,cpu): rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(amp,cpu){return lhs == rhs;}
};


void THGPUTensor_eqValue(THGPUTensor *self_, THGPUTensor *src, float value)
{
  THGPUTensor_logicalValue(self_, src, partial_equal_functor(value));
}


struct partial_not_equal_functor
{
  const float rhs;
  partial_not_equal_functor(float rhs) restrict(cpu,amp) : rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(cpu,amp) {return lhs != rhs;}
};


void THGPUTensor_neValue(THGPUTensor *self_, THGPUTensor *src, float value)
{
  THGPUTensor_logicalValue(self_, src, partial_not_equal_functor(value));
}


template<class Op>
void THGPUTensor_logicalTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2, Op op)
{
  THGPUTensor_resizeAs(self_, src1);
  THArgCheck(THGPUTensor_nElement(src1) == THGPUTensor_nElement(src2), 3, "size do not match");

  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  src1 = THGPUTensor_newContiguous(src1);
  src2 = THGPUTensor_newContiguous(src2);
  DECLARE_BOLT_DEVICE_VECTOR_3(src1, src1_data, src2, src2_data, self, self_data);
  bolt::amp::transform(src1_data.begin(), src1_data.end(), src2_data.begin(), self_data.begin(), op);

  THGPUTensor_free(src1);
  THGPUTensor_free(src2);
  THGPUTensor_freeCopyTo(self, self_);
}


void THGPUTensor_ltTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_logicalTensor(self_, src1, src2, bolt::amp::less<float>());
}


void THGPUTensor_gtTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_logicalTensor(self_, src1, src2, bolt::amp::greater<float>());
}


void THGPUTensor_leTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_logicalTensor(self_, src1, src2, bolt::amp::less_equal<float>());
}


void THGPUTensor_geTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_logicalTensor(self_, src1, src2, bolt::amp::greater_equal<float>());
}


void THGPUTensor_eqTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_logicalTensor(self_, src1, src2, bolt::amp::equal_to<float>());
}


void THGPUTensor_neTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2)
{
  THGPUTensor_logicalTensor(self_, src1, src2, bolt::amp::not_equal_to<float>());
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


float THGPUTensor_normall(THGPUTensor *self, float value)
{
  self = THGPUTensor_newContiguous(self);
  long size = THGPUTensor_nElement(self);
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);

  float result;
  if(value == 0.0f) {
    result = bolt::amp::transform_reduce(self_data.begin(), self_data.begin()+size, partial_not_equal_functor(0.0f), (float)0, bolt::amp::plus<float>());

  } else {
    result = bolt::amp::transform_reduce(self_data.begin(), self_data.begin()+size, norm_functor(value), (float)0, bolt::amp::plus<float>());
    result = pow(result, (float)1.0/value);
  }

  THGPUTensor_free(self);
  return result;
}

void THGPUTensor_norm(THGPUTensor* self, THGPUTensor* src, float value, long dimension)
{
  if(value == 0.0f) {
    THGPUTensor_transformReduceDim(self, src, dimension, partial_not_equal_functor(0.0f), (float)0, bolt::amp::plus<float>());
  } else {
    THGPUTensor_transformReduceDim(self, src, dimension, norm_functor(value), (float)0, bolt::amp::plus<float>());
    THGPUTensor_pow(self, self, 1/value);
  }
}

void THGPUTensor_kernel_renorm(Concurrency::array_view<float, 1> &avData, const float value, const long size, const float maxnorm, long gridSz)
{
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
}

void THGPUTensor_renorm(THGPUTensor* self, THGPUTensor* src, float value, long dimension, float maxnorm)
{
  THGPUTensor *self_;
  THGPUTensor *src_ = THGPUTensor_newTranspose(src, dimension, 0);
  THGPUTensor *data = THGPUTensor_newClone(src_);
  long size = THGPUTensor_nElement(data)/data->size[0];
  
  THArgCheck(dimension >= 0 && dimension < THGPUTensor_nDimension(src), 3, "invalid dimension");
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THGPUTensor_nDimension(src) > 1, 1, "need at least 2 dimensions");
  long gridSize = data->size[0] * 32;

  Concurrency::array_view<float, 1> *pavData = static_cast<Concurrency::array_view<float, 1> *>(data->storage->allocatorContext);

  THGPUTensor_kernel_renorm(*pavData, value, size, maxnorm, gridSize);

  THGPUTensor_free(src_);
  self_ = THGPUTensor_newTranspose(data, dimension, 0);
  THGPUTensor_resizeAs(self, self_);
  THGPUTensor_freeCopyTo(self_, self);
  THGPUTensor_free(data);
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

float THGPUTensor_dist(THGPUTensor *self, THGPUTensor *src, float value)
{
  self = THGPUTensor_newContiguous(self);
  long size = THGPUTensor_nElement(self);
  src = THGPUTensor_newContiguous(src);
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  float result = bolt::amp::inner_product(self_data.begin(), self_data.end(), src_data.begin(), (float) 0,bolt::amp::plus<float>(), dist_functor(value));

  THGPUTensor_free(src);
  THGPUTensor_free(self);
  
  return pow(result, (float)1.0/value);
}


void THGPUTensor_rand(THGPURNGState* rng_state, THGPUTensor *r_, THLongStorage *size)
{
  THGPUTensor_resize(r_, size, NULL);
  THGPUTensor_uniform(rng_state, r_, 0, 1);
}
void THGPUTensor_randn(THGPURNGState* rng_state, THGPUTensor *r_, THLongStorage *size)
{
  THGPUTensor_resize(r_, size, NULL);
  THGPUTensor_normal(rng_state, r_, 0, 1);
}

void THGPUTensor_kernel_indexFill(Concurrency::array_view<float, 1> &srcTensor, Concurrency::array_view<long> &srcStride, Concurrency::array_view<long, 1> indx, long src_nDim, 
                                  int dim, long idx_size, long tensor_size, long size_dim, float val, long nblockx
)
{
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
      //for (int i=0; i<idx_size; i++)
      //{
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
      for (int i = 0; i<idx_size; i++)
      {
        srcTensor[(srcIdx + (int)((indx[i])-1)*srcStride[dim])] = val;        
      }
    }
  });
}

void THGPUTensor_kernel_indexCopy(Concurrency::array_view<float, 1> &resTensor, Concurrency::array_view<float, 1> &srcTensor, Concurrency::array_view<long,1> &resStride, Concurrency::array_view<long, 1> indx, long res_size, long res_nDim, int dim, long idx_size, long src_size, long size_dim, long nblockx)
{
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
      //for (int i=0; i<idx_size; i++)
      //{
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
      for (int i = 0; i<idx_size; i++)
      {
        resTensor[Concurrency::index<1>(resIdx + ((int)(indx[Concurrency::index<1>(i)])-1)*(resStride[Concurrency::index<1>(dim)]))] = srcTensor[Concurrency::index<1>(targetIdx +(int) i*(resStride[Concurrency::index<1>(dim)]))];
      }
    }
  });
}

void THGPUTensor_indexCopy(THGPUTensor *res_, int dim, THLongTensor *indices, THGPUTensor *src)
{
  THGPUTensor *indices_;
  Concurrency::array_view<long,1> *stride_;
  long nIndex = indices->size[0];
  long nRes;
  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THGPUTensor_newContiguous(src);

  nRes = THGPUTensor_nElement(res_);

  /*dim3 nthreads(16, 16);*/
  long nblockx = (long)(ceil((float)nRes / nIndex / (16*16)));
  stride_ =  new Concurrency::array_view<long,1>(Concurrency::extent<1>(res_->nDimension),res_->stride);

  Concurrency::array_view<float, 1> *pavRes = static_cast<Concurrency::array_view<float, 1> *>(res_->storage->allocatorContext);
  Concurrency::array_view<float, 1> *pavSrc = static_cast<Concurrency::array_view<float, 1> *>(src->storage->allocatorContext);
  Concurrency::array_view<long, 1> pavInd(indices->storage->size, indices->storage->data);
  
  THGPUTensor_kernel_indexCopy(*pavRes, *pavSrc, 
                               *stride_, pavInd, nRes,
                               res_->nDimension, dim, nIndex, 
                               THGPUTensor_nElement(src), res_->size[dim],nblockx);
  
  delete stride_;
}


void THGPUTensor_indexFill(THGPUTensor *res_, int dim, THLongTensor *indices, float val)
{
  Concurrency::array_view<long,1> *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < res_->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(res_->nDimension > 0, 2, "Source tensor is empty");


  nRes = THGPUTensor_nElement(res_) / res_->size[dim] * nIndex;
  long nblockx = (long)(ceil((float)nRes / nIndex / (16 * 16)));

  stride_ =  new Concurrency::array_view<long,1>(Concurrency::extent<1>(res_->nDimension),res_->stride);

  Concurrency::array_view<float, 1> *pavRes = static_cast<Concurrency::array_view<float, 1> *>(res_->storage->allocatorContext);
  Concurrency::array_view<long, 1> pavInd(indices->storage->size, indices->storage->data);
  
  THGPUTensor_kernel_indexFill(*pavRes, *stride_, pavInd, res_->nDimension, 
                               dim, nIndex, nRes, res_->size[dim], val, nblockx);
  
  delete stride_;
}

void THGPUTensor_kernel_indexSelect(Concurrency::array_view<float, 1> &resTensor, Concurrency::array_view<float, 1> &srcTensor, Concurrency::array_view<long, 1> &srcStride, THLongTensor *indices, long src_nDim, int dim, long idx_size, long tensor_size, long src_size, long size_dim, long nblockx)
{
  Concurrency::array_view<long,1> indx(indices->storage->size, indices->storage->data);
  Concurrency::extent<2> gridExt(16, nblockx * 16);
  Concurrency::tiled_extent<16,16> t_ext(gridExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16,16>tidx) restrict(amp)
  {
    //int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int thread_idx = tidx.tile[1] * t_ext.tile_dim1 * t_ext.tile_dim0 + tidx.local[0] * t_ext.tile_dim1 + tidx.local[1];

    long flat_size = tensor_size / idx_size; 

    if (thread_idx < flat_size)
    {
      long coeff = 0;
      //for (int i=0; i<idx_size; i++)
      //{
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
      for (int i = 0; i<idx_size; i++)
      {
        resTensor[targetIdx + i*srcStride[dim]] = srcTensor[srcIdx + ((int)(indx[i])-1)*srcStride[dim]];
      }
    }
  });
}

void THGPUTensor_indexSelect(THGPUTensor *res_, THGPUTensor *src, int dim, THLongTensor *indices)
{
  THLongStorage *newSize;
  THGPUTensor *indices_;
  Concurrency::array_view<long> *stride_;
  long nIndex = indices->size[0];
  long nRes;
  
  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize, src->size);
  newSize->data[dim] = nIndex;
  THGPUTensor_resize(res_, newSize, NULL);
  THLongStorage_free(newSize);
  
  nRes = THGPUTensor_nElement(res_);
  long nblockx = (long)(ceil((float)nRes / nIndex/(16 * 16)));

  stride_ =  new Concurrency::array_view<long,1>(Concurrency::extent<1>(src->nDimension),src->stride);
  
  Concurrency::array_view<float, 1> *pavRes = static_cast<Concurrency::array_view<float, 1> *>(res_->storage->allocatorContext);
  Concurrency::array_view<float, 1> *pavSrc = static_cast<Concurrency::array_view<float, 1> *>(src->storage->allocatorContext);

  THGPUTensor_kernel_indexSelect(*pavRes, *pavSrc, *stride_, indices, src->nDimension, dim, 
                                 indices->size[0], nRes,THGPUTensor_nElement(src), src->size[dim],nblockx);

  delete stride_;
}
