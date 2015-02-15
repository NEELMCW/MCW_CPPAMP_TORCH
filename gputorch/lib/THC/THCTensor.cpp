#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCTensorCopy.h"

/**** access methods ****/
THGPUStorage *THGPUTensor_storage(const THGPUTensor *self)
{
  return self->storage;
}

long THGPUTensor_storageOffset(const THGPUTensor *self)
{
  return self->storageOffset;
}

int THGPUTensor_nDimension(const THGPUTensor *self)
{
  return self->nDimension;
}

long THGPUTensor_size(const THGPUTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

long THGPUTensor_stride(const THGPUTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THGPUTensor_newSizeOf(THGPUTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THGPUTensor_newStrideOf(THGPUTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

float *THGPUTensor_data(const THGPUTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THGPUTensor_setFlag(THGPUTensor *self, const char flag)
{
  self->flag |= flag;
}

void THGPUTensor_clearFlag(THGPUTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THGPUTensor_rawInit(THGPUTensor *self);
static void THGPUTensor_rawSet(THGPUTensor *self, THGPUStorage *storage, long storageOffset, int nDimension, long *size, long *stride);
static void THGPUTensor_rawResize(THGPUTensor *self, int nDimension, long *size, long *stride);


/* Empty init */
THGPUTensor *THGPUTensor_new()
{
  THGPUTensor *self = (THGPUTensor*)THAlloc(sizeof(THGPUTensor));
  THGPUTensor_rawInit(self);
  return self;
}

/* Pointer-copy init */
THGPUTensor *THGPUTensor_newWithTensor(THGPUTensor *tensor)
{
  THGPUTensor *self = (THGPUTensor*)THAlloc(sizeof(THGPUTensor));
  THGPUTensor_rawInit(self);
  THGPUTensor_rawSet(self,
                      tensor->storage,
                      tensor->storageOffset,
                      tensor->nDimension,
                      tensor->size,
                      tensor->stride);
  return self;
}

/* Storage init */
THGPUTensor *THGPUTensor_newWithStorage(THGPUStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THGPUTensor *self = (THGPUTensor*)THAlloc(sizeof(THGPUTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THGPUTensor_rawInit(self);
  THGPUTensor_rawSet(self,
                      storage,
                      storageOffset,
                      (size ? size->size : (stride ? stride->size : 0)),
                      (size ? size->data : NULL),
                      (stride ? stride->data : NULL));

  return self;
}
THGPUTensor *THGPUTensor_newWithStorage1d(THGPUStorage *storage, long storageOffset,
                               long size0, long stride0)
{
  return THGPUTensor_newWithStorage4d(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THGPUTensor *THGPUTensor_newWithStorage2d(THGPUStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THGPUTensor_newWithStorage4d(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THGPUTensor *THGPUTensor_newWithStorage3d(THGPUStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THGPUTensor_newWithStorage4d(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THGPUTensor *THGPUTensor_newWithStorage4d(THGPUStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4] = {size0, size1, size2, size3};
  long stride[4] = {stride0, stride1, stride2, stride3};

  THGPUTensor *self = (THGPUTensor*)THAlloc(sizeof(THGPUTensor));
  THGPUTensor_rawInit(self);
  THGPUTensor_rawSet(self, storage, storageOffset, 4, size, stride);

  return self;
}

THGPUTensor *THGPUTensor_newWithSize(THLongStorage *size, THLongStorage *stride)
{
  return THGPUTensor_newWithStorage(NULL, 0, size, stride);
}

THGPUTensor *THGPUTensor_newWithSize1d(long size0)
{
  return THGPUTensor_newWithSize4d(size0, -1, -1, -1);
}

THGPUTensor *THGPUTensor_newWithSize2d(long size0, long size1)
{
  return THGPUTensor_newWithSize4d(size0, size1, -1, -1);
}

THGPUTensor *THGPUTensor_newWithSize3d(long size0, long size1, long size2)
{
  return THGPUTensor_newWithSize4d(size0, size1, size2, -1);
}

THGPUTensor *THGPUTensor_newWithSize4d(long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THGPUTensor *self = (THGPUTensor*)THAlloc(sizeof(THGPUTensor));
  THGPUTensor_rawInit(self);
  THGPUTensor_rawResize(self, 4, size, NULL);

  return self;
}

THGPUTensor *THGPUTensor_newClone(THGPUTensor *self)
{
  THGPUTensor *tensor = THGPUTensor_new();
  THGPUTensor_resizeAs(tensor, self);
  THGPUTensor_copy(tensor, self);
  return tensor;
}

THGPUTensor *THGPUTensor_newContiguous(THGPUTensor *self)
{
  if(!THGPUTensor_isContiguous(self))
    return THGPUTensor_newClone(self);
  else
  {
    THGPUTensor_retain(self);
    return self;
  }
}

THGPUTensor *THGPUTensor_newSelect(THGPUTensor *tensor, int dimension_, long sliceIndex_)
{
  THGPUTensor *self = THGPUTensor_newWithTensor(tensor);
  THGPUTensor_select(self, NULL, dimension_, sliceIndex_);
  return self;
}

THGPUTensor *THGPUTensor_newNarrow(THGPUTensor *tensor, int dimension_, long firstIndex_, long size_)
{
  THGPUTensor *self = THGPUTensor_newWithTensor(tensor);
  THGPUTensor_narrow(self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THGPUTensor *THGPUTensor_newTranspose(THGPUTensor *tensor, int dimension1_, int dimension2_)
{
  THGPUTensor *self = THGPUTensor_newWithTensor(tensor);
  THGPUTensor_transpose(self, NULL, dimension1_, dimension2_);
  return self;
}

THGPUTensor *THGPUTensor_newUnfold(THGPUTensor *tensor, int dimension_, long size_, long step_)
{
  THGPUTensor *self = THGPUTensor_newWithTensor(tensor);
  THGPUTensor_unfold(self, NULL, dimension_, size_, step_);
  return self;
}

/* Resize */
void THGPUTensor_resize(THGPUTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THGPUTensor_rawResize(self, size->size, size->data, (stride ? stride->data : NULL));
}

void THGPUTensor_resizeAs(THGPUTensor *self, THGPUTensor *src)
{
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THGPUTensor_rawResize(self, src->nDimension, src->size, NULL);
}

void THGPUTensor_resize1d(THGPUTensor *tensor, long size0)
{
  THGPUTensor_resize4d(tensor, size0, -1, -1, -1);
}

void THGPUTensor_resize2d(THGPUTensor *tensor, long size0, long size1)
{
  THGPUTensor_resize4d(tensor, size0, size1, -1, -1);
}

void THGPUTensor_resize3d(THGPUTensor *tensor, long size0, long size1, long size2)
{
  THGPUTensor_resize4d(tensor, size0, size1, size2, -1);
}

void THGPUTensor_resize4d(THGPUTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THGPUTensor_rawResize(self, 4, size, NULL);
}

void THGPUTensor_resize5d(THGPUTensor *self, long size0, long size1, long size2, long size3, long size4)
{
    long size[5] = {size0, size1, size2, size3, size4};

  THGPUTensor_rawResize(self, 5, size, NULL);
}

void THGPUTensor_set(THGPUTensor *self, THGPUTensor *src)
{
  if(self != src)
    THGPUTensor_rawSet(self,
                        src->storage,
                        src->storageOffset,
                        src->nDimension,
                        src->size,
                        src->stride);
}

void THGPUTensor_setStorage(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THGPUTensor_rawSet(self,
                      storage_,
                      storageOffset_,
                      (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                      (size_ ? size_->data : NULL),
                      (stride_ ? stride_->data : NULL));
}

void THGPUTensor_setStorage1d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                             long size0_, long stride0_)
{
  THGPUTensor_setStorage4d(self, storage_, storageOffset_,
                            size0_, stride0_,
                            -1, -1,
                            -1, -1,
                            -1, -1);
}

void THGPUTensor_setStorage2d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_)
{
  THGPUTensor_setStorage4d(self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            -1, -1,
                            -1, -1);
}

void THGPUTensor_setStorage3d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_)
{
  THGPUTensor_setStorage4d(self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            size2_, stride2_,
                            -1, -1);
}

void THGPUTensor_setStorage4d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_,
                             long size3_, long stride3_)
{

  long size[4] = {size0_, size1_, size2_, size3_};
  long stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THGPUTensor_rawSet(self, storage_, storageOffset_, 4, size, stride);
}


void THGPUTensor_narrow(THGPUTensor *self, THGPUTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 4, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 5, "out of range");

  THGPUTensor_set(self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THGPUTensor_select(THGPUTensor *self, THGPUTensor *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THGPUTensor_set(self, src);
  THGPUTensor_narrow(self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THGPUTensor_transpose(THGPUTensor *self, THGPUTensor *src, int dimension1, int dimension2)
{
  long z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THGPUTensor_set(self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THGPUTensor_unfold(THGPUTensor *self, THGPUTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THGPUTensor_set(self, src);

  newSize = (long*)THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = (long*)THAlloc(sizeof(long)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;
}

/* we have to handle the case where the result is a number */
void THGPUTensor_squeeze(THGPUTensor *self, THGPUTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THGPUTensor_set(self, src);

  for(d = 0; d < src->nDimension; d++)
  {
    if(src->size[d] != 1)
    {
      if(d != ndim)
      {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->nDimension > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
}

void THGPUTensor_squeeze1d(THGPUTensor *self, THGPUTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THGPUTensor_set(self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}

int THGPUTensor_isContiguous(const THGPUTensor *self)
{
  long z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THGPUTensor_isSameSizeAs(const THGPUTensor *self, const THGPUTensor* src)
{
  int d;
  if (self->nDimension != src->nDimension)
    return 0;
  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

long THGPUTensor_nElement(const THGPUTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THGPUTensor_retain(THGPUTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    ++self->refcount;
}

void THGPUTensor_free(THGPUTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(--self->refcount == 0)
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THGPUStorage_free(self->storage);
      THFree(self);
    }
  }
}

void THGPUTensor_freeCopyTo(THGPUTensor *self, THGPUTensor *dst)
{
  if(self != dst)
    THGPUTensor_copy(dst, self);

  THGPUTensor_free(self);
}

/*******************************************************************************/

static void THGPUTensor_rawInit(THGPUTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

static void THGPUTensor_rawSet(THGPUTensor *self, THGPUStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THGPUStorage_free(self->storage);

    if(storage)
    {
      self->storage = storage;
      THGPUStorage_retain(self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THGPUTensor_rawResize(self, nDimension, size, stride);
}

static void THGPUTensor_rawResize(THGPUTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = (long*)THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = (long*)THRealloc(self->stride, sizeof(long)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THGPUStorage_new();
      if(totalSize+self->storageOffset > self->storage->size)
        THGPUStorage_resize(self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THGPUTensor_set1d(THGPUTensor *tensor, long x0, float value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THGPUStorage_set(tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

float THGPUTensor_get1d(const THGPUTensor *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THGPUStorage_get(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THGPUTensor_set2d(THGPUTensor *tensor, long x0, long x1, float value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THGPUStorage_set(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

float THGPUTensor_get2d(const THGPUTensor *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THGPUStorage_get(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THGPUTensor_set3d(THGPUTensor *tensor, long x0, long x1, long x2, float value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THGPUStorage_set(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

float THGPUTensor_get3d(const THGPUTensor *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THGPUStorage_get(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THGPUTensor_set4d(THGPUTensor *tensor, long x0, long x1, long x2, long x3, float value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THGPUStorage_set(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

float THGPUTensor_get4d(const THGPUTensor *tensor, long x0, long x1, long x2, long x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THGPUStorage_get(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

