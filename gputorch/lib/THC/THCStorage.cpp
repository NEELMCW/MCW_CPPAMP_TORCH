#include "THCStorage.h"
#include "THCGeneral.h"

void THCudaStorage_set(THCudaStorage *self, long index, float value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  Concurrency::array_view<float> avData(Concurrency::extent<1>(self->size),self->data);
  avData[Concurrency::index<1>(index)] = value;
}

float THCudaStorage_get(const THCudaStorage *self, long index)
{
  float value;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  Concurrency::array_view<float> avData(Concurrency::extent<1>(self->size),self->data);
  value = avData[Concurrency::index<1>(index)];
  return value;
}

THCudaStorage* THCudaStorage_new(void)
{
  THCudaStorage *storage = (THCudaStorage *)THAlloc(sizeof(THCudaStorage));
  storage->allocatorContext = new Concurrency::array<float,1>(Concurrency::extent<1>(1));
  Concurrency::array_view<float>avData(*(Concurrency::array<float, 1>*)storage->allocatorContext);
  storage->data = avData.data();
  storage->size = 1;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THCudaStorage* THCudaStorage_newWithSize(long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
  {
    THCudaStorage *storage = (THCudaStorage *)THAlloc(sizeof(THCudaStorage));
    Concurrency::extent<1> eA(size);
    // Allocating device array of given size
    storage->allocatorContext = new Concurrency::array<float>(eA);
    Concurrency::array_view<float>avData(*(Concurrency::array<float>*)storage->allocatorContext);
    storage->data = avData.data();
    storage->size = size;
    storage->refcount = 1;
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    return storage;
  }
  else
  {
    return THCudaStorage_new();
  }
}

THCudaStorage* THCudaStorage_newWithSize1(float data0)
{
  THCudaStorage *self = THCudaStorage_newWithSize(1);
  THCudaStorage_set(self, 0, data0);
  return self;
}

THCudaStorage* THCudaStorage_newWithSize2(float data0, float data1)
{
  THCudaStorage *self = THCudaStorage_newWithSize(2);
  THCudaStorage_set(self, 0, data0);
  THCudaStorage_set(self, 1, data1);
  return self;
}

THCudaStorage* THCudaStorage_newWithSize3(float data0, float data1, float data2)
{
  THCudaStorage *self = THCudaStorage_newWithSize(3);
  THCudaStorage_set(self, 0, data0);
  THCudaStorage_set(self, 1, data1);
  THCudaStorage_set(self, 2, data2);
  return self;
}

THCudaStorage* THCudaStorage_newWithSize4(float data0, float data1, float data2, float data3)
{
  THCudaStorage *self = THCudaStorage_newWithSize(4);
  THCudaStorage_set(self, 0, data0);
  THCudaStorage_set(self, 1, data1);
  THCudaStorage_set(self, 2, data2);
  THCudaStorage_set(self, 3, data3);
  return self;
}

THCudaStorage* THCudaStorage_newWithMapping(const char *fileName, long size, int isShared)
{
  THError("not available yet for THCudaStorage");
  return NULL;
}

THCudaStorage* THCudaStorage_newWithData(float *data, long size)
{
  THCudaStorage *storage = (THCudaStorage *)THAlloc(sizeof(THCudaStorage));
  storage->allocatorContext = new Concurrency::array<float>(Concurrency::extent<1>(size),data);
  Concurrency::array_view<float> avData(*(Concurrency::array<float>*)storage->allocatorContext);
  storage->data = avData.data();
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

void THCudaStorage_retain(THCudaStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    ++self->refcount;
}

void THCudaStorage_free(THCudaStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (--(self->refcount) <= 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM)
    {
      if(self->allocatorContext)
      {
        delete (Concurrency::array<float> *)self->allocatorContext;
        self->allocatorContext = NULL;
      }
      if(self->data)
      {
        Concurrency::array<float,1> delSelf (Concurrency::extent<1>(self->size),self->data);
        delSelf.~array();
        self->data = NULL;
        self->size = 0;
        self->refcount =0;
        self->flag = 0;
      }
    }
    THFree(self);
  }
}

void THCudaStorage_copyFloat(THCudaStorage *self, struct THFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaStorage_rawCopy(self,src->data);
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                           \
  void THCudaStorage_copy##TYPEC(THCudaStorage *self, struct TH##TYPEC##Storage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copy##TYPEC(buffer, src);                            \
    THCudaStorage_copyFloat(self, buffer);                              \
    THFloatStorage_free(buffer);                                        \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)

void THFloatStorage_copyCuda(THFloatStorage *self, struct THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  Concurrency::array<float,1> arrSrc(Concurrency::extent<1>(self->size),src->data);
  Concurrency::array_view<float,1> avSelfCopy(Concurrency::extent<1>(self->size),self->data);
  copy(arrSrc, avSelfCopy);
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                           \
  void TH##TYPEC##Storage_copyCuda(TH##TYPEC##Storage *self, struct THCudaStorage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copyCuda(buffer, src);                               \
    TH##TYPEC##Storage_copyFloat(self, buffer);                         \
    THFloatStorage_free(buffer);                                        \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)

void THCudaStorage_fill(THCudaStorage *self, float value)
{
  Concurrency::array_view<float, 1> srcData(Concurrency::extent<1>(self->size),self->data);
  Concurrency::parallel_for_each(srcData.get_extent(), [=] (Concurrency::index<1> idx) restrict(amp)
  {
    srcData[idx] = value; 
  });
  srcData.synchronize();
}

void THCudaStorage_resize(THCudaStorage *self, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    return;

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM)
    {
      Concurrency::array<float,1> delSelf (Concurrency::extent<1>(self->size),self->data);
      delSelf.~array();
    }
    self->data = NULL;
    self->size = 0;
  }
  else
  {
    Concurrency::array<float, 1> *data = NULL;
    // Resizing the extent
    Concurrency::extent<1> eA(size);
    // Allocating device array of resized value
    data =  new Concurrency::array<float>(eA);
    long copySize = THMin(self->size,size);
    Concurrency::extent<1> copyExt(copySize);
    Concurrency::array_view<float, 1> srcData(copyExt,self->data);
    Concurrency::array_view<float, 1> desData(data->section(copyExt));
    Concurrency::copy(srcData,desData);
    Concurrency::array<float,1> delSelf (Concurrency::extent<1>(self->size),self->data);
    delSelf.~array();
    delete (Concurrency::array<float> *)self->allocatorContext;
    self->allocatorContext = (void *)data;
    self->data = desData.data();
    self->size = size;
  }  
}

void THCudaStorage_rawCopy(THCudaStorage *self, float *src)
{
  Concurrency::array<float> arrSrc(Concurrency::extent<1>(self->size),src);
  Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->size),self->data);
  Concurrency::copy(arrSrc, avSelfCopy);
}

void THCudaStorage_copy(THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  Concurrency::array<float> arrSrc(Concurrency::extent<1>(self->size),src->data);
  Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->size),self->data);
  Concurrency::copy(arrSrc, avSelfCopy);
}

void THCudaStorage_copyCuda(THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  Concurrency::array<float> arrSrc(Concurrency::extent<1>(self->size),src->data);
  Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->size),self->data);
  Concurrency::copy(arrSrc, avSelfCopy);
}
