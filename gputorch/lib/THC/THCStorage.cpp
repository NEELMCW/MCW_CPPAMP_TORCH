#include "THCStorage.h"
#include "THCGeneral.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/fill.h"

void THGPUStorage_set(THGPUStorage *self, long index, float value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  Concurrency::array_view<float> avData(Concurrency::extent<1>(self->size), self->data);
  avData[Concurrency::index<1>(index)] = value;
}

float THGPUStorage_get(const THGPUStorage *self, long index)
{
  float value;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  Concurrency::array_view<float> avData(Concurrency::extent<1>(self->size), self->data);
  value = avData[Concurrency::index<1>(index)];
  return value;
}

THGPUStorage* THGPUStorage_new(void)
{
  THGPUStorage *storage = (THGPUStorage *)THAlloc(sizeof(THGPUStorage));
  storage->allocatorContext = new Concurrency::array_view<float, 1>(Concurrency::extent<1>(1));
  Concurrency::array_view<float>avData(*(Concurrency::array_view<float, 1>*)storage->allocatorContext);
  storage->data = avData.data();
  storage->size = 1;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THGPUStorage* THGPUStorage_newWithSize(long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if (size > 0)
  {
    THGPUStorage *storage = (THGPUStorage *)THAlloc(sizeof(THGPUStorage));
    Concurrency::extent<1> eA(size);
    // Allocating device array of given size
    storage->allocatorContext = new Concurrency::array_view<float>(eA);
    Concurrency::array_view<float>avData(*(Concurrency::array_view<float>*)storage->allocatorContext);
    storage->data = avData.data();
    storage->size = size;
    storage->refcount = 1;
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    return storage;
  }
  else
  {
    return THGPUStorage_new();
  }
}

THGPUStorage* THGPUStorage_newWithSize1(float data0)
{
  THGPUStorage *self = THGPUStorage_newWithSize(1);
  THGPUStorage_set(self, 0, data0);
  return self;
}

THGPUStorage* THGPUStorage_newWithSize2(float data0, float data1)
{
  THGPUStorage *self = THGPUStorage_newWithSize(2);
  THGPUStorage_set(self, 0, data0);
  THGPUStorage_set(self, 1, data1);
  return self;
}

THGPUStorage* THGPUStorage_newWithSize3(float data0, float data1, float data2)
{
  THGPUStorage *self = THGPUStorage_newWithSize(3);
  THGPUStorage_set(self, 0, data0);
  THGPUStorage_set(self, 1, data1);
  THGPUStorage_set(self, 2, data2);
  return self;
}

THGPUStorage* THGPUStorage_newWithSize4(float data0, float data1, float data2, float data3)
{
  THGPUStorage *self = THGPUStorage_newWithSize(4);
  THGPUStorage_set(self, 0, data0);
  THGPUStorage_set(self, 1, data1);
  THGPUStorage_set(self, 2, data2);
  THGPUStorage_set(self, 3, data3);
  return self;
}

THGPUStorage* THGPUStorage_newWithMapping(const char *fileName, long size, int isShared)
{
  THError("not available yet for THGPUStorage");
  return NULL;
}

THGPUStorage* THGPUStorage_newWithData(float *data, long size)
{
  THGPUStorage *storage = (THGPUStorage *)THAlloc(sizeof(THGPUStorage));
  storage->allocatorContext = new Concurrency::array_view<float>(Concurrency::extent<1>(size), data);
  Concurrency::array_view<float> avData(*(Concurrency::array_view<float>*)storage->allocatorContext);
  storage->data = avData.data();
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

void THGPUStorage_retain(THGPUStorage *self)
{
  if (self && (self->flag & TH_STORAGE_REFCOUNTED))
    ++self->refcount;
}

void THGPUStorage_free(THGPUStorage *self)
{
  if (!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (--(self->refcount) <= 0)
  {
    if (self->flag & TH_STORAGE_FREEMEM)
    {
      if (self->allocatorContext)
      {
        delete (Concurrency::array_view<float> *)self->allocatorContext;
        self->allocatorContext = NULL;
      }
      if (self->data)
      {
        Concurrency::array_view<float, 1> delSelf (Concurrency::extent<1>(self->size), self->data);
        delSelf.~array_view();
        self->data = NULL;
        self->size = 0;
        self->refcount =0;
        self->flag = 0;
      }
    }
    THFree(self);
  }
}

void THGPUStorage_copyFloat(THGPUStorage *self, struct THFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THGPUStorage_rawCopy(self, src->data);
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                                     \
void THGPUStorage_copy##TYPEC(THGPUStorage *self, struct TH##TYPEC##Storage *src) \
{                                                                                 \
  THFloatStorage *buffer;                                                         \
  THArgCheck(self->size == src->size, 2, "size does not match");                  \
  buffer = THFloatStorage_newWithSize(src->size);                                 \
  THFloatStorage_copy##TYPEC(buffer, src);                                        \
  THGPUStorage_copyFloat(self, buffer);                                           \
  THFloatStorage_free(buffer);                                                    \
}

TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)

void THFloatStorage_copyGPU(THFloatStorage *self, struct THGPUStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  Concurrency::array_view<float, 1> arrSrc(Concurrency::extent<1>(self->size), src->data);
  Concurrency::array_view<float, 1> avSelfCopy(Concurrency::extent<1>(self->size), self->data);
  copy(arrSrc, avSelfCopy);
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                                     \
void TH##TYPEC##Storage_copyGPU(TH##TYPEC##Storage *self, struct THGPUStorage *src) \
{                                                                                   \
  THFloatStorage *buffer;                                                           \
  THArgCheck(self->size == src->size, 2, "size does not match");                    \
  buffer = THFloatStorage_newWithSize(src->size);                                   \
  THFloatStorage_copyGPU(buffer, src);                                              \
  TH##TYPEC##Storage_copyFloat(self, buffer);                                       \
  THFloatStorage_free(buffer);                                                      \
}

TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)

void THGPUStorage_fill(THGPUStorage *self, float value)
{
  bolt::amp::fill(self->data, self->data+self->size, value);
}

void THGPUStorage_resize(THGPUStorage *self, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  std::cout<<"Resize invoked"<<std::endl;

  if (!(self->flag & TH_STORAGE_RESIZABLE))
    return;

  if (size == 0)
  {
    if (self->flag & TH_STORAGE_FREEMEM)
    {
      Concurrency::array_view<float, 1> delSelf (Concurrency::extent<1>(self->size), self->data);
      delSelf.~array_view();
    }
    self->data = NULL;
    self->size = 0;
  }
  else
  {
    Concurrency::array_view<float, 1> *data = NULL;
    // Resizing the extent
    Concurrency::extent<1> eA(size);
    // Allocating device array of resized value
    data =  new Concurrency::array_view<float>(eA);
    long copySize = THMin(self->size, size);
    Concurrency::extent<1> copyExt(copySize);
    Concurrency::array_view<float, 1> srcData(copyExt, self->data);
    Concurrency::array_view<float, 1> desData(data->section(copyExt));
    Concurrency::copy(srcData, desData);
    Concurrency::array_view<float,1> delSelf (Concurrency::extent<1>(self->size), self->data);
    delSelf.~array_view();
    delete (Concurrency::array_view<float> *)self->allocatorContext;
    self->allocatorContext = (void *)data;
    self->data = desData.data();
    self->size = size;
  }
}

void THGPUStorage_rawCopy(THGPUStorage *self, float *src)
{
  Concurrency::array_view<float> arrSrc(Concurrency::extent<1>(self->size), src);
  Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->size), self->data);
  Concurrency::copy(arrSrc, avSelfCopy);
}

void THGPUStorage_copy(THGPUStorage *self, THGPUStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  Concurrency::array_view<float> arrSrc(Concurrency::extent<1>(self->size), src->data);
  Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->size), self->data);
  Concurrency::copy(arrSrc, avSelfCopy);
}

void THGPUStorage_copyGPU(THGPUStorage *self, THGPUStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  Concurrency::array_view<float> arrSrc(Concurrency::extent<1>(self->size), src->data);
  Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->size), self->data);
  Concurrency::copy(arrSrc, avSelfCopy);
}
