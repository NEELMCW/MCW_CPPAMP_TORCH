#include "THCStorage.h"
#include "THCGeneral.h"
#include "copyHelpers.h"
#include "cl_manage.h"
#include "common.h"
#include "THCBolt.h"

void THGPUStorage_set(THGPUStorage *self, long index, float value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  float* device_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(self->data));
  THGPUCheck(gpuMemcpy(device_ptr, index * sizeof(float), &value, 0, sizeof(float), gpuMemcpyHostToDevice));
}

float THGPUStorage_get(const THGPUStorage *self, long index)
{
  float value;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  float* device_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(self->data));
  THGPUCheck(gpuMemcpy(&value, 0, device_ptr, index * sizeof(float), sizeof(float), gpuMemcpyDeviceToHost));
  return value;
}

THGPUStorage* THGPUStorage_new(void)
{
  int default_size = 0;
  THGPUStorage *storage = (THGPUStorage *)THAlloc(sizeof(THGPUStorage));
  storage->allocatorContext = new Concurrency::array_view<float, 1>(Concurrency::extent<1>(default_size));
  Concurrency::array_view<float>* avData = static_cast<Concurrency::array_view<float, 1>* >(storage->allocatorContext);
  storage->data = avData->data();
  storage->size = default_size;
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
    Concurrency::array_view<float>* avData = new Concurrency::array_view<float>(eA);
    storage->allocatorContext = (void*)avData;
    storage->data = avData->data();
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

// Note that 'data' is on host
THGPUStorage* THGPUStorage_newWithData(float *data, long size)
{
  THGPUStorage *storage = (THGPUStorage *)THAlloc(sizeof(THGPUStorage));
  Concurrency::array_view<float>* avData  = new Concurrency::array_view<float>(Concurrency::extent<1>(size), data);
  storage->allocatorContext = (void*)avData;
  storage->data = data;
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

      self->data = NULL;
      self->size = 0;
      self->refcount =0;
      self->flag = 0;
    }
    THFree(self);
  }
}

void THGPUStorage_fill(THGPUStorage *self, float value)
{
  // Make sure every changes need to be made to its array_view
  PREPARE_AV_WITH_STORAGE(self, pavSelf);
  // Discard host data
  bolt::amp::device_vector<float> avSelf(*pavSelf, self->size, true);
  // Data transfer: 0
  // Memory objects created and released: 0
  bolt::amp::fill(avSelf.begin(), avSelf.end(), value);
}

void THGPUStorage_resize(THGPUStorage *self, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  if (!(self->flag & TH_STORAGE_RESIZABLE))
    return;

  if (size == 0)
  {
    if (self->flag & TH_STORAGE_FREEMEM)
    {
      if (self->allocatorContext)
      {
        Concurrency::array_view<float, 1> delSelf (*(static_cast<Concurrency::array_view<float,1>*>(self->allocatorContext)));
        delSelf.~array_view();
      }
      else
      {
        Concurrency::array_view<float, 1> delSelf (Concurrency::extent<1>(self->size), self->data);
        delSelf.~array_view();
      }
    }
    self->data = NULL;
    self->size = 0;
  }
  else if (self->size != size)
  {
    // Resizing the extent
    Concurrency::extent<1> eA(size);
    // Allocating device array of resized value
    Concurrency::array_view<float, 1> *avDest = new Concurrency::array_view<float>(eA);
    float* dest_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(avDest->data()));
    Concurrency::array_view<float, 1>* avSrc = static_cast<Concurrency::array_view<float, 1>* >(self->allocatorContext);
    float* src_ptr = static_cast<float*>(Concurrency::getAllocator().device_data(self->data));
    // TODO: Async copy
    if (src_ptr)
      THGPUCheck(gpuMemcpy(dest_ptr, 0, src_ptr, 0, THMin(self->size, size) * sizeof(float), gpuMemcpyDeviceToDevice));

    avSrc->~array_view();
    self->allocatorContext = (void *)avDest;
    self->data = avDest->data();
    self->size = size;
    // Set default refcount for the new resized 
    self->refcount = 1;
  }
}
