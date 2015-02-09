#include "THCStorage.h"
#include "THCGeneral.h"
#include "common.h"

void THGPUStorage_set(THGPUStorage *self, long index, float value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  // FIXME: host2device copy
  Concurrency::array_view<float> *avData = static_cast<Concurrency::array_view<float> *>(self->allocatorContext);
  (*avData)[Concurrency::index<1>(index)] = value;
}

float THGPUStorage_get(const THGPUStorage *self, long index)
{
  float value;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  // FIXME: device2host copy
  Concurrency::array_view<float> *avData = static_cast<Concurrency::array_view<float> *>(self->allocatorContext);
  value = (*avData)[Concurrency::index<1>(index)];
  return value;
}

THGPUStorage* THGPUStorage_new(void)
{
  int default_size = 0;
  THGPUStorage *storage = (THGPUStorage *)THAlloc(sizeof(THGPUStorage));
  storage->allocatorContext = new Concurrency::array_view<float, 1>(Concurrency::extent<1>(default_size));
  Concurrency::array_view<float>avData(*(Concurrency::array_view<float, 1>*)storage->allocatorContext);
  storage->data = avData.data();
  storage->size = default_size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THGPUStorage* THGPUStorage_newWithSize(long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
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
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    ++self->refcount;
}

void THGPUStorage_free(THGPUStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (--(self->refcount) <= 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM)
    {
      if (self->allocatorContext)
      {
        delete (Concurrency::array_view<float> *)self->allocatorContext;
        self->allocatorContext = NULL;
      }
      //FIXME: no need to double free and program will not jump to this branch
      // since self->data that is associated with array_view has already been released
      #if 0
      else if (self->data)
      {
        Concurrency::array_view<float, 1> delSelf (Concurrency::extent<1>(self->size), self->data);
        delSelf.~array_view();
      }
      #endif
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
      if (self->allocatorContext) {
        Concurrency::array_view<float, 1> delSelf (*(static_cast<Concurrency::array_view<float,1>*>(self->allocatorContext)));
        delSelf.~array_view();
      } else {
        Concurrency::array_view<float, 1> delSelf (Concurrency::extent<1>(self->size), self->data);
        delSelf.~array_view();
      }
    }
    self->data = NULL;
    self->size = 0;
  }
  else if (self->size != size)
  {
    Concurrency::array_view<float, 1> *data = NULL;
    // Resizing the extent
    Concurrency::extent<1> eA(size);
    // Allocating device array of resized value
    data =  new Concurrency::array_view<float>(eA);
    long copySize = size;//THMin(self->size, size);
    // We need to release the previous container. Generally it is allocated by THGPUStorage_new
    // with a default size (4 bytes if size=1). Note that, the call graph is described as below,
    //   (1) default bytes are created in THGPUStorage_new (#1 clCreateBuffer)
    //   (2) N bytes are reallocated in THGPUStorage_resize in the same storage (#2 clCreateBuffer)
    //         see av is created with a new extent: data =  new Concurrency::array_view<float>(eA);
    //   (3) default bytes need explicitly released in here to avoid memory leak (#1 clReleaseMemObject)
    //   (4) N bytes will be released in THGPUStorage_free called by user (#2 clReleaseMemObject)
    //
    // Note that if the default size is zero in THGPUStorage_new, (1)/(3) will not happen since
    // the underlying driver (CLAMP) just skips memory allocation when a specified size=0
    // Forcelly to release even its refcount is not zero
    if (1)
    {
      Concurrency::array_view<float, 1> previous(*(static_cast<Concurrency::array_view<float,1>*>(self->allocatorContext)));
      previous.~array_view();
    }

    Concurrency::extent<1> copyExt(copySize);
    Concurrency::array_view<float, 1> desData(data->section(copyExt));
    self->allocatorContext = (void *)data;
    self->data = desData.data();
    self->size = size;
    // Set default refcount for the new resized 
    self->refcount=1;
  }
}

