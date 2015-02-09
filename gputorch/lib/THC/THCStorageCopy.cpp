#include "THCStorageCopy.h"
#include "THCGeneral.h"
#include "common.h"

void THGPUStorage_copyFloat(THGPUStorage *self, struct THFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  // FIXME: host2device copy
  THGPUStorage_rawCopy(self, src->data);
}

#define TH_GPU_STORAGE_IMPLEMENT_COPY(TYPEC)                                      \
void THGPUStorage_copy##TYPEC(THGPUStorage *self, struct TH##TYPEC##Storage *src) \
{                                                                                 \
  THFloatStorage *buffer;                                                         \
  THArgCheck(self->size == src->size, 2, "size does not match");                  \
  buffer = THFloatStorage_newWithSize(src->size);                                 \
  THFloatStorage_copy##TYPEC(buffer, src);                                        \
  THGPUStorage_copyFloat(self, buffer);                                           \
  THFloatStorage_free(buffer);                                                    \
}

TH_GPU_STORAGE_IMPLEMENT_COPY(Byte)
TH_GPU_STORAGE_IMPLEMENT_COPY(Char)
TH_GPU_STORAGE_IMPLEMENT_COPY(Short)
TH_GPU_STORAGE_IMPLEMENT_COPY(Int)
TH_GPU_STORAGE_IMPLEMENT_COPY(Long)
TH_GPU_STORAGE_IMPLEMENT_COPY(Double)

void THFloatStorage_copyGPU(THFloatStorage *self, struct THGPUStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  // TODO: device2host copy
  Concurrency::array_view<float, 1> arrSrc(Concurrency::extent<1>(self->size), src->data);
  Concurrency::array_view<float, 1> avSelfCopy(Concurrency::extent<1>(self->size), self->data);
  copy(arrSrc, avSelfCopy);
}

#define TH_GPU_STORAGE_IMPLEMENT_COPYTO(TYPEC)                                      \
void TH##TYPEC##Storage_copyGPU(TH##TYPEC##Storage *self, struct THGPUStorage *src) \
{                                                                                   \
  THFloatStorage *buffer;                                                           \
  THArgCheck(self->size == src->size, 2, "size does not match");                    \
  buffer = THFloatStorage_newWithSize(src->size);                                   \
  THFloatStorage_copyGPU(buffer, src);                                              \
  TH##TYPEC##Storage_copyFloat(self, buffer);                                       \
  THFloatStorage_free(buffer);                                                      \
}

TH_GPU_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_GPU_STORAGE_IMPLEMENT_COPYTO(Char)
TH_GPU_STORAGE_IMPLEMENT_COPYTO(Short)
TH_GPU_STORAGE_IMPLEMENT_COPYTO(Int)
TH_GPU_STORAGE_IMPLEMENT_COPYTO(Long)
TH_GPU_STORAGE_IMPLEMENT_COPYTO(Double)

void THGPUStorage_rawCopy(THGPUStorage *self, float *src)
{
  // TODO: device2device async copy
  Concurrency::array_view<float> avSelfCopy(Concurrency::extent<1>(self->size), self->data);
  avSelfCopy.discard_data();
  MemcpyHostToAV(src, self->size, avSelfCopy);
}

void THGPUStorage_copy(THGPUStorage *self, THGPUStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  // TODO: device2device async copy
  PREPARE_AV_WITH_STORAGE(src, arrSrc);
  PREPARE_AV_WITH_STORAGE(self, avSelfCopy);
  MemcpyAVToAV(arrSrc, src->size, avSelfCopy);
}

void THGPUStorage_copyGPU(THGPUStorage *self, THGPUStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  // TODO: device2device async copy
  PREPARE_AV_WITH_STORAGE(src, arrSrc);
  PREPARE_AV_WITH_STORAGE(self, avSelfCopy);
  MemcpyAVToAV(arrSrc, src->size, avSelfCopy);
}
