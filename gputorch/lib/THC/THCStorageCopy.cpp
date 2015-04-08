#include "THCStorageCopy.h"
#include "THCGeneral.h"
#include "copyHelpers.h"

void THGPUStorage_copyFloat(THGPUStorage *self, struct THFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  float* device_ptr = static_cast<float*>(Concurrency::getDevicePointer(self->data));
  THGPUCheck(gpuMemcpy(device_ptr, 0, src->data, 0, self->size * sizeof(float), gpuMemcpyHostToDevice));
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
  float* device_ptr = static_cast<float*>(Concurrency::getDevicePointer(src->data));
  THGPUCheck(gpuMemcpy(self->data, 0, device_ptr, 0, self->size * sizeof(float), gpuMemcpyDeviceToHost));
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

// FIXME: device2device. 'src' is on device
void THGPUStorage_rawCopy(THGPUStorage *self, float *src)
{
  float* device_ptr = static_cast<float*>(Concurrency::getDevicePointer(self->data));
  // TODO: Async copy
  THGPUCheck(gpuMemcpy(device_ptr, 0, src, 0, self->size * sizeof(float), gpuMemcpyDeviceToDevice));
}

void THGPUStorage_copy(THGPUStorage *self, THGPUStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  float* self_ptr = static_cast<float*>(Concurrency::getDevicePointer(self->data));
  float* src_ptr = static_cast<float*>(Concurrency::getDevicePointer(src->data));
  // TODO: Async copy
  THGPUCheck(gpuMemcpy(self_ptr, 0, src_ptr, 0, self->size * sizeof(float), gpuMemcpyDeviceToDevice));
}

void THGPUStorage_copyGPU(THGPUStorage *self, THGPUStorage *src)
{
  THGPUStorage_copy(self, src);
}
