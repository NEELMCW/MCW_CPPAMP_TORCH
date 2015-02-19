#ifndef COPY_HELPERS_
#define COPY_HELPERS_

#include "THCTensor.h"

// Copy routines from CLAMP runtime

enum gpuMemcpyKind {
  gpuMemcpyHostToHost,
  gpuMemcpyHostToDevice,
  gpuMemcpyDeviceToHost,
  gpuMemcpyDeviceToDevice,
  gpuMemcpyDefault
};

THC_API int gpuMemcpy(void* dst, size_t dst_offset,
                            void* src, size_t src_offset,
                            size_t count, gpuMemcpyKind kind);
THC_API int gpuMemcpyAsync(void* dst, size_t dst_offset,
                                   void* src, size_t src_offset,
                                   size_t count, gpuMemcpyKind kind);


#endif
