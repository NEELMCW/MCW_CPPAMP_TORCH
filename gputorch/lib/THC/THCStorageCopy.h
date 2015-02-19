#ifndef THC_STORAGE_COPY_INC
#define THC_STORAGE_COPY_INC

#include "THCStorage.h"
#include "THCGeneral.h"

/* Support for copy between different Storage types */

THC_API void THGPUStorage_rawCopy(THGPUStorage *storage, float *src);
THC_API void THGPUStorage_copy(THGPUStorage *storage, THGPUStorage *src);
THC_API void THGPUStorage_copyByte(THGPUStorage *storage, struct THByteStorage *src);
THC_API void THGPUStorage_copyChar(THGPUStorage *storage, struct THCharStorage *src);
THC_API void THGPUStorage_copyShort(THGPUStorage *storage, struct THShortStorage *src);
THC_API void THGPUStorage_copyInt(THGPUStorage *storage, struct THIntStorage *src);
THC_API void THGPUStorage_copyLong(THGPUStorage *storage, struct THLongStorage *src);
THC_API void THGPUStorage_copyFloat(THGPUStorage *storage, struct THFloatStorage *src);
THC_API void THGPUStorage_copyDouble(THGPUStorage *storage, struct THDoubleStorage *src);

THC_API void THByteStorage_copyGPU(THByteStorage *self, struct THGPUStorage *src);
THC_API void THCharStorage_copyGPU(THCharStorage *self, struct THGPUStorage *src);
THC_API void THShortStorage_copyGPU(THShortStorage *self, struct THGPUStorage *src);
THC_API void THIntStorage_copyGPU(THIntStorage *self, struct THGPUStorage *src);
THC_API void THLongStorage_copyGPU(THLongStorage *self, struct THGPUStorage *src);
THC_API void THFloatStorage_copyGPU(THFloatStorage *self, struct THGPUStorage *src);
THC_API void THDoubleStorage_copyGPU(THDoubleStorage *self, struct THGPUStorage *src);
THC_API void THGPUStorage_copyGPU(THGPUStorage *self, THGPUStorage *src);

#endif
