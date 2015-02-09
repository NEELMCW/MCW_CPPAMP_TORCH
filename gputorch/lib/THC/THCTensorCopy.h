#ifndef TH_GPU_TENSOR_COPY_INC
#define TH_GPU_TENSOR_COPY_INC

#include "THCTensor.h"
#include "THCGeneral.h"

THC_API void THGPUTensor_copy(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_copyByte(THGPUTensor *self, THByteTensor *src);
THC_API void THGPUTensor_copyChar(THGPUTensor *self, THCharTensor *src);
THC_API void THGPUTensor_copyShort(THGPUTensor *self, THShortTensor *src);
THC_API void THGPUTensor_copyInt(THGPUTensor *self, THIntTensor *src);
THC_API void THGPUTensor_copyLong(THGPUTensor *self, THLongTensor *src);
THC_API void THGPUTensor_copyFloat(THGPUTensor *self, THFloatTensor *src);
THC_API void THGPUTensor_copyDouble(THGPUTensor *self, THDoubleTensor *src);

THC_API void THByteTensor_copyGPU(THByteTensor *self, THGPUTensor *src);
THC_API void THCharTensor_copyGPU(THCharTensor *self, THGPUTensor *src);
THC_API void THShortTensor_copyGPU(THShortTensor *self, THGPUTensor *src);
THC_API void THIntTensor_copyGPU(THIntTensor *self, THGPUTensor *src);
THC_API void THLongTensor_copyGPU(THLongTensor *self, THGPUTensor *src);
THC_API void THFloatTensor_copyGPU(THFloatTensor *self, THGPUTensor *src);
THC_API void THDoubleTensor_copyGPU(THDoubleTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_copyGPU(THGPUTensor *self, THGPUTensor *src);

#endif
