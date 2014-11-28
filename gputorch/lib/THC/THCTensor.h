#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include "THTensor.h"
#include "THCStorage.h"

#undef TH_API
#define TH_API THC_API
#define real float
#define Real GPU

#define TH_GENERIC_FILE "generic/THTensor.h"
#include "generic/THTensor.h"
#undef TH_GENERIC_FILE

#define TH_GENERIC_FILE "generic/THTensorCopy.h"
#include "generic/THTensorCopy.h"
#undef TH_GENERIC_FILE

#undef real
#undef Real
#undef TH_API
#ifdef WIN32
# define TH_API THC_EXTERNC __declspec(dllimport)
#else
# define TH_API THC_EXTERNC
#endif

THC_API void THGPUTensor_fill(THGPUTensor *self, float value);
THC_API void THGPUTensor_copy(THGPUTensor *self, THGPUTensor *src);

THC_API void THByteTensor_copyGPU(THByteTensor *self, THGPUTensor *src);
THC_API void THCharTensor_copyGPU(THCharTensor *self, THGPUTensor *src);
THC_API void THShortTensor_copyGPU(THShortTensor *self, THGPUTensor *src);
THC_API void THIntTensor_copyGPU(THIntTensor *self, THGPUTensor *src);
THC_API void THLongTensor_copyGPU(THLongTensor *self, THGPUTensor *src);
THC_API void THFloatTensor_copyGPU(THFloatTensor *self, THGPUTensor *src);
THC_API void THDoubleTensor_copyGPU(THDoubleTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_copyGPU(THGPUTensor *self, THGPUTensor *src);

#endif
