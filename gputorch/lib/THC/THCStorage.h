#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"
#include "THCGeneral.h"
#include "amp.h"

#undef TH_API
#define TH_API THC_API
#define real float
#define Real GPU

#define TH_GENERIC_FILE "generic/THStorage.h"
#include "generic/THStorage.h"
#undef TH_GENERIC_FILE

#define TH_GENERIC_FILE "generic/THStorageCopy.h"
#include "generic/THStorageCopy.h"
#undef TH_GENERIC_FILE

#undef real
#undef Real
#undef TH_API
#ifdef WIN32
# define TH_API THC_EXTERNC __declspec(dllimport)
#else
# define TH_API THC_EXTERNC
#endif

#endif
