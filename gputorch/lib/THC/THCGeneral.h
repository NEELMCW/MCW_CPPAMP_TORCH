#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include "THGeneral.h"
#include "clBLAS.h"
#undef log1p

extern cl_device_id mdevice;
extern cl_context mcontext;
extern cl_command_queue mqueue;

#ifdef __cplusplus
# define THC_EXTERNC extern "C"
#else
# define THC_EXTERNC extern
#endif

#ifdef WIN32
# ifdef THC_EXPORTS
#  define THC_API THC_EXTERNC __declspec(dllexport)
# else
#  define THC_API THC_EXTERNC __declspec(dllimport)
# endif
#else
# define THC_API THC_EXTERNC
#endif

THC_API void THCudaBlas_init(int numdevices, int current_device);
THC_API void THCudaBlas_shutdown();

THC_API void THCudaInit(void);
THC_API void THCudaShutdown(void);



THC_API void THCudaGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size);

#endif
