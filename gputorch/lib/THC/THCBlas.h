#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"

#undef TH_API
#define TH_API THC_API
#define real float
#define Real GPU
#define THBlas_(NAME) TH_CONCAT_4(TH,Real,Blas_,NAME)

#define TH_GENERIC_FILE "generic/THBlas.h"
#include "generic/THBlas.h"
#undef TH_GENERIC_FILE

#undef THBlas_
#undef real
#undef Real
#undef TH_API

#ifdef WIN32
# define TH_API THC_EXTERNC __declspec(dllimport)
#else
# define TH_API THC_EXTERNC
#endif

#include "amp.h"

void THGPUBlas_gemv_opt(char trans, long m, long n, float alpha, 
  Concurrency::array_view<float> &a, long lda, Concurrency::array_view<float> &x, long incx, float beta, Concurrency::array_view<float> &y, long incy);

void THGPUBlas_gemm_opt(char transa, char transb,
  long m, long n, long k, float alpha,
  Concurrency::array_view<float> &a, long lda, Concurrency::array_view<float> &b, long ldb, float beta,
  Concurrency::array_view<float> &c, long ldc,
  void* cl_A, void* cl_B, void* cl_C,
  long aOffset, long bOffset, long cOffset);

void THGPUBlas_gemv_opt1(char trans, long m, long n, float alpha, 
  float *a, long lda, float *x, long incx, float beta, float *y, long incy,
  void* cl_A, void* cl_X, void* cl_Y, long aOffset, long xOffset, long yOffset);

void THGPUBlas_gemm_opt1(char transa, char transb,
  long m, long n, long k, float alpha,
  float *a, long lda, float *b, long ldb, float beta,
  float *c, long ldc,
  void* cl_A, void* cl_B, void* cl_C,
  long aOffset, long bOffset, long cOffset);


#endif

