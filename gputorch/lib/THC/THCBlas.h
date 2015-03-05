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

void THGPUBlas_gemv(char trans, long m, long n, float alpha,
                        Concurrency::array_view<float> &a, long aOffset,
                        Concurrency::array_view<float> &x, long xOffset, long incx, float beta,
                        Concurrency::array_view<float> &y, long yOffset, long incy,
                        Concurrency::array_view<float> &temp_buf);

void THGPUBlas_gemm(char transa, char transb,
                        const long m, const long n, const long k, const float alpha,
                        Concurrency::array_view<float> &a, long aOffset, long lda,
                        Concurrency::array_view<float> &b, long bOffset, long ldb, const float beta,
                        Concurrency::array_view<float> &c, long cOffset, long ldc);

void THGPUBlas_axpy(long n, float a,
                        Concurrency::array_view<float> &x, long xOffset, long incx,
                        Concurrency::array_view<float> &y, long yOffset, long incy);

void THGPUBlas_ger(long m, long n, float alpha,
                       Concurrency::array_view<float> &x, long xOffset, long incx,
                       Concurrency::array_view<float> &y, long yOffset, long incy,
                       Concurrency::array_view<float> &a, long aOffset, long lda);

#endif
