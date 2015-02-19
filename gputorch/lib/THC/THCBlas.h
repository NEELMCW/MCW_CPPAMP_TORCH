#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"
#include "THCAMPBlasImpl.h"

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
                        Concurrency::array_view<float> &a, long aOffset,
                        Concurrency::array_view<float> &x, long xOffset, long incx, float beta,
                        Concurrency::array_view<float> &y, long yOffset, long incy,
                        Concurrency::array_view<float> &temp_buf);

void THGPUBlas_gemm_opt(char transa, char transb,
                        long m, long n, long k, float alpha,
                        Concurrency::array_view<float> &a, long aOffset, long lda,
                        Concurrency::array_view<float> &b, long bOffset, long ldb, float beta,
                        Concurrency::array_view<float> &c, long cOffset, long ldc,
                        Concurrency::array_view<float> &temp_buff);

void THGPUBlas_axpy_opt(long n, float a,
                        Concurrency::array_view<float> &x, long xOffset, long incx,
                        Concurrency::array_view<float> &y, long yOffset, long incy);

void THGPUBlas_ger_opt(long m, long n, float alpha,
                       Concurrency::array_view<float> &x, long xOffset, long incx,
                       Concurrency::array_view<float> &y, long yOffset, long incy,
                       Concurrency::array_view<float> &a, long aOffset, long lda);

#endif
