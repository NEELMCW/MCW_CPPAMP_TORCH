#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.h"
#else

/* Level 1 */
TH_API void THBlas_(swap)(long n, real *x, long incx, real *y, long incy);
TH_API void THBlas_(scal)(long n, real a, real *x, long incx);
TH_API void THBlas_(copy)(long n, real *x, long incx, real *y, long incy);
TH_API void THBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy);
TH_API real THBlas_(dot)(long n, real *x, long incx, real *y, long incy);

/* Level 2 */
TH_API void THBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy);
TH_API void THBlas_(gemv_opt)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy, void* cl_A, void* cl_X, void* cl_Y);

TH_API void THBlas_(ger)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda);

/* Level 3 */
TH_API void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc);

TH_API void* THBlas_(clCreateBuffer)(long m, long k, float* a);
TH_API void THBlas_(gemm_opt)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta,long offset, real *c, long ldc , void* cl_A, void* cl_B, void* cl_C);

#endif
