#ifndef THCAMPBLASIMPL_H
#define THCAMPBLASIMPL_H

#include "amp.h"


int gemm_AMP(char TransA, char TransB, const int M, const int N, const int K, const float alpha,
             Concurrency::array_view<float> &A_mat, long aOffset, long lda,
             Concurrency::array_view<float>& B_mat, long bOffset, long ldb, const float beta,
             Concurrency::array_view<float>& C_mat, long cOffset, long ldc,
             Concurrency::array_view<float> &temp_buf);

void gemv_AMP(char TransA, int M, int N, float alpha,
              Concurrency::array_view<float> &A, long aOffset,
              Concurrency::array_view<float> &X, long xOffset, long incX, float beta,
              Concurrency::array_view<float> &Y, long yOffset, long incY,
              Concurrency::array_view<float> &temp_buf);

void axpy_AMP(long n, float alpha,
              Concurrency::array_view<float> &X, long xOffset, long incx,
              Concurrency::array_view<float> &Y, long yOffset, long incy);

void ger_AMP(long m, long n, float alpha, 
             Concurrency::array_view<float> &x, long xOffset, long incx,
             Concurrency::array_view<float> &y, long yOffset, long incy,
             Concurrency::array_view<float> &a, long aOffset, long lda);
#endif
