#ifndef GEMM_DEFS_H
#define GEMM_DEFS_H

#include "amp.h"

int gemm_AMP(char TransA, char TransB, const int M, const int N, const int K, const float alpha, float* A, int lda, float* B, int ldb,const float beta, float* C, int ldc);


void gemv_AMP(char TransA,
int M, int N, float alpha, Concurrency::array_view<float> &A,
int lda, Concurrency::array_view<float> &X,  int incX, float beta,
Concurrency::array_view<float> &Y,  int incY);
#endif
