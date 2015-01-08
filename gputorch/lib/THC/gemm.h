#ifndef GEMM_DEFS_H
#define GEMM_DEFS_H

#include "amp.h"

int gemm_AMP(char TransA, char TransB, const int M, const int N, const int K, const float alpha, float* A, int lda, float* B, int ldb,const float beta, float* C, int ldc);
#endif
void gemv_AMP(char TransA,
int M, int N, float alpha, float *A,
int lda, float *X,  int incX,  float beta,
float *Y,  int incY);
