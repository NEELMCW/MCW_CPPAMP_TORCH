#ifndef GEMM_DEFS_H
#define GEMM_DEFS_H

#include <string>
#include "amp.h"

enum class order { row_major, col_major };
enum class transpose { no_trans, trans, conj_trans };

enum AMPBLAS_ORDER {AmpblasRowMajor=101, AmpblasColMajor=102};
enum AMPBLAS_TRANSPOSE {AmpblasNoTrans=111, AmpblasTrans=112, AmpblasConjTrans=113};

struct gemm_tuning_parameters
{
    // work block
    static const int m_block = 16; 
    static const int n_block = 16;
    static const int k_block = 16;

    // tile sizes
    static const int m_c_tile = 16;
    static const int n_c_tile = 16;
     
    static const int m_a_tile = 16;
    static const int n_a_tile = 16;
     
    static const int m_b_tile = 16;
    static const int n_b_tile = 16;

    // shared memory padding
    static const int use_padding = 1;
};

int gemm(char TransA, char TransB, const int M, const int N, const int K, const float alpha,
  float* A, const int lda, float* B, const int ldb,
  const float beta, float* C, const int ldc);
//void gemm(int m, int n, int k, char transa, char transb, float alpha, float* a, float* b, float beta, float* c, int lda, int ldb, int ldc); 
#endif
