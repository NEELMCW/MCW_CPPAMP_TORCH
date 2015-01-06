#include "gemm.h"

// Stage 5: Highly parameterized GEMM
int gemm(char TransA, char TransB, const int M, const int N, const int K, const float alpha,
  float* A, const int lda, float* B, const int ldb,
  const float beta, float* C, const int ldc)
{
  int num_rows_a, /*num_cols_a,*/ num_rows_b; // nrowa, ncola, nrowb

  // use longest possible type for intermediate value storage:
  // %%= if [:rational,:complex,:value].include?(dtype.type); "#{dtype.long_dtype.sizeof} temp1, temp2;"; end%%

  if (TransA == 'n') num_rows_a = M;
  else                        num_rows_a = K;

  if (TransB == 'n') num_rows_b = K;
  else                        num_rows_b = N;

  // Test the input parameters
 /* if (TransA < 111 || TransA > 113) {
    fprintf(stderr, "GEMM: TransA must be CblasNoTrans, CblasTrans, or CblasConjTrans\n");
    return 0;
  } else if (TransB < 111 || TransB > 113) {
    fprintf(stderr, "GEMM: TransB must be CblasNoTrans, CblasTrans, or CblasConjTrans\n");
    return 0;
  } else */if (M < 0) {
    fprintf(stderr, "GEMM: Expected M >= 0\n");
    return 0;
  } else if (N < 0) {
    fprintf(stderr, "GEMM: Expected N >= 0\n");
    return 0;
  } else if (K < 0) {
    fprintf(stderr, "GEMM: Expected K >= 0\n");
    return 0;
  } else if (lda < std::max(1, num_rows_a)) {
    fprintf(stderr, "GEMM: Expected lda >= max(1, num_rows_a), with num_rows_a = %d; got lda=%d\n", num_rows_a, lda);
    return 0;
  } else if (ldb < std::max(1, num_rows_b)) {
    fprintf(stderr, "GEMM: Expected ldb >= max(1, num_rows_b), with num_rows_b = %d; got ldb=%d\n", num_rows_b, ldb);
    return 0;
  } else if (ldc < std::max(1,M)) {
    fprintf(stderr, "GEMM: Expected ldc >= max(1,M) with M=%d; got ldc=%d\n", M, ldc);
    return 0;
  }
std::cout<<"0"<<std::endl;
  // Quick return if possible
  if (!M || !N || (alpha == 0 || !K) && beta == 1) return 0;


  //Concurrency::array_view<float,1> A_mat(M * K, A);
  //Concurrency::array_view<float,1> B_mat(K * N, B);
  //Concurrency::array_view<float,1> C_mat(M * N, C);

  //Concurrency::extent<1> grdExt(1);
  //Concurrency::tiled_extent<1> t_ext(grdExt);


  int i, j, l;
  float temp;
  // For alpha = 0
  if (alpha == 0) {
    if (beta == 0) {
      for (j = 0; j < N; j++)
        for (i = 0; i < M; ++i) {
          C[i+j*ldc] = 0;
        }
    } else {
      for (j = 0; j < N; j++)
        for (i = 0; i < M; ++i) {
          C[i+j*ldc] *= beta;
        }
    }
    return 0;
  }
  std::cout<<"1"<<std::endl;
  Concurrency::array_view<float,1> A_mat(M * K, A);
  Concurrency::array_view<float,1> B_mat(K * N, B);
  Concurrency::array_view<float,1> C_mat(M * N, C);

  Concurrency::extent<1> grdExt(1);
  Concurrency::tiled_extent<16> t_ext(grdExt);
  // Start the operations
  if (TransB == 'n') {
    if (TransA == 'n') {
      // C = alpha*A*B+beta*C
  std::cout<<"2"<<std::endl;
     Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16> tidx) restrict(amp){
      float temp;
      for (int j = tidx.global[0]; j < N; j+=t_ext[0]) {
        for (int l = 0; l < K; ++l) {
            temp = alpha * B_mat[l+j*ldb];
            for (int i = 0; i < M; ++i) {
              C_mat[i+j*ldc] *= beta;
              C_mat[i+j*ldc] += A_mat[i+l*lda] * temp;
            }
        }
      }
      });
    } else {

      // C = alpha*A**T*B + beta*C
  std::cout<<"3"<<std::endl;
      Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1> tidx) restrict(amp){
      float temp;
      for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
          temp = 0;
          for (int l = 0; l < K; ++l) {
            temp += A_mat[l+i*lda] * B_mat[l+j*ldb];
          }

          C_mat[i+j*ldc] = alpha*temp + beta*C_mat[i+j*ldc];
        }
      }
      });

    }

  } else if (TransA == 'n') {

    // C = alpha*A*B**T + beta*C

  std::cout<<"4"<<std::endl;
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1> tidx) restrict(amp){
    float temp;
    for (int j = 0; j < N; ++j) {
      for (int l = 0; l < K; ++l) {
          temp = alpha * B_mat[j+l*ldb];
          for (int i = 0; i < M; ++i) {
            C_mat[i+j*ldc] *= beta;
            C_mat[i+j*ldc] += A_mat[i+l*lda] * temp;
          }
      }
    }
    });

  } else {

    // C = alpha*A**T*B**T + beta*C
  std::cout<<"5"<<std::endl;
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1> tidx) restrict(amp){
    float temp;
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        temp = 0;
        for (int l = 0; l < K; ++l) {
          temp += A_mat[l+i*lda] * B_mat[j+l*ldb];
        }

          C_mat[i+j*ldc] = alpha*temp + beta*C_mat[i+j*ldc];
      }
    }
    });
  }
  return 0;
}
/*void gemm_kernel(float alpha, float* a, float* b, float beta,  float* c, int m, int n, int k, int lda, int ldb, int ldc)
{
  //Concurrency::array_view<float,1> a_mat(m * k, a);
  //Concurrency::array_view<float,1> b_mat(k * n, b);
  //Concurrency::array_view<float,1> c_mat(m * n, c);
  for (int i=0; i<m; i++) {
               for (int j=0; j<n; j++) {
                       c[i * m + j] *= beta;
                       for (int p=0; p<k; p++) {
                               c[i * m + j] += alpha * a[i * m + p] * b[p * k + j];
                       }
               }
       }   
}

void gemm(int m, int n, int k, const char transa, const char transb, float alpha, float* a, float* b, float beta, float* c, int lda, int ldb, int ldc)  
{ 
  float *a_trans, *b_trans;
  int m1, n1, k1;

  if(transa == 't')
  {
    for(int i = 0; i < m; i++)
      for(int j = 0; j < k; j++)
        a_trans[i * k + j] = a[j * m + i];
    m1 = k;
    k1 = m;
  }
  else
  {
    a_trans = a;
    m1 = m;
    k1 = k;
  }

  if(transb == 't')
  {
    for(int i = 0; i < k; i++)
      for(int j = 0; j < n; j++)
        b_trans[i * n + j] = b[j * k + i];
    k1 = n;
    n1 = k;
  }
  else
  {
    b_trans = b;
    k1 = k;
    n1 = n;
  }

  gemm_kernel(alpha, a_trans, b_trans, beta, c, m1, n1, k1, lda, ldb, ldc);
}*/
