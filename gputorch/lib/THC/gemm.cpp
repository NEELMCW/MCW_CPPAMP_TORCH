#include "gemm.h"
#define OFFSET(N, incX) ((incX) > 0 ? 0 : ((N) - 1) * (-(incX)))
#define BLOCK_SIZE 256
#define TILE_DIM 16
void gemm_NoTransAB(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta)
{
  Concurrency::extent<2> grdExt((N+15)&~15 , (M+15)&~15);
  Concurrency::tiled_extent<16, 16> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16, 16> tidx) restrict(amp){
  float CValue = 0;
   
  int Row = tidx.tile[0]*TILE_DIM + tidx.local[0];
  int Col = tidx.tile[1]*TILE_DIM + tidx.local[1];

  tile_static float As[TILE_DIM][TILE_DIM];
  tile_static float Bs[TILE_DIM][TILE_DIM];
                
  for (int k = 0; k < (TILE_DIM + K - 1)/TILE_DIM; k++) {                     
    if (k*TILE_DIM + tidx.local[1] < K && Row < N)        
      Bs[tidx.local[0]][tidx.local[1]] = B[Row*K + k*TILE_DIM + tidx.local[1]];
    else
      Bs[tidx.local[0]][tidx.local[1]] = 0.0;

    if (k*TILE_DIM + tidx.local[0] < K && Col < M)        
      As[tidx.local[0]][tidx.local[1]] = A[(k*TILE_DIM + tidx.local[0])*M + Col];
    else                                                                                                        
      As[tidx.local[0]][tidx.local[1]] = 0.0;
        
    tidx.barrier.wait();

    for (int n = 0; n < TILE_DIM; ++n) CValue += Bs[tidx.local[0]][n] * As[n][tidx.local[1]] * alpha;
               
    tidx.barrier.wait();
   }
   
   if (Row < N && Col < M) 
   {
     C[(tidx.global[0]*M)+tidx.global[1]]*=beta;
     C[(tidx.global[0]*M)+tidx.global[1]]+=CValue;
   }
  });
}

void gemm_NoTransB(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta)
{
  Concurrency::extent<2> grdExt((N+15)&~15, (M+15)&~15);
  Concurrency::tiled_extent<16, 16> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16, 16> tidx) restrict(amp){
  float temp;
  int j = tidx.global[0];
  int i = tidx.global[1];
  if(i<M && j<N)
  {
    temp = 0;
    for (int l = 0; l < K; ++l) {
      temp += A[l+i*lda] * B[l+j*ldb];
    }
    C[i+j*ldc] = alpha*temp + beta*C[i+j*ldc];
  }
  });
}

void gemm_NoTransA(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta)
{
  Concurrency::extent<2> grdExt((N+15)&~15, (M+15)&~15);
  Concurrency::tiled_extent<16, 16> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16, 16> tidx) restrict(amp){
  float temp;
  int j = tidx.global[0];
  int i = tidx.global[1];
  if(i < M && j < N)
  {
    C[i+j*ldc] *= beta;
    for (int l = 0; l < K; ++l) {
      C[i+j*ldc] += A[i+l*lda] * alpha * B[j+l*ldb];
    }
  }
  });
}

void gemm_TransAB(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta)
{
  Concurrency::extent<2> grdExt((N+15)&~15, (M+15)&~15);
  Concurrency::tiled_extent<16, 16> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<16, 16> tidx) restrict(amp){
  float temp;
  int j = tidx.global[0];
  int i = tidx.global[1];
  if(i < M && j < N)
  {
    temp = 0;
    for (int l = 0; l < K; ++l) {
      temp += A[l+i*lda] * B[j+l*ldb];
    }
    C[i+j*ldc] = alpha*temp + beta*C[i+j*ldc];
  }
  });
}

int gemm_AMP(char TransA, char TransB, const int M, const int N, const int K, const float alpha,
  float* A, int lda, float* B, int ldb,
  const float beta, float* C,  int ldc)
{
  int num_rows_a, /*num_cols_a,*/ num_rows_b; // nrowa, ncola, nrowb

  // use longest possible type for intermediate value storage:
  float temp;
  // %%= if [:rational,:complex,:value].include?(dtype.type); "#{dtype.long_dtype.sizeof} temp1, temp2;"; end%%
  int i, j, l;

  if (TransA == 'n') 
    num_rows_a = M;
  else                        
    num_rows_a = K;

  if (TransB == 'n') 
     num_rows_b = K;
  else                        
     num_rows_b = N;

  if (M < 0) {
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

  // Quick return if possible
  if (!M || !N || (alpha == 0 || !K) && beta == 1) return 0;

  // For alpha = 0
  if (alpha == 0) {
    if (beta == 0) {
      for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i) {
          C[i+j*ldc] = 0;
        }
    } else {
      for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i) {
          C[i+j*ldc] *= beta;
        }
    }
    return 0;
  }
  Concurrency::array_view<float,1> A_mat(M * K, A);
  Concurrency::array_view<float,1> B_mat(K * N, B);
  Concurrency::array_view<float,1> C_mat(M * N, C);
  // Start the operations
  if (TransB == 'n') {
    if (TransA == 'n') {
      // C = alpha*A*B+beta*C

      gemm_NoTransAB(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta);
    } else {

      // C = alpha*A**T*B + beta*C

      gemm_NoTransB(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta);
    }

  } else if (TransA == 'n') {

    // C = alpha*A*B**T + beta*C

      gemm_NoTransA(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta);
  } else {

    // C = alpha*A**T*B**T + beta*C

      gemm_TransAB(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta);
  }

  return 0;
}

void gemv_TransA(Concurrency::array_view<float> &A_mat, int aOffset, Concurrency::array_view<float> &X_vec, Concurrency::array_view<float> &Y_vec, float alpha, float beta,int lenX, int lenY)
{

  long size = (lenY + 255) & ~255;
  
  Concurrency::extent<1> compute_domain(size);

  Concurrency::parallel_for_each(compute_domain.tile<BLOCK_SIZE>(),[=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
  {
    int bx = tidx.tile[0];
    int tx = tidx.local[0];

    tile_static float Xds[BLOCK_SIZE];

    int Col = bx * BLOCK_SIZE + tx;

    float Pvalue = 0;

    for(int m = 0; m < (lenX -1)/BLOCK_SIZE+1; ++m)
    {
      if(m * BLOCK_SIZE + tx < lenX)
        Xds[tx] = X_vec[m*BLOCK_SIZE+tx];
      else
        Xds[tx]=0;

      tidx.barrier.wait();

      for(int k = 0; k < BLOCK_SIZE; k++)
        if(Col < lenY && m * BLOCK_SIZE + k < lenX)
          Pvalue += Xds[k] * A_mat[aOffset + m * BLOCK_SIZE + Col * lenX + k];
      tidx.barrier.wait();
    }

   if(Col < lenY)
   {
      Y_vec[Col] *= beta; 
      Y_vec[Col] += alpha * Pvalue;
   }
    
    tidx.barrier.wait();

  });

}


void gemv_NoTransA(Concurrency::array_view<float> &A, Concurrency::array_view<float> &X, Concurrency::array_view<float> &Y, float alpha, float beta,int lenX, int lenY)
{
  for (int j = 0; j < lenX; j++) 
  {
    const float temp = alpha * X[j];
    if (temp != 0.0) {
      for (int i = 0; i < lenY; i++) {
        Y[i]*=beta;
        Y[i] += temp * A[lenX * j + i];
      }
    }
  }
}

void gemv_AMP(char TransA,
int M, int N, float alpha, Concurrency::array_view<float> &A,
int aOffset, Concurrency::array_view<float> &X,  int incX, float beta,
Concurrency::array_view<float> &Y,  int incY)
{
  if (alpha == 0.0)
    return;

  int  i, j;
  int lenX, lenY;
  if (M == 0 || N == 0)
    return;
  if (alpha == 0.0 && beta == 1.0)
    return;
  if (TransA == 'n') {
    lenX = N;
    lenY = M;
  } else {
    lenX = M;
    lenY = N;
  }
 
 if(TransA == 't') {
    gemv_TransA(A, aOffset, X, Y, alpha, beta, lenX, lenY);
    /* form y := alpha*A*x + y */
  } else if (TransA == 'n'){
  /* form y := alpha*A'*x + y */
    gemv_NoTransA(A, X, Y, alpha, beta, lenX, lenY);
  } 
}

