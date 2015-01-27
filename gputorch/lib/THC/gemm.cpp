#include "gemm.h"
#define OFFSET(N, incX) ((incX) > 0 ? 0 : ((N) - 1) * (-(incX)))
#define BLOCK_SIZE 256
#define TILE_DIM 16
#define THREADS 16
void gemm_NoTransAB(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta, long aOffset, long bOffset, long cOffset)
{
  Concurrency::extent<2> grdExt((N+(THREADS-1))&~(THREADS-1) , (M+(THREADS-1))&~(THREADS-1));
  Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp){
  float CValue = 0;
   
  int Row = tidx.tile[0]*TILE_DIM + tidx.local[0];
  int Col = tidx.tile[1]*TILE_DIM + tidx.local[1];

  tile_static float As[TILE_DIM][TILE_DIM];
  tile_static float Bs[TILE_DIM][TILE_DIM];
                
  for (int k = 0; k < (TILE_DIM + K - 1)/TILE_DIM; k++) {                     
    if (k*TILE_DIM + tidx.local[1] < K && Row < N)        
      Bs[tidx.local[0]][tidx.local[1]] = B[bOffset +Row*K + k*TILE_DIM + tidx.local[1]];
    else
      Bs[tidx.local[0]][tidx.local[1]] = 0.0;

    if (k*TILE_DIM + tidx.local[0] < K && Col < M)        
      As[tidx.local[0]][tidx.local[1]] = A[aOffset + (k*TILE_DIM + tidx.local[0])*M + Col];
    else                                                                                                        
      As[tidx.local[0]][tidx.local[1]] = 0.0;
        
    tidx.barrier.wait();

    for (int n = 0; n < TILE_DIM; ++n) CValue += Bs[tidx.local[0]][n] * As[n][tidx.local[1]] * alpha;
               
    tidx.barrier.wait();
   }
   
   if (Row < N && Col < M) 
   {
     C[cOffset + (tidx.global[0]*M)+tidx.global[1]]*=beta;
     C[cOffset + (tidx.global[0]*M)+tidx.global[1]]+=CValue;
   }
  });
}

void gemm_NoTransB(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta , long aOffset, long bOffset, long cOffset)
{
  Concurrency::extent<2> grdExt((N+(THREADS-1))&~(THREADS-1), (M+(THREADS-1))&~(THREADS-1));
  Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp){

  float CValue = 0;

  int Row = tidx.tile[0]*TILE_DIM + tidx.local[0];
  int Col = tidx.tile[1]*TILE_DIM + tidx.local[1];

  tile_static float As[TILE_DIM][TILE_DIM];
  tile_static float Bs[TILE_DIM][TILE_DIM];

  for (int k = 0; k < (TILE_DIM + K - 1)/TILE_DIM; k++)
  {

    if (k*TILE_DIM + tidx.local[1] < K && Row < N)
      Bs[tidx.local[0]][tidx.local[1]] = B[bOffset + Row*K + k*TILE_DIM + tidx.local[1]];
    else
      Bs[tidx.local[0]][tidx.local[1]] = 0.0;

    if (k*TILE_DIM + tidx.local[1] < K && (tidx.tile[1]*TILE_DIM + tidx.local[0]) < M)
      As[tidx.local[0]][tidx.local[1]] = A[aOffset + (tidx.tile[1]*TILE_DIM + tidx.local[0])*K + k*TILE_DIM + tidx.local[1]];
    else
      As[tidx.local[0]][tidx.local[1]] = 0.0;

    tidx.barrier.wait();

    for (int n = 0; n < TILE_DIM; ++n) CValue += Bs[tidx.local[0]][n] * As[tidx.local[1]][n];

    tidx.barrier.wait();
  }

  if (Row < N && Col < M)
  {
    C[cOffset + (Row*M)+Col]*=beta;
    C[cOffset + (Row*M)+Col]+=CValue * alpha;
  }
  });
}

void gemm_NoTransA(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta , long aOffset, long bOffset, long cOffset)
{
  Concurrency::extent<2> grdExt((N+(THREADS-1))&~(THREADS-1), (M+(THREADS-1))&~(THREADS-1));
  Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp){
  float CValue = 0;

  int Row = tidx.tile[0]*TILE_DIM + tidx.local[0];
  int Col = tidx.tile[1]*TILE_DIM + tidx.local[1];

  tile_static float As[TILE_DIM][TILE_DIM];
  tile_static float Bs[TILE_DIM][TILE_DIM];

  for (int k = 0; k < (TILE_DIM + K - 1)/TILE_DIM; k++)
  {

    if (k*TILE_DIM + tidx.local[0] < K && (tidx.tile[0]*TILE_DIM + tidx.local[1]) < N)
      Bs[tidx.local[0]][tidx.local[1]] = B[bOffset + (k*TILE_DIM + tidx.local[0])*N + (tidx.tile[0]*TILE_DIM + tidx.local[1])];
    else
      Bs[tidx.local[0]][tidx.local[1]] = 0.0;

    if (k*TILE_DIM + tidx.local[0] < K && Col < M)
      As[tidx.local[0]][tidx.local[1]] = A[aOffset + (k*TILE_DIM + tidx.local[0])*M + Col];
    else
      As[tidx.local[0]][tidx.local[1]] = 0.0;

    tidx.barrier.wait();

    for (int n = 0; n < TILE_DIM; ++n) CValue += Bs[n][tidx.local[0]] * As[n][tidx.local[1]];

    tidx.barrier.wait();
  }

  if (Row < N && Col < M)
  {
    C[cOffset + (Row*M)+Col]*=beta;
    C[cOffset + (Row*M)+Col]+=CValue * alpha;
  }
  });
}

void gemm_TransAB(Concurrency::array_view<float, 1> &A, Concurrency::array_view<float, 1> &B, Concurrency::array_view<float, 1> &C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, float beta , long aOffset, long bOffset, long cOffset) 
{
  Concurrency::extent<2> grdExt((N+(THREADS-1))&~(THREADS-1), (M+(THREADS-1))&~(THREADS-1));
  Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp){
  float temp;
  int j = tidx.global[0];
  int i = tidx.global[1];
  if(i < M && j < N)
  {
    temp = 0;
    for (int l = 0; l < K; ++l) {
      temp += A[aOffset + l+i*lda] * B[bOffset + j+l*ldb];
    }
    C[cOffset + i+j*ldc] = alpha*temp + beta*C[cOffset + i+j*ldc];
  }
  });
}

int gemm_AMP(char TransA, char TransB, const int M, const int N, const int K, const float alpha,
  Concurrency::array_view<float> &A_mat, int lda, Concurrency::array_view<float>& B_mat, int ldb,
  const float beta, Concurrency::array_view<float>& C_mat,  int ldc, long aOffset, long bOffset, long cOffset)
{
  // use longest possible type for intermediate value storage:

  // %%= if [:rational,:complex,:value].include?(dtype.type); "#{dtype.long_dtype.sizeof} temp1, temp2;"; end%%
  int i, j;

  // Quick return if possible
  if (!M || !N || ((alpha == 0 || !K) && beta == 1)) return 0;

  // For alpha = 0
  if (alpha == 0) {
    if (beta == 0) {
      for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i) {
          C_mat[i+j*ldc] = 0;
        }
    } else {
      for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i) {
          C_mat[i+j*ldc] *= beta;
        }
    }
    return 0;
  }
  // Start the operations
  if (TransB == 'n') {
    if (TransA == 'n') {
      // C = alpha*A*B+beta*C

      gemm_NoTransAB(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta ,aOffset, bOffset, cOffset);
    } else {

      // C = alpha*A**T*B + beta*C

      gemm_NoTransB(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta, aOffset, bOffset, cOffset);
    }

  } else if (TransA == 'n') {

    // C = alpha*A*B**T + beta*C

      gemm_NoTransA(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta, aOffset, bOffset, cOffset);
  } else {

    // C = alpha*A**T*B**T + beta*C

      gemm_TransAB(A_mat, B_mat, C_mat, M, N, K, lda, ldb, ldc, alpha, beta, aOffset, bOffset, cOffset);
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

void axpy_AMP(long n, float alpha, Concurrency::array_view<float> &X, long incx, Concurrency::array_view<float> &Y, long incy)
{
  long size = (n + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
  Concurrency::extent<1> compute_domain(size);
  Concurrency::parallel_for_each(compute_domain.tile<BLOCK_SIZE>(),[=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
  {
    if(tidx.global[0] < n)
    {
      Y[tidx.global[0]] += X[tidx.global[0]] * alpha;
    }
  });
}
