#include "THCBlas.h"
#include "THCGeneral.h"
#define OFFSET(N, incX) ((incX) > 0 ? 0 : ((N) - 1) * (-(incX)))
#define BLOCK_SIZE 256
#define TILE_DIM   16
#define THREADS    16
#define GEMM_BLOCK 256

// Matrix Multiplication with  A and B matrices  not transposed
static void gemm_NoTransAB(Concurrency::array_view<float, 1> &A, long aOffset,
                           Concurrency::array_view<float, 1> &B, long bOffset,
                           Concurrency::array_view<float, 1> &C, long cOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           float alpha, float beta)
{
  // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
  Concurrency::extent<2> grdExt((N + (THREADS - 1)) & ~(THREADS - 1),(M + (THREADS-1)) & ~(THREADS - 1));
  Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp)
  {
    float CValue = 0;
    int Row = tidx.tile[0] * TILE_DIM + tidx.local[0];
    int Col = tidx.tile[1] * TILE_DIM + tidx.local[1];
    tile_static float As[TILE_DIM][TILE_DIM];
    tile_static float Bs[TILE_DIM][TILE_DIM];
                
    for (int k = 0; k < (TILE_DIM + K - 1) / TILE_DIM; k++)
    { 
      // Read Matrix B from global to shared tile
      if (k * TILE_DIM + tidx.local[1] < K && Row < N)
        Bs[tidx.local[0]][tidx.local[1]] = B[bOffset + Row * K + k * TILE_DIM + tidx.local[1]];
      else
        Bs[tidx.local[0]][tidx.local[1]] = 0.0;

      // Read Matrix A from global to shared tile
      if (k*TILE_DIM + tidx.local[0] < K && Col < M)
        As[tidx.local[0]][tidx.local[1]] = A[aOffset + (k * TILE_DIM + tidx.local[0]) * M + Col];
      else
        As[tidx.local[0]][tidx.local[1]] = 0.0;

      // Wait until all shared memory gets filled
      tidx.barrier.wait();

      for (int n = 0; n < TILE_DIM; ++n)
        CValue += Bs[tidx.local[0]][n] * As[n][tidx.local[1]] * alpha;

      tidx.barrier.wait();
    }

    if (Row < N && Col < M)
    {
      C[cOffset + (tidx.global[0] * M) + tidx.global[1]] *= beta;
      C[cOffset + (tidx.global[0] * M) + tidx.global[1]] += CValue;
    }
  });
}

// Matrix Multiplication with  matrix A Transposed
static void gemm_NoTransB(Concurrency::array_view<float, 1> &A, long aOffset,
                          Concurrency::array_view<float, 1> &B, long bOffset,
                          Concurrency::array_view<float, 1> &C, long cOffset,
                          int M, int N, int K, int lda, int ldb, int ldc,
                          float alpha, float beta)
{
  // If K is small then make use of threads and blocks across M and N dimension
  if (K > N && K > M)
  {
    // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
    Concurrency::extent<2> grdExt(N, M * GEMM_BLOCK);
    Concurrency::tiled_extent<1, GEMM_BLOCK> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1, GEMM_BLOCK> tidx) restrict(amp)
    {
      int threadIdx = tidx.local[1];
      int blockIdx = tidx.tile[1];
      int Row = tidx.tile[0];
      int Col = blockIdx;

      tile_static float sh[GEMM_BLOCK];
      sh[threadIdx] = 0;

      for (int tileId = 0; tileId < ((K + GEMM_BLOCK - 1) & ~(GEMM_BLOCK - 1)) / GEMM_BLOCK; tileId++)
      {
        if (tileId * GEMM_BLOCK + threadIdx < K && Col < M && Row < N)
          sh[threadIdx] += A[aOffset + Col * K + tileId * GEMM_BLOCK + threadIdx] * B[bOffset + Row * K + tileId * GEMM_BLOCK + threadIdx];
      }
      tidx.barrier.wait();

      for (int stride = GEMM_BLOCK / 2; stride >= 1; stride /= 2)
      {
        if (threadIdx < stride)
          sh[threadIdx] += sh[threadIdx + stride];
        tidx.barrier.wait();
      }

      if(threadIdx == 0 && Col < M && Row < N)
      {
        C[cOffset + Row * M + Col] *= beta;
        C[cOffset + Row * M + Col] += sh[0] * alpha;
      }
    });
  }
  else
  {
    // Make grid dimension  an exact multiple of corresponding threadBlock dimension
    Concurrency::extent<2> grdExt((N + (THREADS - 1)) & ~(THREADS - 1), (M + (THREADS - 1)) & ~(THREADS - 1));
    Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);
    Concurrency::array_view<float,2> Cmat = C.view_as<2>(Concurrency::extent<2>(N, M));
    // Data in device is up-to-date no need to sync with host
    Cmat.discard_data();
    Concurrency::array_view<float,2> Amat = A.view_as<2>(Concurrency::extent<2>(M, K));
    // Data in device is up-to-date no need to sync with host
    Amat.discard_data();
    Concurrency::array_view<float,2> Bmat = B.view_as<2>(Concurrency::extent<2>(N, K));
    // Data in device is up-to-date no need to sync with host
    Bmat.discard_data();

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp)
    {
      float CValue = 0;
      int Row = tidx.global[0];
      int Col = tidx.global[1];
      tile_static float As[TILE_DIM][TILE_DIM];
      tile_static float Bs[TILE_DIM][TILE_DIM];

      for (int k = 0; k < ((K + (TILE_DIM - 1)) & ~(TILE_DIM - 1)) ; k += TILE_DIM)
      {
        // Read Matrix B from global to shared tile
        if (k + tidx.local[1] < K && Row < N)
          Bs[tidx.local[0]][tidx.local[1]] = Bmat[Row][bOffset + k + tidx.local[1]];
        else
          Bs[tidx.local[0]][tidx.local[1]] = 0.0;

        // Read Matrix A from global to shared tile
        if (k + tidx.local[1] < K && (tidx.tile[1] * TILE_DIM + tidx.local[0]) < M)
          As[tidx.local[0]][tidx.local[1]] = Amat[(tidx.tile[1] * TILE_DIM + tidx.local[0])] [aOffset + k + tidx.local[1]];
        else
          As[tidx.local[0]][tidx.local[1]] = 0.0;

        tidx.barrier.wait();

        // loop Unroll 
        CValue += Bs[tidx.local[0]][0] * As[tidx.local[1]][0] +
                  Bs[tidx.local[0]][1] * As[tidx.local[1]][1] +
                  Bs[tidx.local[0]][2] * As[tidx.local[1]][2] +
                  Bs[tidx.local[0]][3] * As[tidx.local[1]][3] +
                  Bs[tidx.local[0]][4] * As[tidx.local[1]][4] +
                  Bs[tidx.local[0]][5] * As[tidx.local[1]][5] +
                  Bs[tidx.local[0]][6] * As[tidx.local[1]][6] +
                  Bs[tidx.local[0]][7] * As[tidx.local[1]][7] +
                  Bs[tidx.local[0]][8] * As[tidx.local[1]][8] +
                  Bs[tidx.local[0]][9] * As[tidx.local[1]][9] +
                  Bs[tidx.local[0]][10] * As[tidx.local[1]][10] +
                  Bs[tidx.local[0]][11] * As[tidx.local[1]][11] +
                  Bs[tidx.local[0]][12] * As[tidx.local[1]][12] +
                  Bs[tidx.local[0]][13] * As[tidx.local[1]][13] +
                  Bs[tidx.local[0]][14] * As[tidx.local[1]][14] +
                  Bs[tidx.local[0]][15] * As[tidx.local[1]][15];

        tidx.barrier.wait();
      }

      if (Row < N && Col < M)
      {
        Cmat[Row][cOffset + Col] *= beta;
        Cmat[Row][cOffset + Col] += CValue * alpha;
      }
    });
  }
}

// Matrix Multiplication when Matrix B is transposed
static void gemm_NoTransA(Concurrency::array_view<float, 1> &A, long aOffset,
                          Concurrency::array_view<float, 1> &B, long bOffset,
                          Concurrency::array_view<float, 1> &C, long cOffset,
                          int M, int N, int K, int lda, int ldb, int ldc,
                          float alpha, float beta)
{
  // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
  Concurrency::extent<2> grdExt((N + (THREADS - 1)) & ~(THREADS - 1), (M + (THREADS - 1)) & ~(THREADS - 1));
  Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp)
  {
    float CValue = 0;
    int Row = tidx.tile[0] * TILE_DIM + tidx.local[0];
    int Col = tidx.tile[1] * TILE_DIM + tidx.local[1];
    tile_static float As[TILE_DIM][TILE_DIM];
    tile_static float Bs[TILE_DIM][TILE_DIM];
    for (int k = 0; k < (TILE_DIM + K - 1) / TILE_DIM; k++)
    {
      // Read Matrix B from global to shared tile
      if (k * TILE_DIM + tidx.local[0] < K && (tidx.tile[0] * TILE_DIM + tidx.local[1]) < N)
        Bs[tidx.local[0]][tidx.local[1]] = B[bOffset + (k * TILE_DIM + tidx.local[0]) * N + (tidx.tile[0] * TILE_DIM + tidx.local[1])];
      else
        Bs[tidx.local[0]][tidx.local[1]] = 0.0;

      // Read Matrix A from global to shared tile
      if (k*TILE_DIM + tidx.local[0] < K && Col < M)
        As[tidx.local[0]][tidx.local[1]] = A[aOffset + (k * TILE_DIM + tidx.local[0]) * M + Col];
      else
        As[tidx.local[0]][tidx.local[1]] = 0.0;

      tidx.barrier.wait();

      for (int n = 0; n < TILE_DIM; ++n)
        CValue += Bs[n][tidx.local[0]] * As[n][tidx.local[1]];

      tidx.barrier.wait();
    }

    if (Row < N && Col < M)
    {
      C[cOffset + (Row * M)+Col] *= beta;
      C[cOffset + (Row * M)+Col] += CValue * alpha;
    }
  });
}

// Matrix Multiplication when both A and B are transposed
static void gemm_TransAB(Concurrency::array_view<float, 1> &A, long aOffset,
                         Concurrency::array_view<float, 1> &B, long bOffset,
                         Concurrency::array_view<float, 1> &C, long cOffset,
                         int M, int N, int K, long lda, long ldb, long ldc,
                         float alpha, float beta)
{
  // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
  Concurrency::extent<2> grdExt((N + (THREADS - 1)) & ~(THREADS - 1), (M + (THREADS - 1)) & ~(THREADS - 1));
  Concurrency::tiled_extent<THREADS, THREADS> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<THREADS, THREADS> tidx) restrict(amp)
  {
    float temp;
    int j = tidx.global[0];
    int i = tidx.global[1];
    if(i < M && j < N)
    {
      temp = 0;
      for (int l = 0; l < K; ++l)
        temp += A[aOffset + l + i * lda] * B[bOffset + j + l * ldb];

      C[cOffset + i + j * ldc] = alpha * temp + beta * C[cOffset + i + j * ldc];
    }
  });
}

// API used in torch to invoke AMP gemm operation
void THGPUBlas_gemm(char TransA, char TransB, const long M, const long N, const long K, const float alpha,
             Concurrency::array_view<float> &A_mat, long aOffset, long lda,
             Concurrency::array_view<float> &B_mat, long bOffset, long ldb, const float beta,
             Concurrency::array_view<float> &C_mat, long cOffset, long ldc)
{
  int i, j;

  // Quick return if possible
  if (!M || !N || ((alpha == 0 || !K) && beta == 1)) 
    return ;
  // For alpha = 0
  if (alpha == 0)
  {
    if (beta == 0)
    {
      for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i)
          C_mat[i + j * ldc] = 0;
    }
    else
    {
      for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i)
          C_mat[i + j * ldc] *= beta;
    }
    return ;
  }

  if (TransB == 'n')
  {
    if (TransA == 'n')
      gemm_NoTransAB(A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
    else
      gemm_NoTransB(A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  }
  else if (TransA == 'n')
    gemm_NoTransA(A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  else
    gemm_TransAB(A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
}

// Matrix Vector Multiplication where the Matrix A is transposed
static void gemv_TransA(Concurrency::array_view<float> &A_mat, int aOffset,
                        Concurrency::array_view<float> &X_vec, long xOffset,
                        Concurrency::array_view<float> &Y_vec, long yOffset,
                        float alpha, float beta, int lenX, int lenY,
                        Concurrency::array_view<float> &tempBuf)
{
  // Case where Y vector's length (lenY) is very small compared to lenX
  // TO DO: Need to represent this case in a better way
  // The parameter tempBuf is used in this case to make a global sychronization across threadblocks
  if((lenX - lenY) > 5000)
  {
    // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
    int len_X = (lenX + (BLOCK_SIZE - 1)) & ~(BLOCK_SIZE - 1);
    int num_blocks = len_X / BLOCK_SIZE;
    Concurrency::extent<1> grdExt(len_X);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext,[=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
      tile_static float t[BLOCK_SIZE];
      for (int Col = 0; Col < lenY; Col++)
      {
        int blockIdx = tidx.tile[0];
        int threadIdx = tidx.local[0];
        tempBuf[Col * num_blocks + blockIdx] = 0;
        t[threadIdx] = 0;

        if (Col < lenY && blockIdx * BLOCK_SIZE + threadIdx < lenX)
          t[threadIdx] = X_vec[xOffset + blockIdx * BLOCK_SIZE + threadIdx] * A_mat[aOffset + Col * lenX + blockIdx * BLOCK_SIZE + threadIdx];

        tidx.barrier.wait();

        for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
        {
          if(threadIdx < stride)
            t[threadIdx] += t[threadIdx + stride];
        }
        tempBuf[Col * num_blocks + blockIdx] = t[0];
        tidx.barrier.wait();
      }

      if (tidx.tile[0] == 0)
      {
        for(int Col = 0; Col < lenY; Col++)
        {
          tile_static float sh[BLOCK_SIZE];
          int threadId = tidx.local[0];
          sh[tidx.local[0]] = 0;

          for (int i = threadId; i < num_blocks; i += tidx.tile_dim0)
            sh[threadId] += tempBuf[Col * num_blocks + i];

          tidx.barrier.wait();

          for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
          {
            if(threadId < stride)
              sh[threadId] += sh[threadId + stride];
          }
          tidx.barrier.wait();
          Y_vec[yOffset + Col] *= beta;
          Y_vec[yOffset + Col] += alpha * sh[0];
        }
      }
    });
  }
  else
  {
    // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
    Concurrency::extent<1> grdExt(lenY * BLOCK_SIZE);
    Concurrency::tiled_extent<BLOCK_SIZE> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
    {
      int threadIdx = tidx.local[0];
      int blockIdx = tidx.tile[0];
      int Col = blockIdx;

      tile_static float sh[BLOCK_SIZE];
      sh[threadIdx] = 0;

      for (int tileId = 0; tileId < ((lenX + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1)) / BLOCK_SIZE; tileId++)
      {
        if (tileId * BLOCK_SIZE + threadIdx < lenX && Col < lenY)
          sh[threadIdx] += X_vec[xOffset + tileId * BLOCK_SIZE + threadIdx] * A_mat[aOffset + Col * lenX + tileId * BLOCK_SIZE + threadIdx];
      }
      tidx.barrier.wait();

      for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
      {
        if (threadIdx < stride)
          sh[threadIdx] += sh[threadIdx + stride];
        tidx.barrier.wait();
      }

      if(threadIdx == 0 && Col < lenY)
      {
        Y_vec[yOffset + Col] *= beta;
        Y_vec[yOffset + Col] += alpha * sh[0];
      }
    });
  }
}

static void gemv_NoTransA(Concurrency::array_view<float> &A, long aOffset,
                          Concurrency::array_view<float> &X, long xOffset,
                          Concurrency::array_view<float> &Y, long yOffset,
                          float alpha, float beta, int lenX, int lenY)
{
  // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
  long size = (lenY + 255) & ~255;
  Concurrency::extent<1> compute_domain(size);

  Concurrency::parallel_for_each(compute_domain.tile<BLOCK_SIZE>(),[=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
  {
    int bx = tidx.tile[0];
    int tx = tidx.local[0];
    tile_static float Xds[BLOCK_SIZE];
    int Col = bx * BLOCK_SIZE + tx;
    float Pvalue = 0;

    for (int m = 0; m < (lenX - 1) / BLOCK_SIZE + 1; ++m)
    {
      if (m * BLOCK_SIZE + tx < lenX)
        Xds[tx] = X[xOffset + m * BLOCK_SIZE + tx];
      else
        Xds[tx]=0;

      tidx.barrier.wait();

      for (int k = 0; k < BLOCK_SIZE; k++)
        if (Col < lenY && m * BLOCK_SIZE + k < lenX)
          Pvalue += Xds[k] * A[aOffset + Col + (m * BLOCK_SIZE + k) * lenY];

      tidx.barrier.wait();
    }
    if (Col < lenY)
    {
      Y[yOffset + Col] *= beta;
      Y[yOffset + Col] += alpha * Pvalue;
    }
  });
}

// API used in torch to invoke matrix vector multiplication
void THGPUBlas_gemv(char TransA, long M, long N, float alpha,
              Concurrency::array_view<float> &A, long aOffset,
              Concurrency::array_view<float> &X, long xOffset, long incX, float beta,
              Concurrency::array_view<float> &Y, long yOffset, long incY,
              Concurrency::array_view<float> &temp_buf)
{
  if (alpha == 0.0)
    return;

  int lenX, lenY;
  if (M == 0 || N == 0)
    return;

  if (alpha == 0.0 && beta == 1.0)
    return;

  if (TransA == 'n')
  {
    lenX = N;
    lenY = M;
  }
  else
  {
    lenX = M;
    lenY = N;
  }

  if (TransA == 't')
    gemv_TransA(A, aOffset, X, xOffset, Y, yOffset, alpha, beta, lenX, lenY, temp_buf);
  else if (TransA == 'n')
    gemv_NoTransA(A, aOffset, X, xOffset, Y, yOffset, alpha, beta, lenX, lenY);
}

// Scale vecotr X and add to vectory Y
void THGPUBlas_axpy(long n, float alpha,
              Concurrency::array_view<float> &X, long xOffset, long incx,
              Concurrency::array_view<float> &Y, long yOffset, long incy)
{
  // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
  long size = (n + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
  Concurrency::extent<1> compute_domain(size);

  Concurrency::parallel_for_each(compute_domain.tile<BLOCK_SIZE>(),[=] (Concurrency::tiled_index<BLOCK_SIZE> tidx) restrict(amp)
  {
    if(tidx.global[0] < n)
      Y[yOffset + tidx.global[0]] += X[xOffset + tidx.global[0]] * alpha;
  });
}

// Single Precision General matrix rank 1 operation
void THGPUBlas_ger(long m, long n, float alpha,
             Concurrency::array_view<float> &x, long xOffset, long incx,
             Concurrency::array_view<float> &y, long yOffset, long incy,
             Concurrency::array_view<float> &a, long aOffset, long lda)
{
  // Make grid size in each dimension an exact multiple of threadblock size in the corresponding dimension
  long M = (m + 15) & ~15;
  long N = (n + 15) & ~15;
  Concurrency::extent<2> compute_domain(M, N);
  Concurrency::parallel_for_each(compute_domain.tile<16, 16>(),[=] (Concurrency::tiled_index<16, 16> tidx) restrict(amp)
  {
    int i = tidx.global[0];
    int j = tidx.global[1];
    if(i < m && j < n)
      a[aOffset + j*m + i] += x[xOffset + i] * y[yOffset + j] * alpha;
  });
}
