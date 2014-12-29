#include "THCBlas.h"
#include "THCGeneral.h"
#include<iostream>

void THGPUBlas_init(int devices, int device)
{
  cl_int err;
  err = clblasSetup();
  if (err != CL_SUCCESS)
  {
    printf("clblasSetup() failed with %d\n", err);
    return;
  }
}

void THGPUBlas_shutdown()
{
  /* Finalize work with clblas. */
  clblasTeardown();
  /* Release OpenCL working objects. */
}

void THGPUBlas_setHandle(int device)
{
  //current_handle = &handles[device];
}

void THGPUBlas_swap(long n, float *x, long incx, float *y, long incy)
{
  if (n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if ( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_event event = NULL;
    cl_mem bufX, bufY;

    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    int lenX = 1 + (n-1)*abs(i_incx);
    int lenY = 1 + (n-1)*abs(i_incy);

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (lenX*sizeof(cl_float)), x, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (lenY*sizeof(cl_float)), y, &err);
    err = clblasSswap( i_n, bufX, 0, i_incx, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);

    if (err != CL_SUCCESS) 
    {
      printf("clblasSswap() failed with %d\n", err);
    }
    else
    {
      /* Fetch results of calculations from GPU memory. */
      err = clEnqueueReadBuffer(mqueue, bufX, CL_TRUE, 0, (lenX*sizeof(float)), x, 0, NULL, NULL);
      err = clEnqueueReadBuffer(mqueue, bufY, CL_TRUE, 0, (lenY*sizeof(float)), y, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);

    //THCublasCheck(cublasSswap(*current_handle, i_n, x, i_incx, y, i_incy));
    return;
  }
  THError("Cublas_swap only supports n, incx and"
          " incy upto signed integer limits: %d", INT_MAX);
}

void THGPUBlas_scal(long n, float a, float *x, long incx)
{
  if (n == 1)
    incx = 1;

  if ( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX;
    cl_event event = NULL;
    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    cl_float alpha = (cl_float)a;
    int lenX = 1 + (n-1)*abs(i_incx);

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, ( lenX * sizeof(float)), x, &err);

    err = clblasSscal( i_n, alpha, bufX, 0, i_incx, 1, &mqueue, 0, NULL,  &event);

    if (err != CL_SUCCESS)
    {
      printf("clblasSscal() failed with %d\n", err);
    }
    else
    {
      /* Fetch results of calculations from GPU memory. */
      err = clEnqueueReadBuffer(mqueue, bufX, CL_TRUE, 0, (lenX*sizeof(float)), x, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufX);
    //THCublasCheck(cublasSscal(*current_handle, i_n, &a, x, i_incx));
    return;
  }
  THError("Cublas_scal only supports n and incx "
          "upto signed integer limits: %d", INT_MAX);
}

void THGPUBlas_copy(long n, float *x, long incx, float *y, long incy)
{
  if (n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if ( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX, bufY;
    cl_event event = NULL;
    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    int lenX = 1 + (n-1)*abs(i_incx);
    int lenY = 1 + (n-1)*abs(i_incy);

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, (lenX*sizeof(float)), x, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (lenY*sizeof(float)), y, &err);


    /* Call clblas function. */
    err = clblasScopy( i_n, bufX, 0, i_incx, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
      printf("clblasScopy() failed with %d\n", err);
    }
    else
    {
      /* Fetch results of calculations from GPU memory. */
      err = clEnqueueReadBuffer(mqueue, bufY, CL_TRUE, 0, (lenY*sizeof(float)), y, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);

    //THCublasCheck(cublasScopy(*current_handle, i_n, x, i_incx, y, i_incy));
    return;
  }

  THError("Cublas_copy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
}

void THGPUBlas_axpy(long n, float a, float *x, long incx, float *y, long incy)
{
  if (n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if ( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX, bufY;
    cl_event event = NULL;
    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    int lenX = 1 + (n-1)*abs(i_incx);
    int lenY = 1 + (n-1)*abs(i_incy);
    cl_float alpha = (cl_float)a;

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, (lenX*sizeof(float)), x, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (lenY*sizeof(float)), y, &err);
    /* Call clblas function. */
    err = clblasSaxpy( i_n, alpha, bufX, 0, i_incx, bufY, 0, i_incy, 1, &mqueue, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("clblasSaxpy() failed with %d\n", err);
    }
    else
    {
        err = clEnqueueReadBuffer(mqueue, bufY, CL_TRUE, 0, (lenY*sizeof(float)), y, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);

    //THCublasCheck(cublasSaxpy(*current_handle, i_n, &a, x, i_incx, y, i_incy));
    return;
  }

  THError("Cublas_axpy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
}

float THGPUBlas_dot(long n, float *x, long incx, float *y, long incy)
{
  if (n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if ( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX, bufY, bufDotP, scratchBuff;
    cl_event event = NULL;
    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    int lenX = 1 + (n-1)*abs(i_incx);
    int lenY = 1 + (n-1)*abs(i_incy);

    float result;

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, (lenX*sizeof(float)), x, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, (lenY*sizeof(float)), y, &err);
    // Allocate 1 element space for dotProduct
    bufDotP = clCreateBuffer(mcontext, CL_MEM_WRITE_ONLY, (sizeof(float)), NULL, &err);
    // Allocate minimum of N elements
    scratchBuff = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, (n*sizeof(float)), NULL, &err);


    /* Call clblas function. */
    err = clblasSdot( i_n, bufDotP, 0, bufX, 0, i_incx, bufY, 0, i_incy, scratchBuff, 1, &mqueue, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
      printf("clblasSdot() failed with %d\n", err);
    }
    else
    {
      /* Wait for calculations to be finished. */
      /* Fetch results of calculations from GPU memory. */
      err = clEnqueueReadBuffer(mqueue, bufDotP, CL_TRUE, 0, sizeof(float), &result, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufDotP);
    clReleaseMemObject(scratchBuff);

    //THCublasCheck(cublasSdot(*current_handle, i_n, x, i_incx, y, i_incy, &result));
    //gpuDeviceSynchronize();
    return result;
  }
  THError("Cublas_dot only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
  return -1;
}

/* Level 2 */
void THGPUBlas_gemv(char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{

  int transa_ = ((trans == 't') || (trans == 'T'));

  if (n == 1)
    lda = m;


  int i_incx = (int)incx;
  int i_incy = (int)incy;
  int lenM, lenN;
  //cublasOperation_t op;
  clblasTranspose op;
  if (trans == 't')
  {
    op = clblasTrans;
    lenM = 1 + (m-1)*abs(i_incx);
    lenN = 1 + (n-1)*abs(i_incy);
  }
  else if (trans == 'n')
  {
    op = clblasNoTrans;
    lenM = 1 + (n-1)*abs(i_incx);
    lenN = 1 + (m-1)*abs(i_incy);
  }
  else if (trans == 'c')
  {
    op = clblasConjTrans;
    lenM = 1 + (n-1)*abs(i_incx);
    lenN = 1 + (m-1)*abs(i_incy);
  }
  clblasOrder order = clblasColumnMajor;



  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX, bufY, bufA;
    cl_event event = NULL;

    size_t i_m = (size_t)m;
    size_t i_lda = (size_t)lda;

    size_t i_n = (size_t)n;
    //int lenM = m;
    //int lenN = n;

    bufA = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, lenM * lenN * sizeof(*a), a, &err);
    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, lenM * sizeof(*x), x, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, lenN * sizeof(*y), y, &err);

    /* Call clblas extended function. */
    err = clblasSgemv(order, op, i_m , i_n , alpha, bufA, 0, i_lda, bufX, 0, i_incx, beta, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);

    if (err != CL_SUCCESS)
    {
      printf("clblasSgemvEx() failed with %d\n", err);
    }
    else
    {
      /* Fetch results of calculations from GPU memory. */
      err = clEnqueueReadBuffer(mqueue, bufY, CL_TRUE, 0, lenN * sizeof(*y), y, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufA);

    //THCublasCheck(cublasSgemv(*current_handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_gemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

/* Level 2 */
void THGPUBlas_gemv_opt(char trans, long m, long n, float alpha, 
  float *a, long lda, float *x, long incx, float beta, float *y, long incy,
  void* cl_A, void* cl_X, void* cl_Y)
{

  int transa_ = ((trans == 't') || (trans == 'T'));

  if (n == 1)
    lda = m;


  int i_incx = (int)incx;
  int i_incy = (int)incy;
  int lenM, lenN;
  //cublasOperation_t op;
  clblasTranspose op;
  if (trans == 't')
  {
    op = clblasTrans;
    lenM = 1 + (m-1)*abs(i_incx);
    lenN = 1 + (n-1)*abs(i_incy);
  }
  else if (trans == 'n')
  {
    op = clblasNoTrans;
    lenM = 1 + (n-1)*abs(i_incx);
    lenN = 1 + (m-1)*abs(i_incy);
  }
  else if (trans == 'c')
  {
    op = clblasConjTrans;
    lenM = 1 + (n-1)*abs(i_incx);
    lenN = 1 + (m-1)*abs(i_incy);
  }
  clblasOrder order = clblasColumnMajor;



  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX, bufY, bufA;
    cl_event event = NULL;

    size_t i_m = (size_t)m;
    size_t i_lda = (size_t)lda;

    size_t i_n = (size_t)n;
    //int lenM = m;
    //int lenN = n;
   if (cl_A == NULL)
      bufA = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, lenM * lenN * sizeof(*a), a, &err);
   else
     bufA = static_cast<cl_mem>(cl_A);
   if (cl_X == NULL)
     bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, lenM * sizeof(*x), x, &err);
   else
    bufX = static_cast<cl_mem>(cl_X);
   if (cl_Y == NULL)
     bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, lenN * sizeof(*y), y, &err);
   else
      bufY = static_cast<cl_mem>(cl_Y);

    /* Call clblas extended function. */
    err = clblasSgemv(order, op, i_m , i_n , alpha, bufA, 0, i_lda, bufX, 0, i_incx, beta, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);

    if (err != CL_SUCCESS)
    {
      printf("clblasSgemvEx() failed with %d\n", err);
    }
    else
    {
      /* Fetch results of calculations from GPU memory. */
      err = clEnqueueReadBuffer(mqueue, bufY, CL_TRUE, 0, lenN * sizeof(*y), y, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    if (cl_Y == NULL)
      clReleaseMemObject(bufY);
    if (cl_X == NULL)
      clReleaseMemObject(bufX);
    if (cl_A == NULL)
      clReleaseMemObject(bufA);

    //THCublasCheck(cublasSgemv(*current_handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_gemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THGPUBlas_ger(long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
{
  if (n == 1)
    lda = m;

  if ( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    size_t i_m = (size_t)m;
    size_t i_n = (size_t)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cl_int err;
    cl_mem bufX, bufY, bufA;
    cl_event event = NULL;
    clblasOrder order = clblasColumnMajor;

    int lenM = 1 + (m-1)*abs(i_incx);
    int lenN = 1 + (n-1)*abs(i_incy);

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (m * i_lda * sizeof(float)), a, &err);
    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, lenM * sizeof(float), x, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, lenN * sizeof(float), y, &err);


    /* Call clblas function. */
    err = clblasSger(order, i_m, i_n, alpha, bufX, 0, i_incx, bufY, 0, i_incy, bufA, 0, i_lda, 1, &mqueue, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
      printf("clblasSger() failed with %d\n", err);
    }
    else
    {
      /* Fetch results of calculations from GPU memory. */
      err = clEnqueueReadBuffer(mqueue, bufA, CL_TRUE, 0, (m * i_lda * sizeof(float)), a, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufA);

    return;
  }
  THError("Cublas_ger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

/* Level 3 */
void THGPUBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if (n == 1)
    ldc = m;

  if (transa_)
  {
    if (m == 1)
      lda = k;
  }
  else
  {
    if (k == 1)
      lda = m;
  }

  if (transb_)
  {
    if (k == 1)
      ldb = n;
  }
  else
  {
    if (n == 1)
      ldb = k;
  }

  //cublasOperation_t opa;
  //if (transa == 't') opa = CUBLAS_OP_T;
  //else if (transa == 'n') opa = CUBLAS_OP_N;
  //else if (transa == 'c') opa = CUBLAS_OP_C;
  //else THError("transa must be one of: t, n, c");

  //cublasOperation_t opb;
  //if (transb == 't') opb = CUBLAS_OP_T;
  //else if (transb == 'n') opb = CUBLAS_OP_N;
  //else if (transb == 'c') opb = CUBLAS_OP_C;
  //else THError("transb must be one of: t, n, c");
  clblasTranspose opa;
  if (transa == 't') opa = clblasTrans;
  else if (transa == 'n') opa = clblasNoTrans;
  else if (transa == 'c') opa = clblasConjTrans;

  clblasTranspose opb;
  if (transb == 't') opb = clblasTrans;
  else if (transb == 'n') opb = clblasNoTrans;
  else if (transb == 'c') opb = clblasConjTrans;

  if ( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    size_t i_m = (size_t)m;
    size_t i_n = (size_t)n;
    size_t i_k = (size_t)k;

    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cl_int err;
    cl_mem bufC, bufB, bufA;
    cl_event event = NULL;
    clblasOrder order = clblasColumnMajor;


    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, m * k * sizeof(*a),  a, &err);
    bufB = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, k * n * sizeof(*b),  b, &err);
    bufC = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR , m * n * sizeof(*c), c, &err);

    err = clblasSgemm(order, opa, opb, m, n, k, alpha, bufA, 0, i_lda, bufB, 0, i_ldb, beta, bufC, 0, i_ldc, 1, &mqueue, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("clblasSgemmEx() failed with %d\n", err);
    }
    else
    {
      err = clEnqueueReadBuffer(mqueue, bufC, CL_TRUE, 0, m * n * sizeof(*c), c, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);

    //THCublasCheck(cublasSgemm(*current_handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

void* THGPUBlas_clCreateBuffer(long m, long k, float* a) {
  return (void*)clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, m * k * sizeof(*a),  a, NULL);
}

/* Level 3 optimized. bufA, bufB are created outside as m,n,k is not changed in loops*/
void THGPUBlas_gemm_opt(char transa, char transb,
  long m, long n, long k, float alpha,
  float *a, long lda, float *b, long ldb, float beta,
  float *c, long ldc,
  void* cl_A, void* cl_B, void* cl_C)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if (n == 1)
    ldc = m;

  if (transa_)
  {
    if (m == 1)
      lda = k;
  }
  else
  {
    if (k == 1)
      lda = m;
  }

  if (transb_)
  {
    if (k == 1)
      ldb = n;
  }
  else
  {
    if (n == 1)
      ldb = k;
  }

  //cublasOperation_t opa;
  //if (transa == 't') opa = CUBLAS_OP_T;
  //else if (transa == 'n') opa = CUBLAS_OP_N;
  //else if (transa == 'c') opa = CUBLAS_OP_C;
  //else THError("transa must be one of: t, n, c");

  //cublasOperation_t opb;
  //if (transb == 't') opb = CUBLAS_OP_T;
  //else if (transb == 'n') opb = CUBLAS_OP_N;
  //else if (transb == 'c') opb = CUBLAS_OP_C;
  //else THError("transb must be one of: t, n, c");
  clblasTranspose opa;
  if (transa == 't') opa = clblasTrans;
  else if (transa == 'n') opa = clblasNoTrans;
  else if (transa == 'c') opa = clblasConjTrans;

  clblasTranspose opb;
  if (transb == 't') opb = clblasTrans;
  else if (transb == 'n') opb = clblasNoTrans;
  else if (transb == 'c') opb = clblasConjTrans;

  if ( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    size_t i_m = (size_t)m;
    size_t i_n = (size_t)n;
    size_t i_k = (size_t)k;

    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cl_int err;
    cl_mem bufC, bufB, bufA;
    cl_event event = NULL;
    clblasOrder order = clblasColumnMajor;


    /* Prepare OpenCL memory objects and place matrices inside them. */
    if (cl_A == NULL)
      bufA = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, m * k * sizeof(*a),  a, &err);
    else
      bufA = static_cast<cl_mem>(cl_A);
    if (cl_B == NULL)
      bufB = clCreateBuffer(mcontext, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, k * n * sizeof(*b),  b, &err); 
    else
      bufB = static_cast<cl_mem>(cl_B);
    if (cl_C == NULL)
      bufC = clCreateBuffer(mcontext, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR , m * n * sizeof(*c), c, &err);
    else
      bufC = static_cast<cl_mem>(cl_C);

    err = clblasSgemm(order, opa, opb, m, n, k, alpha, bufA, 0, i_lda, bufB, 0, i_ldb, beta, bufC, 0, i_ldc, 1, &mqueue, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("clblasSgemmEx() failed with %d\n", err);
    }
    else
    {
      err = clEnqueueReadBuffer(mqueue, bufC, CL_TRUE, 0, m * n * sizeof(*c), c, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    if (cl_C == NULL)
      clReleaseMemObject(bufC);
    if (cl_B == NULL)
      clReleaseMemObject(bufB);
    if (cl_A == NULL)
      clReleaseMemObject(bufA);

    //THCublasCheck(cublasSgemm(*current_handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

