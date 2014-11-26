#include "THCBlas.h"
#include "THCGeneral.h"
#include<iostream>


void THCudaBlas_init(int devices, int device)
{
    cl_int err;
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(mqueue);
        clReleaseContext(mcontext);
        return;
    }
}

void THCudaBlas_shutdown()
{
    /* Finalize work with clblas. */
    clblasTeardown();
    /* Release OpenCL working objects. */
    clReleaseCommandQueue(mqueue);
    clReleaseContext(mcontext);
}

void THCudaBlas_setHandle(int device)
{
  //current_handle = &handles[device];
}

void THCudaBlas_swap(long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_event event = NULL;
    cl_mem bufX, bufY;

    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    int lenX = 1 + (n-1)*abs(i_incx);
    int lenY = 1 + (n-1)*abs(i_incy);

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, (lenX*sizeof(cl_float)), NULL, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, (lenY*sizeof(cl_float)), NULL, &err);
    err = clEnqueueWriteBuffer(mqueue, bufX, CL_TRUE, 0, (lenX*sizeof(cl_float)), x, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufY, CL_TRUE, 0, (lenY*sizeof(cl_float)), y, 0, NULL, NULL);
    err = clblasSswap( i_n, bufX, 0, i_incx, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);

    if (err != CL_SUCCESS) 
    {
        printf("clblasSswap() failed with %d\n", err);
    }
    else
    {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
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

void THCudaBlas_scal(long n, float a, float *x, long incx)
{
  if(n == 1)
    incx = 1;

  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX;
    cl_event event = NULL;
    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    cl_float alpha = (cl_float)a;
    int lenX = 1 + (n-1)*abs(i_incx);

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, ( lenX * sizeof(float)), NULL, &err);
    err = clEnqueueWriteBuffer(mqueue, bufX, CL_TRUE, 0, (lenX * sizeof(float)), x, 0, NULL, NULL);

    err = clblasSscal( i_n, alpha, bufX, 0, i_incx, 1, &mqueue, 0, NULL,  &event);

    if (err != CL_SUCCESS) {
        printf("clblasSscal() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
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

void THCudaBlas_copy(long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    cl_int err;
    cl_mem bufX, bufY;
    cl_event event = NULL;
    size_t i_n = (size_t)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    int lenX = 1 + (n-1)*abs(i_incx);
    int lenY = 1 + (n-1)*abs(i_incy);

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, (lenX*sizeof(float)), NULL, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, (lenY*sizeof(float)), NULL, &err);

    err = clEnqueueWriteBuffer(mqueue, bufX, CL_TRUE, 0, (lenX*sizeof(float)), x, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufY, CL_TRUE, 0, (lenY*sizeof(float)), y, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasScopy( i_n, bufX, 0, i_incx, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasScopy() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
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

void THCudaBlas_axpy(long n, float a, float *x, long incx, float *y, long incy)
{
    if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
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
    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, (lenX*sizeof(float)), NULL, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, (lenY*sizeof(float)), NULL, &err);
    err = clEnqueueWriteBuffer(mqueue, bufX, CL_TRUE, 0, (lenX*sizeof(float)), x, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufY, CL_TRUE, 0, (lenY*sizeof(float)), y, 0, NULL, NULL);
    /* Call clblas function. */
    err = clblasSaxpy( i_n, alpha, bufX, 0, i_incx, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSaxpy() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
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

float THCudaBlas_dot(long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
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

    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, (lenX*sizeof(float)), NULL, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, (lenY*sizeof(float)), NULL, &err);
    // Allocate 1 element space for dotProduct
    bufDotP = clCreateBuffer(mcontext, CL_MEM_WRITE_ONLY, (sizeof(float)), NULL, &err);
    // Allocate minimum of N elements
    scratchBuff = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, (n*sizeof(float)), NULL, &err);

    err = clEnqueueWriteBuffer(mqueue, bufX, CL_TRUE, 0, (lenX*sizeof(float)), x, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufY, CL_TRUE, 0, (lenY*sizeof(float)), y, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasSdot( i_n, bufDotP, 0, bufX, 0, i_incx, bufY, 0, i_incy, scratchBuff, 1, &mqueue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSdot() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(mqueue, bufDotP, CL_TRUE, 0, sizeof(float), &result, 0, NULL, NULL);
    }
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufDotP);
    clReleaseMemObject(scratchBuff);

    //THCublasCheck(cublasSdot(*current_handle, i_n, x, i_incx, y, i_incy, &result));
    //cudaDeviceSynchronize();
    return result;
  }
  THError("Cublas_dot only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
  return -1;
}

/* Level 2 */
void THCudaBlas_gemv(char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{

  int transa_ = ((trans == 't') || (trans == 'T'));
  
 std::cout<<"\n CodeN"<<n<<std::endl;
 std::cout<<"\n CodeM"<<m<<std::endl;

 if(n == 1)
   lda = m;


  //cublasOperation_t op;
  clblasTranspose op;
  if (trans == 't')
  {
     std::cout<<"Trans"<<std::endl;
     op = clblasTrans;
  }
  else if (trans == 'n') 
  {
     std::cout<<"NoTrans"<<std::endl;
     op = clblasNoTrans;
  }
  else if (trans == 'c') 
  {
     std::cout<<"ConjTrans"<<std::endl;
     op = clblasConjTrans;
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
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    int lenM = 1 + (m-1)*abs(i_incx);
    int lenN = 1 + (n-1)*abs(i_incy);
    //int lenM = m;
    //int lenN = n;

   

    bufA = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, lenM * lenN * sizeof(*a), NULL, &err);
    bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, lenM * sizeof(*x), NULL, &err);
    bufY = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, lenN * sizeof(*y), NULL, &err);
    err = clEnqueueWriteBuffer(mqueue, bufA, CL_TRUE, 0, lenM * lenN * sizeof(*a), a, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufX, CL_TRUE, 0, lenM * sizeof(*x), x, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufY, CL_TRUE, 0, lenN * sizeof(*y), y, 0, NULL, NULL);

    /* Call clblas extended function. */
    err = clblasSgemv(order, op, i_m , i_n , alpha, bufA, 0, i_lda, bufX, 0, i_incx, beta, bufY, 0, i_incy, 1, &mqueue, 0, NULL, &event);

    if (err != CL_SUCCESS) {
        printf("clblasSgemvEx() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
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

void THCudaBlas_ger(long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
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
      bufA = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, (m * i_lda * sizeof(float)), NULL, &err);
      bufX = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, lenM * sizeof(float), NULL, &err);
      bufY = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, lenN * sizeof(float), NULL, &err);

      err = clEnqueueWriteBuffer(mqueue, bufA, CL_TRUE, 0, m * i_lda * sizeof(float), a, 0, NULL, NULL);
      err = clEnqueueWriteBuffer(mqueue, bufX, CL_TRUE, 0, lenM * sizeof(float), x, 0, NULL, NULL);
      err = clEnqueueWriteBuffer(mqueue, bufY, CL_TRUE, 0, lenN * sizeof(float), y, 0, NULL, NULL);

      /* Call clblas function. */
      err = clblasSger(order, i_m, i_n, alpha, bufX, 0, i_incx, bufY, 0, i_incy, bufA, 0, i_lda, 1, &mqueue, 0, NULL, &event);
      if (err != CL_SUCCESS) {
          printf("clblasSger() failed with %d\n", err);
      }
      else {
          /* Wait for calculations to be finished. */
          err = clWaitForEvents(1, &event);
          /* Fetch results of calculations from GPU memory. */
          err = clEnqueueReadBuffer(mqueue, bufA, CL_TRUE, 0, (m * i_lda * sizeof(float)), a, 0, NULL, NULL);
      }
      /* Release OpenCL memory objects. */
      clReleaseMemObject(bufY);
      clReleaseMemObject(bufX);
      clReleaseMemObject(bufA);


      //THCublasCheck(cublasSger(*current_handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_ger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

/* Level 3 */
void THCudaBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
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

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
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
    bufA = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, m * k * sizeof(*a), NULL, &err);
    bufB = clCreateBuffer(mcontext, CL_MEM_READ_ONLY, k * n * sizeof(*b), NULL, &err);
    bufC = clCreateBuffer(mcontext, CL_MEM_READ_WRITE, m * n * sizeof(*c), NULL, &err);
    err = clEnqueueWriteBuffer(mqueue, bufA, CL_TRUE, 0, m * k * sizeof(*a), a, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufB, CL_TRUE, 0, k * n * sizeof(*b), b, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(mqueue, bufC, CL_TRUE, 0, m * n * sizeof(*c), c, 0, NULL, NULL);
    /* Call clblas extended function. Perform gemm for the lower right sub-matrices */

    err = clblasSgemm(order, opa, opb, m, n, k, alpha, bufA, 0, i_lda, bufB, 0, i_ldb, beta, bufC, 0, i_ldc, 1, &mqueue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemmEx() failed with %d\n", err);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
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

