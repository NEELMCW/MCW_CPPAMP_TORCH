/**
 * Copyright 2014 Facebook
 */

#include<assert.h>

static const int NTHREADS = 32;

void gpunn_ClassNLLCriterion_updateOutput_kernel1(Concurrency::array_view<float,1> &avOutput,
                                                 Concurrency::array_view<float,1> &avInput,
                                                 Concurrency::array_view<float,1> &avTarget,
                                                 int ntarget)
{
  Concurrency::extent<1> grdExt(1);
  Concurrency::tiled_extent<1> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1> tidx) restrict(amp)
  {
    //assert(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
      assert(tidx.local[0] == 0);

    // TODO: T4951791 Reuse code between updateOutput_kernel1 and
    // updateOutput_kernel.
    // Verify whether `register` does anything here.
    register int i, t;
    for (i = 0; i < ntarget; i++) {
      t = avTarget[i] - 1;
      if (t >= 0)
        avOutput[0] = -avInput[t];
    }
  });
}

void gpunn_ClassNLLCriterion_updateOutput_kernel(Concurrency::array_view<float,1> &avOutput,
                                                Concurrency::array_view<float,1> &avInput,
                                                Concurrency::array_view<float,1> &avTarget,
                                                int nframe, int ndim,
                                                int sizeAverage, int ntarget)
{
  Concurrency::extent<1> grdExt(1 * 32);
  Concurrency::tiled_extent<32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<32> tidx) restrict(amp)
  {
    tile_static float shInputs[NTHREADS];
    // Verify whether `register` does anything here.
    register int i, j, t;

    shInputs[tidx.local[0]] = .0;
    for (i = tidx.local[0]; i < nframe; i += NTHREADS) {
      for (j = 0; j < ntarget; ++j) {
        t = (int)avTarget[i * ntarget + j] - 1;
        if (t >= 0)
          shInputs[tidx.local[0]] += avInput[i * ndim + t];
      }
    }
    tidx.barrier.wait();

    // TODO: T4951791 Reuse code between updateOutput_kernel1 and
    // updateOutput_kernel
    if (tidx.local[0] == 0) {
      avOutput[0] = .0;
      for (i = 0; i < NTHREADS; ++i)
        avOutput[0] += shInputs[i];
      if (sizeAverage)
        avOutput[0] /= nframe;
      avOutput[0] = -(avOutput[0]);
    }
  });
}

void gpunn_ClassNLLCriterion_updateGradInput_kernel(Concurrency::array_view<float,1> &avGradInput,
                                                   Concurrency::array_view<float,1> &avTarget,
                                                   int nframe, int ndim,
                                                   float grad, int ntarget)
{
  Concurrency::extent<1> grdExt(1 * 32);
  Concurrency::tiled_extent<32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<32> tidx) restrict(amp)
  {
    register int i, j, t;
    for (i = tidx.local[0]; i < nframe; i += NTHREADS) {
      for (j = 0; j < ntarget; ++j) {
        t = (int)avTarget[i * ntarget + j] - 1;
        if (t >= 0)
          avGradInput[i * ndim + t] = grad;
      }
    }
  });
}

static int gpunn_ClassNLLCriterion_updateOutput(lua_State *L) {
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  input = THGPUTensor_newContiguous(input);

  THGPUTensor *target = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  target = THGPUTensor_newContiguous(target);
  int ntarget = 1;
  if (target->nDimension > 1)
    ntarget = target->size[1];

  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "outputTensor", "torch.GPUTensor");
  output = THGPUTensor_newContiguous(output);

  PREPARE_AV(output, pavOutput);
  PREPARE_AV(input, pavInput);
  PREPARE_AV(target, pavTarget);
  if (input->nDimension == 1)
  {
    gpunn_ClassNLLCriterion_updateOutput_kernel1(*pavOutput, *pavInput, *pavTarget, ntarget);
  }
  else if (input->nDimension == 2)
  {
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    gpunn_ClassNLLCriterion_updateOutput_kernel (*pavOutput, *pavInput, *pavTarget,
                                                 input->size[0], input->size[1],
                                                 sizeAverage, ntarget);
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  THGPUTensor_free(output);
  THGPUTensor_free(target);
  THGPUTensor_free(input);

  return 1;
}

static int gpunn_ClassNLLCriterion_updateGradInput(lua_State *L)
{

  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  input = THGPUTensor_newContiguous(input);

  THGPUTensor *target = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  target = THGPUTensor_newContiguous(target);
  float *target_data = THGPUTensor_data(target);

  int ntarget = 1;
  if (target->nDimension > 1)
    ntarget = target->size[1];

  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata( L, 1, "gradInput", "torch.GPUTensor");
  gradInput = THGPUTensor_newContiguous(gradInput);
  float *gradInput_data = THGPUTensor_data(gradInput);

  float grad = -1.0;
  if (input->nDimension == 1)
  {
    if (ntarget > 1)
      THArgCheck(0, 2, "multi-target not implemented");
    float tid;

    tid = target_data[0];
    gradInput_data[(int)tid - 1] = grad;

  }
  else if (input->nDimension == 2)
  {
    int nframe = input->size[0];
    int ndim = input->size[1];
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    if (sizeAverage)
      grad /= nframe;

    PREPARE_AV(gradInput, pavGradInput);
    PREPARE_AV(target, pavTarget);
    gpunn_ClassNLLCriterion_updateGradInput_kernel(*pavGradInput, *pavTarget, nframe, ndim, grad, ntarget);
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  THGPUTensor_free(gradInput);
  THGPUTensor_free(target);
  THGPUTensor_free(input);

  return 1;
}

static const struct luaL_Reg gpunn_ClassNLLCriterion__[] = {
    {"ClassNLLCriterion_updateOutput", gpunn_ClassNLLCriterion_updateOutput},
    {"ClassNLLCriterion_updateGradInput",
     gpunn_ClassNLLCriterion_updateGradInput},
    {NULL, NULL}};

void gpunn_ClassNLLCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_ClassNLLCriterion__, "nn");
  lua_pop(L, 1);
}

