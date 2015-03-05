#include <iostream>
#include <vector>
#include <numeric>
#include "amp_math.h"
#include "THCBolt.h"

static int gpunn_AbsCriterion_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  float sum;
  long size = THGPUTensor_nElement(input);
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  sum = boltInnerProduct_plus_abs(input, target);

  if(sizeAverage)
    sum /= size;

  THGPUTensor_free(input);
  THGPUTensor_free(target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");
  lua_pushnumber(L, sum);
  return 1;
}

static int gpunn_AbsCriterion_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  long size = THGPUTensor_nElement(input);
  float norm = (sizeAverage ? 1./size : 1.);
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);
  THGPUTensor_resizeAs(gradInput, input);

  boltTransform_abs(input, target, gradInput, norm);

  THGPUTensor_free(input);
  THGPUTensor_free(target);
  return 1;
}

static const struct luaL_Reg gpunn_AbsCriterion__ [] = {
  {"AbsCriterion_updateOutput", gpunn_AbsCriterion_updateOutput},
  {"AbsCriterion_updateGradInput", gpunn_AbsCriterion_updateGradInput},
  {NULL, NULL}
};

static void gpunn_AbsCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_AbsCriterion__, "nn");
  lua_pop(L,1);
}
