#include <numeric>
#include "amp_math.h"
#include "THCBolt.h"

static int gpunn_DistKLDivCriterion_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  luaL_argcheck(L, THGPUTensor_nElement(input) == THGPUTensor_nElement(target), 2,
                "input and target need to have the same number of elements");

  float sum;
  long size = THGPUTensor_nElement(input);
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  sum = boltInnerProduct_plus_kl(input, target);

  if (sizeAverage)
    sum /= size;

  THGPUTensor_free(input);
  THGPUTensor_free(target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");
  lua_pushnumber(L, sum);
  return 1;
}

static int gpunn_DistKLDivCriterion_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  luaL_argcheck(L, THGPUTensor_nElement(input) == THGPUTensor_nElement(target), 2,
                "input and target need to have the same number of elements");

  long size = THGPUTensor_nElement(input);
  float norm = (sizeAverage ? 2./size : 2.);
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);
  THGPUTensor_resizeAs(gradInput, input);

  boltTransform_kl(input, target, gradInput, norm);

  THGPUTensor_free(input);
  THGPUTensor_free(target);
  return 1;
}

static const struct luaL_Reg gpunn_DistKLDivCriterion__ [] = {
  {"DistKLDivCriterion_updateOutput", gpunn_DistKLDivCriterion_updateOutput},
  {"DistKLDivCriterion_updateGradInput", gpunn_DistKLDivCriterion_updateGradInput},
  {NULL, NULL}
};

static void gpunn_DistKLDivCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_DistKLDivCriterion__, "nn");
  lua_pop(L,1);
}
