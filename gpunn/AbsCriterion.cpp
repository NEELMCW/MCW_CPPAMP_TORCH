#include <iostream>
#include <vector>
#include <numeric>
#include "common.h"
#include "amp_math.h"

struct abs_functor
{
  abs_functor() {}

  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    float z = x-y;
    return z >= 0 ? z : -z;
  }
};

static int gpunn_AbsCriterion_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  float sum;

  long size = THGPUTensor_nElement(input);
  THGPUTensor *input_orig = input;
  THGPUTensor *target_orig = target;
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, target, target_data);
  sum = bolt::amp::inner_product(input_data.begin(), input_data.end(), target_data.begin(), (float) 0, bolt::amp::plus<float>(), abs_functor());

  if(sizeAverage)
    sum /= size;
  if (input_orig != input) {
    THGPUTensor_free(input);
    input = NULL;
  }
  if (target_orig != target) {
    THGPUTensor_free(target);
    target = NULL;
  }

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct abs_updateGradInput_functor
{
  const float norm;

  abs_updateGradInput_functor(float norm_) restrict(amp,cpu): norm(norm_) {}

  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return (x - y) >= 0 ? norm : -norm;
  }
};

static int gpunn_AbsCriterion_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  long size = THGPUTensor_nElement(input);
  float norm = (sizeAverage ? 1./size : 1.);
  THGPUTensor *input_orig = input;
  THGPUTensor *target_orig = target;
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);
  THGPUTensor_resizeAs(gradInput, input);

  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, target, target_data, gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), target_data.begin(), gradInput_data.begin(), abs_updateGradInput_functor(norm));

  if (input_orig != input) {
    THGPUTensor_free(input);
    input = NULL;
  }
  if (target_orig != target) {
    THGPUTensor_free(target);
    target = NULL;
  }
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
