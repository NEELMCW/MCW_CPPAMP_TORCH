#include <iostream>
#include <vector>
#include <numeric>
#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
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

static int cunn_AbsCriterion_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  float sum;

  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);
  target = THCudaTensor_newContiguous(target);

  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input) + THCudaTensor_nElement(input));
  bolt::amp::device_vector<float> target_data(THCudaTensor_data(target), THCudaTensor_data(target) + THCudaTensor_nElement(target));
  sum = bolt::amp::inner_product(input_data.begin(), input_data.end(), target_data.begin(), (float) 0, bolt::amp::plus<float>(), abs_functor());

  if(sizeAverage)
    sum /= size;

  THCudaTensor_free(input);
  THCudaTensor_free(target);

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

static int cunn_AbsCriterion_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);
  float norm = (sizeAverage ? 1./size : 1.);
  input = THCudaTensor_newContiguous(input);
  target = THCudaTensor_newContiguous(target);
  THCudaTensor_resizeAs(gradInput, input);

  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input) + THCudaTensor_nElement(input));
  bolt::amp::device_vector<float> target_data(THCudaTensor_data(target), THCudaTensor_data(target) + THCudaTensor_nElement(target));
  bolt::amp::device_vector<float> gradInput_data(THCudaTensor_data(gradInput), THCudaTensor_data(gradInput) + THCudaTensor_nElement(gradInput));

  bolt::amp::transform(input_data.begin(), input_data.end(), target_data.begin(), gradInput_data.begin(), abs_updateGradInput_functor(norm));

  THCudaTensor_free(input);
  THCudaTensor_free(target);
  return 1;
}

static const struct luaL_Reg cunn_AbsCriterion__ [] = {
  {"AbsCriterion_updateOutput", cunn_AbsCriterion_updateOutput},
  {"AbsCriterion_updateGradInput", cunn_AbsCriterion_updateGradInput},
  {NULL, NULL}
};

static void cunn_AbsCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_AbsCriterion__, "nn");
  lua_pop(L,1);
}
