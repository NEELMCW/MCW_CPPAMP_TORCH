#include <numeric>
#include "common.h"
#include "amp_math.h"

struct kl_functor
{
  kl_functor() {}

  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return y > 0 ? y * (Concurrency::fast_math::log(y) - x) : 0;
  }
};


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

  DECLARE_BOLT_DEVICE_VECTOR(target, target_data);
  DECLARE_BOLT_DEVICE_VECTOR(input, input_data);
  sum = bolt::amp::inner_product(input_data.begin() + input->storageOffset,
                                 input_data.begin() + input->storageOffset + size,
                                 target_data.begin() + target->storageOffset,
                                 (float) 0, bolt::amp::plus<float>(), kl_functor());
  if (sizeAverage)
    sum /= size;

  THGPUTensor_free(input);
  THGPUTensor_free(target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct kl_updateGradInput_functor
{
  const float norm;

  kl_updateGradInput_functor(float norm_) restrict(amp,cpu) : norm(norm_) {}

  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return y > 0 ? norm * (-y) : 0;
  }
};

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

  DECLARE_BOLT_DEVICE_VECTOR(input, input_data);
  DECLARE_BOLT_DEVICE_VECTOR(target, target_data);
  DECLARE_BOLT_DEVICE_VECTOR(gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin() + input->storageOffset,
                       input_data.begin() + input->storageOffset + size,
                       target_data.begin() + target->storageOffset,
                       gradInput_data.begin() + gradInput->storageOffset,
                       kl_updateGradInput_functor(norm));

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
