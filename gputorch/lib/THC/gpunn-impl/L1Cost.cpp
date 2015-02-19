#include "common.h"
#include "amp_math.h"

struct l1cost_functor
{
  l1cost_functor() restrict(amp,cpu) {}

  float operator()(float x, float y) const restrict(amp,cpu)
  {
      return Concurrency::fast_math::fabs(x)+Concurrency::fast_math::fabs(y);
  }
};

static int gpunn_L1Cost_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");

  float sum;
  long size = THGPUTensor_nElement(input);
  input = THGPUTensor_newContiguous(input);
  DECLARE_BOLT_DEVICE_VECTOR(input, input_data);
  sum = bolt::amp::reduce(input_data.begin() + input->storageOffset,
                          input_data.begin() + input->storageOffset + size,
                          (float) 0, l1cost_functor());

  THGPUTensor_free(input);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

struct l1cost_updateGradInput_functor
{
  l1cost_updateGradInput_functor() {}

  float operator()(float x) const restrict(amp,cpu)
    {
      if(x > 0)
        return 1;
      else if(x < 0)
        return -1;
      else
        return 0;
  }
};

static int gpunn_L1Cost_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(gradInput, input);

  DECLARE_BOLT_DEVICE_VECTOR(gradInput, gradInput_data);
  DECLARE_BOLT_DEVICE_VECTOR(input, input_data);

  bolt::amp::transform(input_data.begin() + input->storageOffset,
                       input_data.begin() + input->storageOffset + size,
                       gradInput_data.begin() + gradInput->storageOffset,
                       l1cost_updateGradInput_functor());

  THGPUTensor_free(input);
  return 1;
}

static const struct luaL_Reg gpunn_L1Cost__ [] = {
  {"L1Cost_updateOutput", gpunn_L1Cost_updateOutput},
  {"L1Cost_updateGradInput", gpunn_L1Cost_updateGradInput},
  {NULL, NULL}
};

static void gpunn_L1Cost_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_L1Cost__, "nn");
  lua_pop(L,1);
}
