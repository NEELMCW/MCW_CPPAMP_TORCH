#include "common.h"
#include "amp_math.h"

struct squareupdateOutput_functor
{
  float operator()(const float& input) const restrict(amp,cpu)
  {
    return input * input;
  }
};

static int gpunn_Square_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  input = THGPUTensor_newContiguous(input);

  THGPUTensor_resizeAs(output, input);
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, output, output_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), squareupdateOutput_functor());

  THGPUTensor_free(input);
  return 1;
}

struct squareupdateGradInput_functor
{
  float operator()(const float& input, const float& gradOutput) const restrict(amp,cpu)
  {
    return 2.0 * gradOutput * input;
  }
};

static int gpunn_Square_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  gradOutput = THGPUTensor_newContiguous(gradOutput);
  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(gradInput, input);

  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, gradInput, gradInput_data, gradOutput, gradOutput_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), gradOutput_data.begin(),gradInput_data.begin(), squareupdateGradInput_functor());

  THGPUTensor_free(input);
  THGPUTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg gpunn_Square__ [] = {
  {"Square_updateOutput", gpunn_Square_updateOutput},
  {"Square_updateGradInput", gpunn_Square_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Square_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Square__, "nn");
  lua_pop(L,1);
}
