#include "common.h"
#include "amp_math.h"
struct sqrtupdateOutput_functor
{
  const double bias;

  sqrtupdateOutput_functor(double bias_) restrict(amp,cpu): bias(bias_) {}

  float operator()(const float& input) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::sqrt(input + bias);
  }
};

static int gpunn_Sqrt_updateOutput(lua_State *L)
{
  double bias = luaT_getfieldchecknumber(L,1,"eps");
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  long size = THGPUTensor_nElement(input);
  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(output, input);

  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, output, output_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), sqrtupdateOutput_functor(bias));

  THGPUTensor_free(input);
  return 1;
}

struct sqrtupdateGradInput_functor
{
  const double bias;

  sqrtupdateGradInput_functor(double bias_) restrict(amp,cpu) : bias(bias_) {}

  float operator()(const float& output, const float& gradOutput) const restrict(amp,cpu)
  {
    return 0.5 * gradOutput / output;
  }
};

static int gpunn_Sqrt_updateGradInput(lua_State *L)
{
  double bias = luaT_getfieldchecknumber(L,1,"eps");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  long size = THGPUTensor_nElement(output);

  gradOutput = THGPUTensor_newContiguous(gradOutput);
  THGPUTensor_resizeAs(gradInput, output);

  DECLARE_BOLT_DEVICE_VECTOR_3(output, output_data, gradInput, gradInput_data, gradOutput, gradOutput_data);
  bolt::amp::transform(output_data.begin(), output_data.end(), gradOutput_data.begin(),gradInput_data.begin(), sqrtupdateGradInput_functor(bias));

  THGPUTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg gpunn_Sqrt__ [] = {
  {"Sqrt_updateOutput", gpunn_Sqrt_updateOutput},
  {"Sqrt_updateGradInput", gpunn_Sqrt_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Sqrt_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Sqrt__, "nn");
  lua_pop(L,1);
}
