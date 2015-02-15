#include "common.h"
#include "amp_math.h"

struct sigmoidupdateOutput_functor
{
  float operator()(const float& input) const restrict(amp,cpu)
  {
    return 1./(1.+ Concurrency::fast_math::exp(-input));
  }
};

static int gpunn_Sigmoid_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  long size = THGPUTensor_nElement(input);
  input = THGPUTensor_newContiguous(input);

  THGPUTensor_resizeAs(output, input);

  DECLARE_BOLT_DEVICE_VECTOR(output, output_data);
  DECLARE_BOLT_DEVICE_VECTOR(input, input_data);

  bolt::amp::transform(input_data.begin() + input->storageOffset,
                       input_data.begin() + input->storageOffset + size,
                       output_data.begin() + output->storageOffset,
                       sigmoidupdateOutput_functor());

  THGPUTensor_free(input);
  return 1;
}

struct sigmoidupdateGradInput_functor
{
  float operator()(const float& output, const float& gradOutput) const restrict(amp,cpu)
  {
    return gradOutput * (1.-output) * output;
  }
};

static int gpunn_Sigmoid_updateGradInput(lua_State *L)
{
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  long size = THGPUTensor_nElement(output);

  gradOutput = THGPUTensor_newContiguous(gradOutput);

  THGPUTensor_resizeAs(gradInput, output);

  DECLARE_BOLT_DEVICE_VECTOR(output, output_data);
  DECLARE_BOLT_DEVICE_VECTOR(gradOutput, gradOutput_data);
  DECLARE_BOLT_DEVICE_VECTOR(gradInput, gradInput_data);
   bolt::amp::transform(output_data.begin() + output->storageOffset,
                        output_data.begin() + output->storageOffset + size,
                        gradOutput_data.begin() + gradOutput->storageOffset,
                        gradInput_data.begin() + gradInput->storageOffset,
                        sigmoidupdateGradInput_functor());

  THGPUTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg gpunn_Sigmoid__ [] = {
  {"Sigmoid_updateOutput", gpunn_Sigmoid_updateOutput},
  {"Sigmoid_updateGradInput", gpunn_Sigmoid_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Sigmoid_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Sigmoid__, "nn");
  lua_pop(L,1);
}
