#include "common.h"

struct hardtanhupdateOutput_functor
{
  float operator()(const float& input) const restrict(amp,cpu)
  {
    if (input < -1)
      return -1;
    else if (input <= 1)
      return input;
    else
      return 1;
  }
};

static int gpunn_HardTanh_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(output, input);

  auto dv_output_data = output->get_bolt_dev_vec();
  auto dv_input_data = input->get_bolt_dev_vec();

  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + size,
                       dv_output_data.begin() + output->storageOffset,
                       hardtanhupdateOutput_functor());

  THGPUTensor_free(input);
  return 1;
}

struct hardtanhupdateGradInput_functor
{
  float operator()(const float& input, const float& gradOutput) const restrict(amp,cpu)
  {
    if (input < -1 || input > 1)
      return 0;
    else
      return gradOutput;
  }
};

static int gpunn_HardTanh_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  gradOutput = THGPUTensor_newContiguous(gradOutput);

  THGPUTensor_resizeAs(gradInput, input);

  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_gradOutput_data = gradOutput->get_bolt_dev_vec();
  auto dv_gradInput_data = gradInput->get_bolt_dev_vec();

  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + size,
                       dv_gradOutput_data.begin() + gradOutput->storageOffset,
                       dv_gradInput_data.begin() + gradInput->storageOffset,
                       hardtanhupdateGradInput_functor());

  THGPUTensor_free(gradOutput);
  THGPUTensor_free(input);
  return 1;
}

static const struct luaL_Reg gpunn_HardTanh__ [] = {
  {"HardTanh_updateOutput", gpunn_HardTanh_updateOutput},
  {"HardTanh_updateGradInput", gpunn_HardTanh_updateGradInput},
  {NULL, NULL}
};

static void gpunn_HardTanh_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_HardTanh__, "nn");
  lua_pop(L,1);
}
