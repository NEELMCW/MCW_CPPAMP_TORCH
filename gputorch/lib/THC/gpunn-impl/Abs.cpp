#include <iostream>
#include <vector>
#include "common.h"
#include "amp_math.h"

struct absupdateOutput_functor
{
  float operator()(const float& input) const  restrict(amp,cpu)
  {
    return Concurrency::fast_math::fabs(input);
  }
};

static int gpunn_Abs_updateOutput(lua_State *L)
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
                       absupdateOutput_functor());

  THGPUTensor_free(input);
  return 1;
}

struct absupdateGradInput_functor
{
  float operator()(const float& input, const float& gradOutput) const restrict(amp,cpu)
  {
    if(input < 0)
      return -gradOutput;
    else 
      return gradOutput; 
  }
};

static int gpunn_Abs_updateGradInput(lua_State *L)
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
                       absupdateGradInput_functor());

  THGPUTensor_free(gradOutput);
  THGPUTensor_free(input);
  return 1;
}

static const struct luaL_Reg gpunn_Abs__ [] = {
  {"Abs_updateOutput", gpunn_Abs_updateOutput},
  {"Abs_updateGradInput", gpunn_Abs_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Abs_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Abs__, "nn");
  lua_pop(L,1);
}
