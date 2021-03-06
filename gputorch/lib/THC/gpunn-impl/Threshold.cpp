#include "amp_math.h"

struct thresholdupdateOutput_functor
{
  const double threshold;
  const double val;

  thresholdupdateOutput_functor(double threshold_, double val_) restrict(amp,cpu): threshold(threshold_), val(val_) {}

 float operator()(const float& input) const restrict(amp,cpu)
  {
    return (input > threshold) ? input : val;
  }
};

static int gpunn_Threshold_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(output, input);

  auto dv_output_data = output->get_bolt_dev_vec();
  auto dv_input_data = input->get_bolt_dev_vec();

  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + size,
                       dv_output_data.begin() + output->storageOffset,
                       thresholdupdateOutput_functor(threshold, val));

  THGPUTensor_free(input);
  return 1;
}

struct thresholdupdateGradInput_functor
{
  const double threshold;
  const double val;

  thresholdupdateGradInput_functor(double threshold_, double val_) restrict(amp,cpu) : threshold(threshold_), val(val_) {}

  float operator()(const float& input, const float& gradOutput) const restrict(amp,cpu)
  {
    return (input > threshold) ? gradOutput : 0;
  }
};

static int gpunn_Threshold_updateGradInput(lua_State *L)
{
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THGPUTensor_nElement(output);

  gradOutput = THGPUTensor_newContiguous(gradOutput);
  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(gradInput, output);

  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_gradOutput_data = gradOutput->get_bolt_dev_vec();
  auto dv_gradInput_data = gradInput->get_bolt_dev_vec();

  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + size,
                       dv_gradOutput_data.begin() + gradOutput->storageOffset,
                       dv_gradInput_data.begin() + gradInput->storageOffset,
                       thresholdupdateGradInput_functor(threshold, val));

  THGPUTensor_free(input);
  THGPUTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg gpunn_Threshold__ [] = {
  {"Threshold_updateOutput", gpunn_Threshold_updateOutput},
  {"Threshold_updateGradInput", gpunn_Threshold_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Threshold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Threshold__, "nn");
  lua_pop(L,1);
}

