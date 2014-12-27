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

  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, output, output_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), 
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
  THGPUTensor_resizeAs(gradInput, output);

  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, gradInput, gradInput_data, gradOutput, gradOutput_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), gradOutput_data.begin(), gradInput_data.begin(), thresholdupdateGradInput_functor(threshold, val));

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

