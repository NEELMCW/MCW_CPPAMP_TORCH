#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
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

  bolt::amp::device_vector<float> output_data(THGPUTensor_data(output), THGPUTensor_data(output) + THGPUTensor_nElement(output));
  bolt::amp::device_vector<float> input_data(THGPUTensor_data(input), THGPUTensor_data(input)+ THGPUTensor_nElement(input));
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

  bolt::amp::device_vector<float> input_data(THGPUTensor_data(input),THGPUTensor_data(input) + THGPUTensor_nElement(input));
  bolt::amp::device_vector<float> gradOutput_data(THGPUTensor_data(gradOutput), THGPUTensor_data(gradOutput) + THGPUTensor_nElement(gradOutput));
  bolt::amp::device_vector<float> gradInput_data(THGPUTensor_data(gradInput), THGPUTensor_data(gradInput) + THGPUTensor_nElement(gradInput));
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

