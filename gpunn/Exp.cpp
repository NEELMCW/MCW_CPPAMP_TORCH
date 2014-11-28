#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
#include "amp_math.h"

struct expupdateOutput_functor
{
  float operator()(const float& input) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::exp(input);
  }
};

static int gpunn_Exp_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  long size = THGPUTensor_nElement(input);
  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(output, input);

  bolt::amp::device_vector<float> output_data(THGPUTensor_data(output), THGPUTensor_data(output) + THGPUTensor_nElement(output));
  bolt::amp::device_vector<float> input_data(THGPUTensor_data(input), THGPUTensor_data(input) + THGPUTensor_nElement(input));
  bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), expupdateOutput_functor());

  THGPUTensor_free(input);
  return 1;
}

struct expupdateGradInput_functor
{
  float operator()(const float& output, const float& gradOutput) const restrict(amp,cpu)
  {
    return gradOutput * output;
  }
};

static int gpunn_Exp_updateGradInput(lua_State *L)
{
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  long size = THGPUTensor_nElement(output);

  gradOutput = THGPUTensor_newContiguous(gradOutput);

  THGPUTensor_resizeAs(gradInput, output);

  bolt::amp::device_vector<float> output_data(THGPUTensor_data(output), THGPUTensor_data(output) + THGPUTensor_nElement(output));
  bolt::amp::device_vector<float> gradOutput_data(THGPUTensor_data(gradOutput), THGPUTensor_data(gradOutput) + THGPUTensor_nElement(gradOutput));
  bolt::amp::device_vector<float> gradInput_data(THGPUTensor_data(gradInput), THGPUTensor_data(gradInput) + THGPUTensor_nElement(gradInput));
  bolt::amp::transform(output_data.begin(), output_data.end(), gradOutput_data.begin(),gradInput_data.begin(), expupdateGradInput_functor());

  THGPUTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg gpunn_Exp__ [] = {
  {"Exp_updateOutput", gpunn_Exp_updateOutput},
  {"Exp_updateGradInput", gpunn_Exp_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Exp_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Exp__, "nn");
  lua_pop(L,1);
}
