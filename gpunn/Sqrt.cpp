#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
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

  bolt::amp::device_vector<float> output_data(THGPUTensor_data(output), THGPUTensor_data(output) + THGPUTensor_nElement(output));
  bolt::amp::device_vector<float> input_data(THGPUTensor_data(input), THGPUTensor_data(input) + THGPUTensor_nElement(input));
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

  bolt::amp::device_vector<float> output_data(THGPUTensor_data(output), THGPUTensor_data(output) + THGPUTensor_nElement(output));
  bolt::amp::device_vector<float> gradOutput_data(THGPUTensor_data(gradOutput), THGPUTensor_data(gradOutput) + THGPUTensor_nElement(gradOutput));
  bolt::amp::device_vector<float> gradInput_data(THGPUTensor_data(gradInput), THGPUTensor_data(gradInput) + THGPUTensor_nElement(gradInput));
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
