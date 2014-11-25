#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
#include "amp_math.h"

struct squareupdateOutput_functor
{
  float operator()(const float& input) const restrict(amp,cpu)
  {
    return input * input;
  }
};

static int cunn_Square_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  bolt::amp::device_vector<float> output_data(THCudaTensor_data(output), THCudaTensor_data(output) + THCudaTensor_nElement(output));
  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input) + THCudaTensor_nElement(input));
  bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), squareupdateOutput_functor());

  THCudaTensor_free(input);
  return 1;
}

struct squareupdateGradInput_functor
{
  float operator()(const float& input, const float& gradOutput) const restrict(amp,cpu)
  {
    return 2.0 * gradOutput * input;
  }
};

static int cunn_Square_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  gradOutput = THCudaTensor_newContiguous(gradOutput);
  THCudaTensor_resizeAs(gradInput, input);

  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input) + THCudaTensor_nElement(input));
  bolt::amp::device_vector<float> gradOutput_data(THCudaTensor_data(gradOutput), THCudaTensor_data(gradOutput) + THCudaTensor_nElement(gradOutput));
  bolt::amp::device_vector<float> gradInput_data(THCudaTensor_data(gradInput), THCudaTensor_data(gradInput) + THCudaTensor_nElement(gradInput));
  bolt::amp::transform(input_data.begin(), input_data.end(), gradOutput_data.begin(),gradInput_data.begin(), squareupdateGradInput_functor());

  THCudaTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_Square__ [] = {
  {"Square_updateOutput", cunn_Square_updateOutput},
  {"Square_updateGradInput", cunn_Square_updateGradInput},
  {NULL, NULL}
};

static void cunn_Square_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Square__, "nn");
  lua_pop(L,1);
}
