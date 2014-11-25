#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
#include "amp_math.h"


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

static int cunn_HardTanh_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  bolt::amp::device_vector<float> output_data(THCudaTensor_data(output), THCudaTensor_data(output) + THCudaTensor_nElement(output));
  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input) + THCudaTensor_nElement(input));
  bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), hardtanhupdateOutput_functor());

  THCudaTensor_free(input);
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

static int cunn_HardTanh_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);
  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, input);

  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input) + THCudaTensor_nElement(input));
  bolt::amp::device_vector<float> gradOutput_data(THCudaTensor_data(gradOutput), THCudaTensor_data(gradOutput) + THCudaTensor_nElement(gradOutput));
  bolt::amp::device_vector<float> gradInput_data(THCudaTensor_data(gradInput), THCudaTensor_data(gradInput) + THCudaTensor_nElement(gradInput));
  bolt::amp::transform(input_data.begin(), input_data.end(), gradOutput_data.begin(),gradInput_data.begin(), hardtanhupdateGradInput_functor());


  THCudaTensor_free(gradOutput);
  THCudaTensor_free(input);
  return 1;
}

static const struct luaL_Reg cunn_HardTanh__ [] = {
  {"HardTanh_updateOutput", cunn_HardTanh_updateOutput},
  {"HardTanh_updateGradInput", cunn_HardTanh_updateGradInput},
  {NULL, NULL}
};

static void cunn_HardTanh_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_HardTanh__, "nn");
  lua_pop(L,1);
}
