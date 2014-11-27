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

static int cunn_Threshold_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  bolt::amp::device_vector<float> output_data(THCudaTensor_data(output), THCudaTensor_data(output) + THCudaTensor_nElement(output));
  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input)+ THCudaTensor_nElement(input));
  bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), 
                    thresholdupdateOutput_functor(threshold, val));

  THCudaTensor_free(input);
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

static int cunn_Threshold_updateGradInput(lua_State *L)
{
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THCudaTensor_nElement(output);

  gradOutput = THCudaTensor_newContiguous(gradOutput);
  THCudaTensor_resizeAs(gradInput, output);

  bolt::amp::device_vector<float> input_data(THCudaTensor_data(input),THCudaTensor_data(input) + THCudaTensor_nElement(input));
  bolt::amp::device_vector<float> gradOutput_data(THCudaTensor_data(gradOutput), THCudaTensor_data(gradOutput) + THCudaTensor_nElement(gradOutput));
  bolt::amp::device_vector<float> gradInput_data(THCudaTensor_data(gradInput), THCudaTensor_data(gradInput) + THCudaTensor_nElement(gradInput));
  bolt::amp::transform(input_data.begin(), input_data.end(), gradOutput_data.begin(), gradInput_data.begin(), thresholdupdateGradInput_functor(threshold, val));

  THCudaTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_Threshold__ [] = {
  {"Threshold_updateOutput", cunn_Threshold_updateOutput},
  {"Threshold_updateGradInput", cunn_Threshold_updateGradInput},
  {NULL, NULL}
};

static void cunn_Threshold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Threshold__, "nn");
  lua_pop(L,1);
}

