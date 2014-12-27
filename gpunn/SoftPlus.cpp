#include "common.h"
#include "amp_math.h"
struct softPlusupdateOutput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateOutput_functor(float threshold_, float beta_) restrict(amp,cpu) : threshold(threshold_), beta(beta_) {}

  float operator()(const float& input) const restrict(amp,cpu)
  {
    float betain = beta * input;
    return ((betain) > threshold) ? input : (1/beta) * Concurrency::precise_math::log1p(Concurrency::fast_math::exp(betain));
  }
};

static int gpunn_SoftPlus_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");

  input = THGPUTensor_newContiguous(input);

  THGPUTensor_resizeAs(output, input);

   DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, output, output_data);
   bolt::amp::transform(input_data.begin(), input_data.end(), output_data.begin(), softPlusupdateOutput_functor(threshold, beta));

  THGPUTensor_free(input);
  return 1;
}

struct softPlusupdateGradInput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateGradInput_functor(float threshold_, float beta_)restrict(amp,cpu): threshold(threshold_), beta(beta_) {}

  float operator()(const float& output, const float& gradOutput) const restrict(amp,cpu)
  {
    float betaout = beta * output;
    float exp_bo = Concurrency::fast_math::exp(betaout);
    return ((betaout) > threshold) ? gradOutput : gradOutput * (exp_bo - 1) / exp_bo;
  }
};

static int gpunn_SoftPlus_updateGradInput(lua_State *L)
{
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");

  gradOutput = THGPUTensor_newContiguous(gradOutput);
  THGPUTensor_resizeAs(gradInput, output);

   DECLARE_BOLT_DEVICE_VECTOR_3(output, output_data, gradInput, gradInput_data, gradOutput, gradOutput_data);
   bolt::amp::transform(output_data.begin(), output_data.end(), gradOutput_data.begin(),gradInput_data.begin(), softPlusupdateGradInput_functor(threshold,beta));

  THGPUTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg gpunn_SoftPlus__ [] = {
  {"SoftPlus_updateOutput", gpunn_SoftPlus_updateOutput},
  {"SoftPlus_updateGradInput", gpunn_SoftPlus_updateGradInput},
  {NULL, NULL}
};

void gpunn_SoftPlus_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SoftPlus__, "nn");
  lua_pop(L,1);
}

