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
  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(output, input);

  auto dv_output_data = output->get_bolt_dev_vec();
  auto dv_input_data = input->get_bolt_dev_vec();

  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + size,
                       dv_output_data.begin() + output->storageOffset,
                       softPlusupdateOutput_functor(threshold, beta));

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
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THGPUTensor_nElement(output);

  gradOutput = THGPUTensor_newContiguous(gradOutput);
  THGPUTensor_resizeAs(gradInput, output);

  auto dv_output_data = output->get_bolt_dev_vec();
  auto dv_gradOutput_data = gradOutput->get_bolt_dev_vec();
  auto dv_gradInput_data = gradInput->get_bolt_dev_vec();

  bolt::amp::transform(dv_output_data.begin() + output->storageOffset,
                       dv_output_data.begin() + output->storageOffset + size,
                       dv_gradOutput_data.begin() + gradOutput->storageOffset,
                       dv_gradInput_data.begin() + gradInput->storageOffset,
                       softPlusupdateGradInput_functor(threshold,beta));

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

