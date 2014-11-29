#include<numeric>
#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
#include "amp_math.h"

struct mse_functor
{
  mse_functor() restrict(amp,cpu) {}

  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    float z = x-y;
    return z*z;
  }
};


static int gpunn_MSECriterion_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  float sum;

  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  bolt::amp::device_vector<float> input_data(THGPUTensor_data(input), THGPUTensor_data(input)+THGPUTensor_nElement(input));
  bolt::amp::device_vector<float> target_data(THGPUTensor_data(target), THGPUTensor_data(target)+THGPUTensor_nElement(target));
  sum = bolt::amp::inner_product(input_data.begin(), input_data.end(), target_data.begin(), (float) 0, bolt::amp::plus<float>(), mse_functor());

  if(sizeAverage)
    sum /= size;

  THGPUTensor_free(input);
  THGPUTensor_free(target);
 
  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct mse_updateGradInput_functor
{
  const float norm;

  mse_updateGradInput_functor(float norm_) restrict(amp,cpu) : norm(norm_) {}

  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return norm * (x - y);
  }
};

static int gpunn_MSECriterion_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  long size = THGPUTensor_nElement(input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  THGPUTensor_resizeAs(gradInput, input);

  bolt::amp::device_vector<float> input_data(THGPUTensor_data(input), THGPUTensor_data(input)+THGPUTensor_nElement(input));
  bolt::amp::device_vector<float> target_data(THGPUTensor_data(target), THGPUTensor_data(target)+THGPUTensor_nElement(target));
  bolt::amp::device_vector<float> gradInput_data(THGPUTensor_data(gradInput), THGPUTensor_data(gradInput)+THGPUTensor_nElement(gradInput));

  bolt::amp::transform(input_data.begin(), input_data.end(), target_data.begin(), gradInput_data.begin(), mse_updateGradInput_functor(norm));

  THGPUTensor_free(input);
  THGPUTensor_free(target);
  return 1;
}

#define MSECRITERION_THREADS 128

void gpunn_MSECriterion_updateOutput_kernel(THGPUStorage* output, THGPUTensor *input, THGPUTensor *target, int nframe, int dim)
{
  Concurrency::array_view<float,1> avInp(Concurrency::extent<1>(input->storage->size), THGPUTensor_data(input));
  Concurrency::array_view<float,1> avTarget(Concurrency::extent<1>(target->storage->size), THGPUTensor_data(target));
  Concurrency::array_view<float,1> avOutput(Concurrency::extent<1>(output->size), output->data);
  Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
  Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp) 
  {
    tile_static float buffer[MSECRITERION_THREADS];
    float *input_k = avInp.data();
    float *target_k = avTarget.data();

    int i_start = tidx.local[0];
    int i_end = dim;
    int i_step = t_ext.tile_dim0;

    // mse
    buffer[i_start] = 0;
    for (int i=i_start; i<i_end; i+=i_step)
    {
      float z = input_k[i] - target_k[i];
      buffer[i_start] += z*z;
    }
    tidx.barrier.wait();

    //reduce
    if (i_start == 0)
    {
      avOutput[0] = 0;
      for (int i=0; i<i_step; i++)
      {
        avOutput[0] += buffer[i];
      }
    }
  });
  avOutput.synchronize();
}

void gpunn_MSECriterion_updateGradInput_kernel(THGPUTensor *gradInput, THGPUTensor *input, THGPUTensor *target, float norm, int nframe, int dim)
{
  Concurrency::array_view<float,1> avInp(Concurrency::extent<1>(input->storage->size), THGPUTensor_data(input));
  Concurrency::array_view<float,1> avTarget(Concurrency::extent<1>(target->storage->size), THGPUTensor_data(target));
  Concurrency::array_view<float,1> avGradInput(Concurrency::extent<1>(gradInput->storage->size), THGPUTensor_data(gradInput));
  Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
  Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp)
  {
    float *input_k = avInp.data();
    float *target_k = avTarget.data();
    float *gradInput_k = avGradInput.data();

    int i_start = tidx.local[0];
    int i_end = dim;
    int i_step = t_ext.tile_dim0;

    // gradInput
    for (int i=i_start; i<i_end; i+=i_step)
      gradInput_k[i] = norm*(input_k[i] - target_k[i]);
  });
  avInp.synchronize();
}

static int gpunn_MSECriterion_updateOutput2(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  long size = THGPUTensor_nElement(input);
  THGPUTensor *temp1 = input;
  THGPUTensor *temp2 = target;

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  THGPUStorage *output = THGPUStorage_newWithSize(1);

  size = (size + (MSECRITERION_THREADS - 1)) & ~(MSECRITERION_THREADS - 1);
  gpunn_MSECriterion_updateOutput_kernel(output, input, target, 1, size);

  lua_pushnumber(L, THGPUStorage_get(output, 0));

  if (input != temp1)
  {
    THGPUTensor_free(input);
    input = NULL;
  }
  if (target != temp2)
  {
    THGPUTensor_free(target);
    target = NULL;
  }
  THGPUStorage_free(output);

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  return 1;
}

static int gpunn_MSECriterion_updateGradInput2(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  long size = THGPUTensor_nElement(input);
  float norm = (sizeAverage ? 2./size : 2.);

  THGPUTensor *temp1 = input;
  THGPUTensor *temp2 = target;
  THGPUTensor *temp3 = gradInput;
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  THGPUTensor_resizeAs(gradInput, input);

  gpunn_MSECriterion_updateGradInput_kernel(gradInput, input, target, norm, 1, size);

  if (input != temp1)
  {
    THGPUTensor_free(input);
    input = NULL;
  }
  if (target != temp2)
  {
    THGPUTensor_free(target);
    target = NULL;
  }
  if (gradInput != temp3)
  {
    THGPUTensor_free(gradInput);
    gradInput = NULL;
  }
  return 1;
}

static const struct luaL_Reg gpunn_MSECriterion__ [] = {
  {"MSECriterion_updateOutput", gpunn_MSECriterion_updateOutput},
  {"MSECriterion_updateGradInput", gpunn_MSECriterion_updateGradInput},
  {"MSECriterion_updateOutput2", gpunn_MSECriterion_updateOutput2},
  {"MSECriterion_updateGradInput2", gpunn_MSECriterion_updateGradInput2},
  {NULL, NULL}
};

static void gpunn_MSECriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_MSECriterion__, "nn");
  lua_pop(L,1);
}
