#include<numeric>
#include "common.h"
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
  luaL_argcheck(L, THGPUTensor_nElement(input) == THGPUTensor_nElement(target), 2,
                "input and target need to have the same number of elements");

  float sum;

  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  DECLARE_BOLT_DEVICE_VECTOR(target, target_data);
  DECLARE_BOLT_DEVICE_VECTOR(input, input_data);

  sum = bolt::amp::inner_product(input_data.begin() + input->storageOffset,
                                 input_data.begin() + input->storageOffset + size,
                                 target_data.begin() + target->storageOffset,
                                 (float) 0, bolt::amp::plus<float>(), mse_functor());

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
  luaL_argcheck(L, THGPUTensor_nElement(input) == THGPUTensor_nElement(target), 2,
                "input and target need to have the same number of elements");

  long size = THGPUTensor_nElement(input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  THGPUTensor_resizeAs(gradInput, input);

  DECLARE_BOLT_DEVICE_VECTOR(input, input_data);
  DECLARE_BOLT_DEVICE_VECTOR(target, target_data);
  DECLARE_BOLT_DEVICE_VECTOR(gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin() + input->storageOffset,
                       input_data.begin() + input->storageOffset + size,
                       target_data.begin() + target->storageOffset,
                       gradInput_data.begin() + gradInput->storageOffset,
                       mse_updateGradInput_functor(norm));

  THGPUTensor_free(input);
  THGPUTensor_free(target);
  return 1;
}

#define MSECRITERION_THREADS 128

void gpunn_MSECriterion_updateOutput_kernel(Concurrency::array_view<float,1> &avOutput,
  Concurrency::array_view<float,1> &avInp, Concurrency::array_view<float,1> &avTarget, int nframe, int dim, int sizeAverage)
{
  Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
  Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp) 
  {
    tile_static float buffer[MSECRITERION_THREADS];
    float *input_k = avInp.data();
    float *target_k = avTarget.data();
    int k = tidx.tile[0];
    input_k += k*dim;
    target_k += k*dim;

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
      if (sizeAverage)
       avOutput[0] /= dim;
    }
  });
}

void gpunn_MSECriterion_updateGradInput_kernel(Concurrency::array_view<float,1> &avGradInput,
  Concurrency::array_view<float,1> &avInp, Concurrency::array_view<float,1> &avTarget, float norm, int nframe, int dim)
{
  Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
  Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp)
  {
    float *input_k = avInp.data();
    float *target_k = avTarget.data();
    float *gradInput_k = avGradInput.data();

    int k = tidx.tile[0];
    gradInput_k += k*dim;
    input_k += k*dim;
    target_k += k*dim;

    int i_start = tidx.local[0];
    int i_end = dim;
    int i_step = t_ext.tile_dim0;

    // gradInput
    for (int i=i_start; i<i_end; i+=i_step)
      gradInput_k[i] = norm*(input_k[i] - target_k[i]);
  });
}

static int cunn_MSECriterion_updateOutput2(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  THGPUStorage *output = THGPUStorage_newWithSize(1);

  PREPARE_AV_WITH_STORAGE(output, pavOutput);
  PREPARE_AV(input, pavInput);
  PREPARE_AV(target, pavTarget);
  gpunn_MSECriterion_updateOutput_kernel(*pavOutput, *pavInput, *pavTarget, 1, size, sizeAverage);

  lua_pushnumber(L, THGPUStorage_get(output, 0));


  THGPUTensor_free(input);
  THGPUTensor_free(target);
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

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  THGPUTensor_resizeAs(gradInput, input);

  PREPARE_AV(gradInput, pavGradInput);
  PREPARE_AV(input, pavInput);
  PREPARE_AV(target, pavTarget);
  gpunn_MSECriterion_updateGradInput_kernel(*pavGradInput, *pavInput, *pavTarget, norm, 1, size);

  THGPUTensor_free(input);
  THGPUTensor_free(target);
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
