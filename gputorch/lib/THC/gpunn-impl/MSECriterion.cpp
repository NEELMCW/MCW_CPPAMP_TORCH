#include<numeric>
#include "amp_math.h"
#include "THCBolt.h"

static int gpunn_MSECriterion_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  luaL_argcheck(L, THGPUTensor_nElement(input) == THGPUTensor_nElement(target),
                2, "input and target need to have the same number of elements");

  long size = THGPUTensor_nElement(input);
  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  float sum = boltInnerProduct_plus_mse( input, target);

  if(sizeAverage)
    sum /= size;

  THGPUTensor_free(input);
  THGPUTensor_free(target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");
  lua_pushnumber(L, sum);

  return 1;
}

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

  boltTransform_mse(input, target, gradInput, norm);

  THGPUTensor_free(input);
  THGPUTensor_free(target);
  return 1;
}

#define MSECRITERION_THREADS 128

void gpunn_MSECriterion_updateOutput_kernel(Concurrency::array_view<float,1> &avOutput, long outOffset,
                                            Concurrency::array_view<float,1> &avInp, long inpOffset,
                                            Concurrency::array_view<float,1> &avTarget, long targetOffset,
                                            int nframe, int dim, int sizeAverage)
{
  Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
  Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp)
  {
    tile_static float buffer[MSECRITERION_THREADS];
    float *input_k = avInp.data() + inpOffset;
    float *target_k = avTarget.data() + targetOffset;
    int k = tidx.tile[0];
    input_k += k * dim;
    target_k += k * dim;

    int i_start = tidx.local[0];
    int i_end = dim;
    int i_step = t_ext.tile_dim0;

    // mse
    buffer[i_start] = 0;
    for (int i = i_start; i < i_end; i += i_step)
    {
      float z = input_k[i] - target_k[i];
      buffer[i_start] += z * z;
    }
    tidx.barrier.wait();

    //reduce
    if (i_start == 0)
    {
      avOutput[outOffset] = 0;
      for (int i = 0; i < i_step; i++)
      {
        avOutput[outOffset] += buffer[i];
      }
      if (sizeAverage)
       avOutput[outOffset] /= dim;
    }
  });
}

void gpunn_MSECriterion_updateGradInput_kernel(Concurrency::array_view<float,1> &avGradInput, long gradInOffset,
                                               Concurrency::array_view<float,1> &avInp, long inpOffset,
                                               Concurrency::array_view<float,1> &avTarget, long targetOffset,
                                               float norm, int nframe, int dim)
{
  Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
  Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp)
  {
    float *input_k = avInp.data() + inpOffset;
    float *target_k = avTarget.data() + targetOffset;
    float *gradInput_k = avGradInput.data() + gradInOffset;

    int k = tidx.tile[0];
    gradInput_k += k * dim;
    input_k += k * dim;
    target_k += k * dim;

    int i_start = tidx.local[0];
    int i_end = dim;
    int i_step = t_ext.tile_dim0;

    // gradInput
    for (int i = i_start; i < i_end; i += i_step)
      gradInput_k[i] = norm*(input_k[i] - target_k[i]);
  });
}

static int gpunn_MSECriterion_updateOutput2(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *target = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  long size = THGPUTensor_nElement(input);

  input = THGPUTensor_newContiguous(input);
  target = THGPUTensor_newContiguous(target);

  THGPUStorage *output = THGPUStorage_newWithSize(1);

  Concurrency::array_view<float, 1> *pavOutput = static_cast<Concurrency::array_view<float, 1> *>(output->allocatorContext);
  auto avInput = input->get_array_view();
  auto avTarget = target->get_array_view();

  //Since there is no storageOffset for THGPUStorage the 2nd Argument is set to 0 
  gpunn_MSECriterion_updateOutput_kernel(*pavOutput, 0,
                                         avInput, input->storageOffset,
                                         avTarget, target->storageOffset,
                                         1, size, sizeAverage);

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

  auto avGradInput = gradInput->get_array_view();
  auto avInput = input->get_array_view();
  auto avTarget = target->get_array_view();

  gpunn_MSECriterion_updateGradInput_kernel(avGradInput, gradInput->storageOffset,
                                            avInput, input->storageOffset,
                                            avTarget, target->storageOffset,
                                            norm, 1, size);

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
