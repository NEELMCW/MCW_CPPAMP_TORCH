/*#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>*/
#include<numeric>

struct mse_functor
{
  mse_functor() {}

  float operator()(const float& x, const float& y) const
    {
      float z = x-y;
      return z*z;
  }
};


static int cunn_MSECriterion_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  float sum;

  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);
  target = THCudaTensor_newContiguous(target);

  std::vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input)+THCudaTensor_nElement(input));
  std::vector<float> target_data(THCudaTensor_data(target), THCudaTensor_data(target)+THCudaTensor_nElement(target));
  sum = std::inner_product(input_data.begin(), input_data.end(), target_data.begin(), (float) 0, std::plus<float>(), mse_functor());

  if(sizeAverage)
    sum /= size;

  THCudaTensor_free(input);
  THCudaTensor_free(target);
 
  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct mse_updateGradInput_functor
{
  const float norm;

  mse_updateGradInput_functor(float norm_) : norm(norm_) {}

   float operator()(const float& x, const float& y) const
    {
      return norm * (x - y);
  }
};

static int cunn_MSECriterion_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  long size = THCudaTensor_nElement(input);
  float norm = (sizeAverage ? 2./size : 2.);

  input = THCudaTensor_newContiguous(input);
  target = THCudaTensor_newContiguous(target);

  THCudaTensor_resizeAs(gradInput, input);

  std::vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input)+THCudaTensor_nElement(input));
  std::vector<float> target_data(THCudaTensor_data(target), THCudaTensor_data(target)+THCudaTensor_nElement(target));
   std::vector<float> gradInput_data(THCudaTensor_data(gradInput), THCudaTensor_data(gradInput)+THCudaTensor_nElement(gradInput));

  std::transform(input_data.begin(), input_data.end(), target_data.begin(), gradInput_data.begin(), mse_updateGradInput_functor(norm));


  std::copy(gradInput_data.begin(), gradInput_data.end(), gradInput->storage->data);

  THCudaTensor_free(input);
  THCudaTensor_free(target);
  return 1;
}

#define MSECRITERION_THREADS 128

void cunn_MSECriterion_updateOutput_kernel(THCudaStorage* output, THCudaTensor *input, THCudaTensor *target, int nframe, int dim)
{
    Concurrency::array_view<float,1> avInp(Concurrency::extent<1>(input->storage->size), THCudaTensor_data(input));
    Concurrency::array_view<float,1> avTarget(Concurrency::extent<1>(target->storage->size), THCudaTensor_data(target));
    Concurrency::array_view<float,1> avOutput(Concurrency::extent<1>(output->size), output->data);
    Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
    Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp) 
    {
        tile_static float buffer[MSECRITERION_THREADS];
        //int k = blockIdx.x;
        //int k = tidx.tile[0];
        float *input_k = avInp.data();
        //input_k += k * t_ext.tile_dim0;
        float *target_k = avTarget.data();
        //target_k += k * t_ext.tile_dim0;

        //int i_start = threadIdx.x;
        int i_start = tidx.local[0];
        int i_end = dim;
        //int i_step = blockDim.x;
        int i_step = t_ext.tile_dim0;

        // mse
        buffer[i_start] = 0;
        for (int i=i_start; i<i_end; i+=i_step)
        {
            float z = input_k[i] - target_k[i];
            buffer[i_start] += z*z;
        }
        //__syncthreads();
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

void cunn_MSECriterion_updateGradInput_kernel(THCudaTensor *gradInput, THCudaTensor *input, THCudaTensor *target, float norm, int nframe, int dim)
{
    Concurrency::array_view<float,1> avInp(Concurrency::extent<1>(input->storage->size), THCudaTensor_data(input));
    Concurrency::array_view<float,1> avTarget(Concurrency::extent<1>(target->storage->size), THCudaTensor_data(target));
    Concurrency::array_view<float,1> avGradInput(Concurrency::extent<1>(gradInput->storage->size), THCudaTensor_data(gradInput));
    Concurrency::extent<1> grdExt(MSECRITERION_THREADS);
    Concurrency::tiled_extent<MSECRITERION_THREADS> t_ext(grdExt);
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MSECRITERION_THREADS> tidx) restrict(amp)
    {
        //int k = blockIdx.x;
        //int k = tidx.tile[0];
        float *input_k = avInp.data();
        //input_k += k * t_ext.tile_dim0;
        float *target_k = avTarget.data();
        //target_k += k * t_ext.tile_dim0;
        float *gradInput_k = avGradInput.data();
        //gradInput_k += k * t_ext.tile_dim0;

        //int i_start = threadIdx.x;
        int i_start = tidx.local[0];
        int i_end = dim;
        //int i_step = blockDim.x;
        int i_step = t_ext.tile_dim0;

        // gradInput
        for (int i=i_start; i<i_end; i+=i_step)
            gradInput_k[i] = norm*(input_k[i] - target_k[i]);
    });
    avInp.synchronize();
}

static int cunn_MSECriterion_updateOutput2(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  long size = THCudaTensor_nElement(input);
  THCudaTensor *temp1 = input;
  THCudaTensor *temp2 = target;

  input = THCudaTensor_newContiguous(input);
  target = THCudaTensor_newContiguous(target);

  THCudaStorage *output = THCudaStorage_newWithSize(1);

   size = (size + (MSECRITERION_THREADS - 1)) & ~(MSECRITERION_THREADS - 1);

   cunn_MSECriterion_updateOutput_kernel(output, input, target, 1, size);

   lua_pushnumber(L, THCudaStorage_get(output, 0));

   if (input != temp1)
   {
        THCudaTensor_free(input);
        input = NULL;
   }
   if (target != temp2)
   {
       THCudaTensor_free(target);
       target = NULL;
   }
   THCudaStorage_free(output);

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  return 1;
}

static int cunn_MSECriterion_updateGradInput2(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);
  float norm = (sizeAverage ? 2./size : 2.);
  
    THCudaTensor *temp1 = input;
    THCudaTensor *temp2 = target;
    THCudaTensor *temp3 = gradInput;
    input = THCudaTensor_newContiguous(input);
    target = THCudaTensor_newContiguous(target);

    THCudaTensor_resizeAs(gradInput, input);
    
    cunn_MSECriterion_updateGradInput_kernel(gradInput, input, target, norm, 1, size);

    if (input != temp1)
    {
        THCudaTensor_free(input);
        input = NULL;
    }
    if (target != temp2)
    {
        THCudaTensor_free(target);
        target = NULL;
    }
    if (gradInput != temp3)
    {
        THCudaTensor_free(gradInput);
        gradInput = NULL;
    }
    return 1;


  return 1;
}


static const struct luaL_Reg cunn_MSECriterion__ [] = {
  {"MSECriterion_updateOutput", cunn_MSECriterion_updateOutput},
  {"MSECriterion_updateGradInput", cunn_MSECriterion_updateGradInput},
  {"MSECriterion_updateOutput2", cunn_MSECriterion_updateOutput2},
  {"MSECriterion_updateGradInput2", cunn_MSECriterion_updateGradInput2},
  {NULL, NULL}
};

static void cunn_MSECriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_MSECriterion__, "nn");
  lua_pop(L,1);
}
