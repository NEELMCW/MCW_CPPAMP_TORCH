#define MULTIMARGIN_THREADS 128

void cunn_MultiMarginCriterion_updateOutput_kernel(THCudaStorage *output, THCudaStorage *input, THCudaStorage *target, int nframe, int dim, int sizeaverage)
{
    Concurrency::array_view<float,1> avInp(Concurrency::extent<1>(input->size), input->data);
    Concurrency::array_view<float,1> avTarget(Concurrency::extent<1>(target->size), target->data);
    Concurrency::array_view<float,1> avOutput(Concurrency::extent<1>(output->size), output->data);
    Concurrency::extent<1> grdExt(MULTIMARGIN_THREADS);
    Concurrency::tiled_extent<MULTIMARGIN_THREADS> t_ext(grdExt);
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MULTIMARGIN_THREADS> tidx) restrict(amp) 
    {
        tile_static float buffer[MULTIMARGIN_THREADS];
        //int k = blockIdx.x;
        int k = tidx.tile[0];
        float *input_k = avInp.data();
        //intput_k += k*dim;
        float *output_k = avOutput.data();
        //output_k += k;
        int target_k = ((int)avTarget[k])-1;
        float input_target_k = input_k[target_k];

        //int i_start = threadIdx.x;
        int i_start = tidx.local[0];
        int i_end = dim;
        //int i_step = blockDim.x;
        int i_step = t_ext.tile_dim0;

        buffer[i_start] = 0;
        for(int i = i_start; i < i_end; i += i_step)
        {
        float z = 1 - input_target_k + input_k[i];
        if(i == target_k)
            continue;
    
        if(z > 0)
            buffer[i_start] += z;
        }
        //__syncthreads();
        tidx.barrier.wait();

        // reduce
        if (i_start == 0)
        {
        float sum = 0;
        for (int i=0; i<i_step; i++)
            sum += buffer[i];

        if(sizeaverage)
            *output_k = sum/dim;
        else
            *output_k = sum;
        }
    });
}

void cunn_MultiMarginCriterion_updateGradInput_kernel(THCudaTensor *gradInput, THCudaTensor *input, THCudaTensor *target, int nframe, int dim, int sizeaverage)
{
    Concurrency::array_view<float,1> avInp(Concurrency::extent<1>(input->storage->size), THCudaTensor_data(input));
    Concurrency::array_view<float,1> avTarget(Concurrency::extent<1>(target->storage->size), THCudaTensor_data(target));
    Concurrency::array_view<float,1> avgradInput(Concurrency::extent<1>(gradInput->storage->size), THCudaTensor_data(gradInput));
    Concurrency::extent<1> grdExt(MULTIMARGIN_THREADS);
    Concurrency::tiled_extent<MULTIMARGIN_THREADS> t_ext(grdExt);
    float g = (float)(sizeaverage ? 1.0/((float)dim) : 1.0);
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<MULTIMARGIN_THREADS> tidx) restrict(amp) 
    {
        tile_static float buffer[MULTIMARGIN_THREADS];
        //int k = blockIdx.x;
        int k = tidx.tile[0];
        float *input_k = avInp.data();
        //intput_k += k*dim;
        float *gradInput_k = avgradInput.data();
        //gradInput += k*dim;
        int target_k = ((int)avTarget[k])-1;
        float input_target_k = input_k[target_k];

        //int i_start = threadIdx.x;
        int i_start = tidx.local[0];
        int i_end = dim;
        //int i_step = blockDim.x;
        int i_step = t_ext.tile_dim0;

        buffer[i_start] = 0;
        for(int i = i_start; i < i_end; i += i_step)
        {
            float z = 1 - input_target_k + input_k[i];
            if(i == target_k)
                continue;

            if(z > 0)
            {
                buffer[i_start] -= g;
                gradInput_k[i] = g;
            }
            else
                gradInput_k[i] = 0;
        }
        //__syncthreads();
        tidx.barrier.wait();

        // reduce
        if (i_start == 0)
        {
            float gradInput_target_k = 0;
            for (int i=0; i<i_step; i++)
                gradInput_target_k += buffer[i];
            gradInput_k[target_k] = gradInput_target_k;
        }
    });
}

static int cunn_MultiMarginCriterion_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  input = THCudaTensor_newContiguous(input);

  if(input->nDimension == 1)
  {
    float target_ = luaL_checknumber(L, 3);
    THCudaStorage *target = THCudaStorage_newWithSize(1);
    THCudaStorage *output = THCudaStorage_newWithSize(1);

    THCudaStorage_fill(target, target_);

    cunn_MultiMarginCriterion_updateOutput_kernel(output,
                                                                 input->storage,
                                                                 target,
                                                                 1, input->size[0],
                                                                 sizeaverage);
    lua_pushnumber(L, THCudaStorage_get(output, 0));

    THCudaStorage_free(output);
    THCudaStorage_free(target);
  }
  else if(input->nDimension == 2)
  {
    THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *output = THCudaTensor_newWithSize1d(input->size[0]);
    cunn_MultiMarginCriterion_updateOutput_kernel(output->storage,
                                                                 input->storage,
                                                                 target->storage,
                                                                 input->size[0], input->size[1],
                                                                 sizeaverage);
    lua_pushnumber(L, THCudaTensor_sumall(output));
    THCudaTensor_free(output);
  }
  else
    THError("vector or matrix expected");


  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  THCudaTensor_free(input);
  return 1;
}

static int cunn_MultiMarginCriterion_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor_resizeAs(gradInput, input);

  if(gradInput->nDimension == 1)
  {
    float target_ = luaL_checknumber(L, 3);
    THCudaTensor *target = THCudaTensor_newWithSize1d(1);

    THCudaTensor_fill(target, target_);

        cunn_MultiMarginCriterion_updateGradInput_kernel(gradInput, input, target, 1, gradInput->size[0], sizeaverage);

    THCudaTensor_free(target);
  }
  else if(gradInput->nDimension == 2)
  {
    THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

        cunn_MultiMarginCriterion_updateGradInput_kernel(gradInput, input, target, gradInput->size[0], gradInput->size[1], sizeaverage);
  }
  else
    THError("vector or matrix expected");

  return 1;
}

static const struct luaL_Reg cunn_MultiMarginCriterion__ [] = {
  {"MultiMarginCriterion_updateOutput", cunn_MultiMarginCriterion_updateOutput},
  {"MultiMarginCriterion_updateGradInput", cunn_MultiMarginCriterion_updateGradInput},
  {NULL, NULL}
};

static void cunn_MultiMarginCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_MultiMarginCriterion__, "nn");
  lua_pop(L,1);
}
