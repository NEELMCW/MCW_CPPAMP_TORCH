#define MINUS_LOG_THRESHOLD -18.42
#define LOGSOFTMAX_THREADS 128
#include "amp_math.h"


void cunn_LogSoftMax_updateOutput_kernel(THCudaTensor *output, THCudaTensor *input, int nframe, int dim)
{
    Concurrency::array_view<float,1> avInp(Concurrency::extent<1>(input->storage->size), THCudaTensor_data(input));
    Concurrency::array_view<float,1> avOutput(Concurrency::extent<1>(output->storage->size), THCudaTensor_data(output));
   // nframe = (nframe + (LOGSOFTMAX_THREADS -1)) &~(LOGSOFTMAX_THREADS-1);
    Concurrency::extent<1> grdExt(nframe * 128);
    Concurrency::tiled_extent<LOGSOFTMAX_THREADS> t_ext(grdExt);
    //std::cout<<"Update OutPut kernel invoked"<<std::endl;
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<LOGSOFTMAX_THREADS> tidx) restrict(amp) 
    {
        tile_static float buffer[LOGSOFTMAX_THREADS+1];
        //int k = blockIdx.x;
        int k = tidx.tile[0];

        //int i_start = threadIdx.x;
        unsigned int i_start = tidx.local[0];
        int i_end = dim;
        //int i_step = blockDim.x;
        int i_step = t_ext.tile_dim0;

        // max?
        buffer[i_start] = -FLT_MAX;
        for (int i=i_start; i<i_end; i+=i_step)
        {
        float z = avInp[k * dim +i];
        if(buffer[i_start] < z)
            buffer[i_start] = z;
        }

        // reduce
        for (unsigned int stride = i_step >> 1; stride > 0; stride >>= 1)
        {
            //__syncthreads();
            tidx.barrier.wait();
            if ((i_start < stride) && (buffer[i_start] < buffer[i_start + stride]))
                buffer[i_start] = buffer[i_start + stride];
        }
        if (i_start == 0)
        {
            float max_k = -FLT_MAX;
            if(max_k < buffer[0])
                max_k = buffer[0];
            buffer[LOGSOFTMAX_THREADS] = max_k;
        }

        //__syncthreads();
        tidx.barrier.wait();

        // logadd?
        float max_k = buffer[LOGSOFTMAX_THREADS];
        buffer[i_start] = 0;
        for (int i=i_start; i<i_end; i+=i_step)
            buffer[i_start] += Concurrency::fast_math::expf(avInp[k*dim+i]-max_k);

        // reduce
        for (unsigned int stride = i_step >> 1; stride > 0; stride >>= 1)
        {
            //__syncthreads();
            tidx.barrier.wait();
            if (i_start < stride)
                buffer[i_start] += buffer[i_start+stride];
        }
        if (i_start == 0)
            buffer[LOGSOFTMAX_THREADS] = max_k + Concurrency::fast_math::logf(buffer[0]);

        //__syncthreads();
        tidx.barrier.wait();
        // logsoftmax
        float logsum_k = buffer[LOGSOFTMAX_THREADS];
        for (int i=i_start; i<i_end; i+=i_step)
            avOutput[k *dim + i] = avInp[k * dim +i] - logsum_k;
    });
    avOutput.synchronize();
}

void cunn_LogSoftMax_updateGradInput_kernel(THCudaTensor *gradInput, THCudaTensor *output, THCudaTensor *gradOutput, int nframe, int dim)
{
    Concurrency::array_view<float,1> avGradInput(Concurrency::extent<1>(gradInput->storage->size), THCudaTensor_data(gradInput));
    Concurrency::array_view<float,1> avOutput(Concurrency::extent<1>(output->storage->size), THCudaTensor_data(output));
    Concurrency::array_view<float,1> avGradOutput(Concurrency::extent<1>(gradOutput->storage->size), THCudaTensor_data(gradOutput));
    Concurrency::extent<1> grdExt(nframe * LOGSOFTMAX_THREADS);
    Concurrency::tiled_extent<LOGSOFTMAX_THREADS> t_ext(grdExt);
    //std::cout<<"UpdateGradInputkernel invoked"<<std::endl;
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<LOGSOFTMAX_THREADS> tidx) restrict(amp) 
    {
        tile_static float buffer[LOGSOFTMAX_THREADS];
        //int k = blockIdx.x;
        int k = tidx.tile[0];
        float *gradInput_k = avGradInput.data();
        gradInput_k += k*dim;
        float *output_k = avOutput.data();
        output_k += k*dim;
        float *gradOutput_k = avGradOutput.data();
        gradOutput_k += k*dim;

        //int tx = threadIdx.x;
        unsigned int tx = tidx.local[0];

        int i_end = dim;
        //int i_step = blockDim.x;
        int i_step = t_ext.tile_dim0;

        // sum?
        buffer[tx] = 0;
        for (int i=tx; i<i_end; i+=i_step)
            buffer[tx] += gradOutput_k[i];

        // reduce
        for (unsigned int stride = t_ext.tile_dim0 >> 1; stride > 0; stride >>= 1)
        {
            //__syncthreads();
            tidx.barrier.wait();
            if (tx < stride)
                buffer[tx] += buffer[tx+stride];
        }

        //__syncthreads();
        tidx.barrier.wait();

        float sum_k = buffer[0];
        for (int i=tx; i<i_end; i+=i_step)
            gradInput_k[i] = gradOutput_k[i] - Concurrency::fast_math::expf(output_k[i])*sum_k;
    });
}

static int cunn_LogSoftMax_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  THCudaTensor *temp = input;
  input = THCudaTensor_newContiguous(input);
  //std::cout<<"Before logSoft resize"<<std::endl;
  THCudaTensor_resizeAs(output, input);

  if(input->nDimension == 1)
  {
    cunn_LogSoftMax_updateOutput_kernel(output, input, 1, input->size[0]);
  }
  else if(input->nDimension == 2)
  {
        cunn_LogSoftMax_updateOutput_kernel(output, input, input->size[0], input->size[1]);
  }
  else
  THError("vector or matrix expected");
  //std::cout<<"LogSoftMax finished"<<std::endl;
  

  if (input != temp)
  {
      THCudaTensor_free(input);
        input = NULL;
  }

  return 1;
}

static int cunn_LogSoftMax_updateGradInput(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor *tempOutput = output;
  THCudaTensor *tempGradOutput = gradOutput;
  output = THCudaTensor_newContiguous(output);
  gradOutput = THCudaTensor_newContiguous(gradOutput);
  //std::cout<<"inside logsoftmax input"<<std::endl;
  THCudaTensor_resizeAs(gradInput, output);

  if(gradInput->nDimension == 1)
  {
        cunn_LogSoftMax_updateGradInput_kernel(gradInput, output, gradOutput, 1, gradInput->size[0]);
  }
  else if(gradInput->nDimension == 2)
  {
        cunn_LogSoftMax_updateGradInput_kernel (gradInput, output, gradOutput, gradInput->size[0], gradInput->size[1]);
  }
  else
    THError("vector or matrix expected");

    if (output != tempOutput)
    {
        THCudaTensor_free(output);
        output = NULL;
    }

    if (gradOutput != tempGradOutput)
    {
        THCudaTensor_free(gradOutput);
        gradOutput = NULL;
    }

  return 1;
}

static const struct luaL_Reg cunn_LogSoftMax__ [] = {
  {"LogSoftMax_updateOutput", cunn_LogSoftMax_updateOutput},
  {"LogSoftMax_updateGradInput", cunn_LogSoftMax_updateGradInput},
  {NULL, NULL}
};

static void cunn_LogSoftMax_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_LogSoftMax__, "nn");
  lua_pop(L,1);
}
