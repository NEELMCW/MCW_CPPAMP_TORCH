#define MINUS_LOG_THRESHOLD -18.42
#define LOGSOFTMAX_THREADS 256
#include "amp_math.h"

void gpunn_LogSoftMax_updateOutput_kernel(Concurrency::array_view<float,1> &avOutput, long outOffset,
                                          Concurrency::array_view<float,1> &avInp, long inOffset,
                                          int nframe, int dim)
{
  Concurrency::extent<1> grdExt(nframe * LOGSOFTMAX_THREADS);
  Concurrency::tiled_extent<LOGSOFTMAX_THREADS> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<LOGSOFTMAX_THREADS> tidx) restrict(amp)
  {
    tile_static float buffer[LOGSOFTMAX_THREADS+1];
    int k = tidx.tile[0];

    unsigned int i_start = tidx.local[0];
    int i_end = dim;
    int i_step = t_ext.tile_dim0;

    // max?
    buffer[i_start] = -FLT_MAX;
    for (int i = i_start; i < i_end; i += i_step)
    {
      float z = avInp[inOffset + k * dim + i];
      if(buffer[i_start] < z)
        buffer[i_start] = z;
    }

    // reduce
    for (unsigned int stride = i_step >> 1; stride > 0; stride >>= 1)
    {
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

    tidx.barrier.wait();
    // logadd?
    float max_k = buffer[LOGSOFTMAX_THREADS];
    buffer[i_start] = 0;
    for (int i = i_start; i < i_end; i += i_step)
      buffer[i_start] += Concurrency::fast_math::expf(avInp[inOffset + k * dim + i] - max_k);

    // reduce
    for (unsigned int stride = i_step >> 1; stride > 0; stride >>= 1)
    {
      tidx.barrier.wait();
      if (i_start < stride)
        buffer[i_start] += buffer[i_start+stride];
    }

    if (i_start == 0)
      buffer[LOGSOFTMAX_THREADS] = max_k + Concurrency::fast_math::logf(buffer[0]);

    tidx.barrier.wait();
    // logsoftmax
    float logsum_k = buffer[LOGSOFTMAX_THREADS];
    for (int i = i_start; i < i_end; i += i_step)
      avOutput[outOffset + k *dim + i] = avInp[inOffset + k * dim +i] - logsum_k;
  });
}

void gpunn_LogSoftMax_updateGradInput_kernel(Concurrency::array_view<float,1> &avGradInput, long gradInOffset,
                                             Concurrency::array_view<float,1> &avOutput, long outOffset, 
                                             Concurrency::array_view<float,1> &avGradOutput, long gradOutOffset,
                                             int nframe, int dim)
{
  Concurrency::extent<1> grdExt(nframe * LOGSOFTMAX_THREADS);
  Concurrency::tiled_extent<LOGSOFTMAX_THREADS> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<LOGSOFTMAX_THREADS> tidx) restrict(amp)
  {
    tile_static float buffer[LOGSOFTMAX_THREADS];
    int k = tidx.tile[0];
    float *gradInput_k = avGradInput.data() + gradInOffset;
    gradInput_k += k * dim;
    float *output_k = avOutput.data() + outOffset;
    output_k += k * dim;
    float *gradOutput_k = avGradOutput.data() + gradOutOffset;
    gradOutput_k += k * dim;

    unsigned int tx = tidx.local[0];

    int i_end = dim;
    int i_step = t_ext.tile_dim0;

    // sum?
    buffer[tx] = 0;
    for (int i = tx; i < i_end; i += i_step)
      buffer[tx] += gradOutput_k[gradOutOffset + i];

    // reduce
    for (unsigned int stride = t_ext.tile_dim0 >> 1; stride > 0; stride >>= 1)
    {
      tidx.barrier.wait();
      if (tx < stride)
        buffer[tx] += buffer[tx+stride];
    }

    tidx.barrier.wait();
    float sum_k = buffer[0];
    for (int i = tx; i < i_end; i += i_step)
      gradInput_k[gradInOffset + i] = gradOutput_k[gradOutOffset + i] - Concurrency::fast_math::expf(output_k[outOffset + i]) * sum_k;
  });
}

static int gpunn_LogSoftMax_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  input = THGPUTensor_newContiguous(input);
  THGPUTensor_resizeAs(output, input);

  auto avOutput = output->get_array_view();
  auto avInput = input->get_array_view();

  if (input->nDimension == 1)
  {
    gpunn_LogSoftMax_updateOutput_kernel(avOutput, output->storageOffset,
                                         avInput, input->storageOffset,
                                         1, input->size[0]);
  }
  else if (input->nDimension == 2)
  {
    gpunn_LogSoftMax_updateOutput_kernel(avOutput, output->storageOffset,
                                         avInput, input->storageOffset,
                                         input->size[0], input->size[1]);
  }
  else
    THError("vector or matrix expected");

  THGPUTensor_free(input);
  return 1;
}

static int gpunn_LogSoftMax_updateGradInput(lua_State *L)
{
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  output = THGPUTensor_newContiguous(output);
  gradOutput = THGPUTensor_newContiguous(gradOutput);

  THGPUTensor_resizeAs(gradInput, output);

  auto avGradInput = gradInput->get_array_view();
  auto avGradOutput = gradOutput->get_array_view();
  auto avOutput = output->get_array_view();

  if (gradInput->nDimension == 1)
  {
    gpunn_LogSoftMax_updateGradInput_kernel(avGradInput, gradInput->storageOffset,
                                            avOutput, output->storageOffset,
                                            avGradOutput, gradOutput->storageOffset,
                                            1, gradInput->size[0]);
  }
  else if (gradInput->nDimension == 2)
  {
    gpunn_LogSoftMax_updateGradInput_kernel (avGradInput, gradInput->storageOffset,
                                             avOutput, output->storageOffset,
                                             avGradOutput, gradOutput->storageOffset,
                                             gradInput->size[0], gradInput->size[1]);
  }
  else
    THError("vector or matrix expected");

  THGPUTensor_free(output);
  THGPUTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg gpunn_LogSoftMax__ [] = {
  {"LogSoftMax_updateOutput", gpunn_LogSoftMax_updateOutput},
  {"LogSoftMax_updateGradInput", gpunn_LogSoftMax_updateGradInput},
  {NULL, NULL}
};

static void gpunn_LogSoftMax_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_LogSoftMax__, "nn");
  lua_pop(L,1);
}
