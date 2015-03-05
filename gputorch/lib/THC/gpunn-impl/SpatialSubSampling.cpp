#define GPU_MAX_THREADS 256

/*
 * Description:
 *    this function subsamples an input 3D tensor along dimensions 1 and 2
 *    3D input, 3D output, 1D weight, 1D bias
 */

void subsample(Concurrency::array_view<float,1> &avInput, long inOffset,
               Concurrency::array_view<float,1> &avOutput, long outOffset,
               Concurrency::array_view<float,1> &avWeight, long weightOffset,
               Concurrency::array_view<float,1> &avBias, long biasOffset,
               int input_n, int input_h, int input_w, 
               int kH, int kW, int dH, int dW, int xBlocks)
{
  // output size
  int yBlocks = (int)(16L / input_n);
  yBlocks = yBlocks < 1 ? 1 : yBlocks;
  Concurrency::extent<2> grdExt(yBlocks * 8 , xBlocks * 32);
  Concurrency::tiled_extent<8, 32> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<8, 32> tidx) restrict(amp)
  {
    float input = 0;
    float output= 0;
    float weight = 0;
    float bias = 0;
    // iterators
    int xx, yy;
    int output_w = (input_w - kW) / dW + 1;
    int output_h = (input_h - kH) / dH + 1;
    // compute offsets based on thread/block ID
    int o = tidx.tile[1];
    int i = o;
    int k = tidx.tile[1] % input_n;

    int xx_start = tidx.local[1];
    int xx_end = output_w;
    int xx_step = tidx.tile_dim1;

    int yy_start = tidx.global[0];
    int yy_end = output_h;
    int yy_step = t_ext[0];

    // select input/output plane
    output = output + o*output_w*output_h;
    input = input + i*input_w*input_h;

    // Get the good mask for (k,i) (k out, i in)
    float the_weight = avWeight[weightOffset + weight+ k];

    // Initialize to the bias
    float the_bias = avBias[biasOffset + bias+k];

    // For all output pixels...
    for (yy = yy_start; yy < yy_end; yy += yy_step)
    {
      for (xx = xx_start; xx < xx_end; xx += xx_step)
      {
        // Compute the mean of the input image...
        float ptr_input = input + yy * dH * input_w + xx * dW;
        float ptr_output = output + yy * output_w + xx;
        float sum = 0;
        int kx, ky;
        for (ky = 0; ky < kH; ky++)
        {
          for (kx = 0; kx < kW; kx++)
            sum += avInput[inOffset + ptr_input + kx];
          ptr_input += input_w; // next input line
        }
        // Update output
        avOutput[outOffset + ptr_output] = the_weight*sum + the_bias;
      }
    }
  });
}

/*
 * Description:
 *    this function computes the gradWeight from input and gradOutput
 */
void subgradweight(Concurrency::array_view<float,1> &avInput, long inOffset,
                   Concurrency::array_view<float,1> &avGradOutput, long gradOutOffset,
                   Concurrency::array_view<float,1> &avGradWeight, long gradWeightOffset,
                   Concurrency::array_view<float,1> &avGradBias, long gradBiasOffset,
                   long* Stride_Input, long* Stride_Output,
                   int input_n, int input_h, int input_w, int kH, int kW,
                   int dH, int dW, float scale, long sl)
{
  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;
  int inputStride = sl * Stride_Input[0];// inputTensor->stride[0];
  int outputStride = sl * Stride_Output[0];//gradOutputTensor->stride[0];

  int xBlocks = input_n;
  Concurrency::extent<2> grdExt(8 , xBlocks * 32);
  Concurrency::tiled_extent<8, 32> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<8, 32> tidx) restrict(amp)
  {
    // iterators
    int xx, yy;

    // compute offsets based on thread/block ID
    int o = tidx.tile[1];
    int i = o;
    int k = tidx.tile[1] % input_n;

    float input = (float)inputStride;
    float gradOutput= (float)outputStride;
    float gradWeight = 0;
    float gradBias = 0;

    int xx_start = tidx.local[1];
    int xx_end = output_w;
    int xx_step = tidx.tile_dim1;

    int yy_start = tidx.local[0];
    int yy_end = output_h;
    int yy_step = tidx.tile_dim0;

    // select input/output plane
    gradOutput = gradOutput + o*output_w*output_h;
    input = input + i*input_w*input_h;

    // thread ID
    int tid = tidx.tile_dim1 * tidx.local[0] + tidx.local[1];

    // create array to hold partial sums
    tile_static float sums[GPU_MAX_THREADS];
    sums[tid] = 0;

    // compute partial sums
    for (yy = yy_start; yy < yy_end; yy += yy_step)
    {
      for (xx = xx_start; xx < xx_end; xx += xx_step)
      {
        float ptr_input = input + yy * dH * input_w + xx * dW;
        float ptr_gradOutput = gradOutput + yy * output_w + xx;
        float z = avGradOutput[gradOutOffset + ptr_gradOutput];
        long kx, ky;
        for (ky = 0; ky < kH; ky++)
        {
          for (kx = 0; kx < kW; kx++)
          {
            sums[tid] += z * avInput[inOffset + ptr_input + kx];
          }
          ptr_input += input_w;
        }
      }
    }
    tidx.barrier.wait();

    // reduce: accumulate all partial sums to produce final gradWeight
    if ((tidx.local[1] == 0) && (tidx.local[0] == 0))
    {
      for (int i = 0; i < tidx.tile_dim1 * tidx.tile_dim0; i++)
        avGradWeight[gradWeightOffset + gradWeight + k] += scale * sums[i];
    }
    tidx.barrier.wait();

    // compute gradBias
    sums[tid] = 0;
    for (int i=tid; i<output_w*output_h; i+=(tidx.tile_dim1 * tidx.tile_dim0))
    {
      sums[tid] += avGradOutput[gradOutOffset + gradOutput + i];
    }
    tidx.barrier.wait();

    // reduce gradBias
    if ((tidx.local[1] == 0) && (tidx.local[0] == 0))
    {
      for (int i=0; i<(tidx.tile_dim1 * tidx.tile_dim0); i++)
        avGradBias[gradBiasOffset + gradBias + k] += scale * sums[i];
    }
  });
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
void subgradinput(Concurrency::array_view<float,1> &avGradInput, long gradInOffset,
                  Concurrency::array_view<float,1> &avGradOutput, long gradOutOffset,
                  Concurrency::array_view<float,1> &avWeight, long weightOffset,
                  int input_n, int input_h, int input_w,
                  int kH, int kW, int dH, int dW, int xBlocks)
{
  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  int yBlocks = (int)(16L / input_n);
  yBlocks = yBlocks < 1 ? 1 : yBlocks;
  Concurrency::extent<2> grdExt(yBlocks * 8 , xBlocks * 32);
  Concurrency::tiled_extent<8, 32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<8, 32> tidx) restrict(amp)
  {
    // iterators
    int xx, yy;
    float gradInput = 0;
    float gradOutput= 0;
    float weight = 0;

    // compute offsets based on thread/block ID
    int o = tidx.tile[1];
    int i = o;
    int k = tidx.tile[1] % input_n;

    int xx_start = tidx.local[1];
    int xx_end = output_w;
    int xx_step = tidx.tile_dim1;

    int yy_start = tidx.global[0];
    int yy_end = output_h;
    int yy_step = t_ext[0];

    // select input/output plane
    gradOutput = gradOutput + o * output_w * output_h;
    gradInput = gradInput + i * input_w * input_h;

    // get weight
    float the_weight = avWeight[weightOffset + weight + k];

    // compute gradInput
    for (yy = yy_start; yy < yy_end; yy += yy_step)
    {
      for (xx = xx_start; xx < xx_end; xx += xx_step)
      {
        float ptr_gradInput = gradInput + yy * dH * input_w + xx * dW;
        float ptr_gradOutput = gradOutput + yy * output_w + xx;
        float z = avGradOutput[gradOutOffset + ptr_gradOutput] * the_weight;
        int kx, ky;
        for (ky = 0; ky < kH; ky++)
        {
          for (kx = 0; kx < kW; kx++)
            avGradInput[gradInOffset + ptr_gradInput+kx] += z;
          ptr_gradInput += input_w;
        }
      }
    }
  });
}

static int gpunn_SpatialSubSampling_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THGPUTensor *weight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *bias = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "bias", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  auto avWeight = weight->get_array_view();
  auto avBias = bias->get_array_view();

  if (input->nDimension == 3)
  {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    luaL_argcheck(L, input->size[0] == nInputPlane, 2, "invalid number of input planes");
    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THGPUTensor_newContiguous(input);
    THGPUTensor_resize3d(output, nInputPlane, nOutputRows, nOutputCols);

    int xBlocks = nInputPlane;

    auto avInput = input->get_array_view();
    auto avOutput = output->get_array_view();


    subsample (avInput, input->storageOffset,
               avOutput, output->storageOffset,
               avWeight, weight->storageOffset,
               avBias, bias->storageOffset,
               nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);
  }
  else
  {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;
    int xBlocks = nInputPlane * nbatch;

    luaL_argcheck(L, input->size[1] == nInputPlane, 2, "invalid number of input planes");
    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THGPUTensor_newContiguous(input);
    THGPUTensor_resize4d(output, nbatch, nInputPlane, nOutputRows, nOutputCols);

    auto avInput = input->get_array_view();
    auto avOutput = output->get_array_view();

    subsample(avInput, input->storageOffset,
              avOutput, output->storageOffset,
              avWeight, weight->storageOffset,
              avBias, bias->storageOffset,
              nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);
  }

  // clean
  THGPUTensor_free(input);
  return 1;
}

static int gpunn_SpatialSubSampling_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  luaL_argcheck(L, dW == kW, 1, "dW and kW must be equal (this will be fixed soon)");
  luaL_argcheck(L, dH == kH, 1, "dH and kH must be equal (this will be fixed soon)");

  THGPUTensor *weight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  auto avGradOutput = gradOutput->get_array_view();
  auto avWeight = weight->get_array_view();

  if (input->nDimension == 3)
  {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    int xBlocks = nInputPlane;

    THGPUTensor_resizeAs(gradInput, input);
    THGPUTensor_zero(gradInput);

    auto avGradInput = gradInput->get_array_view();

    subgradinput (avGradInput, gradInput->storageOffset,
                  avGradOutput, gradOutput->storageOffset,
                  avWeight, weight->storageOffset,
                  nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);

  }
  else
  {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];
    int xBlocks = nInputPlane * nbatch;

    THGPUTensor_resizeAs(gradInput, input);
    THGPUTensor_zero(gradInput);

    auto avGradInput = gradInput->get_array_view();

    subgradinput (avGradInput, gradInput->storageOffset,
                  avGradOutput, gradOutput->storageOffset,
                  avWeight, weight->storageOffset,
                  nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);
  }
  return 1;
}

static int gpunn_SpatialSubSampling_accGradParameters(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  luaL_argcheck(L, dW == kW, 1, "dW and kW must be equal (this will be fixed soon)");
  luaL_argcheck(L, dH == kH, 1, "dH and kH must be equal (this will be fixed soon)");

  THGPUTensor *gradWeight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.GPUTensor");
  THGPUTensor *gradBias = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.GPUTensor");

  auto avGradWeight = gradWeight->get_array_view();
  auto avGradBias = gradBias->get_array_view();
  auto avGradOutput = gradOutput->get_array_view();

  if (input->nDimension == 3)
  {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long sl = 0;

    input = THGPUTensor_newContiguous(input);
    auto avInput = input->get_array_view();

    subgradweight (avInput, input->storageOffset,
                   avGradOutput, gradOutput->storageOffset,
                   avGradWeight, gradWeight->storageOffset,
                   avGradBias, gradBias->storageOffset,
                   input->stride, gradOutput->stride, 
                   nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, scale, sl);
  }
  else
  {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];

    input = THGPUTensor_newContiguous(input);
    auto avInput = input->get_array_view();

    // run gradweight kernel
    long sl;
    for (sl = 0; sl < nbatch; sl++)
    {
      subgradweight (avInput, input->storageOffset,
                     avGradOutput, gradOutput->storageOffset,
                     avGradWeight, gradWeight->storageOffset,
                     avGradBias, gradBias->storageOffset,
                     input->stride, gradOutput->stride,
                     nInputPlane, nInputRows, nInputCols,
                     kH, kW, dH, dW, scale, sl);
    }
  }

  // clean
  THGPUTensor_free(input);
  return 0;
}

static const struct luaL_Reg gpunn_SpatialSubSampling__ [] = {
  {"SpatialSubSampling_updateOutput", gpunn_SpatialSubSampling_updateOutput},
  {"SpatialSubSampling_updateGradInput", gpunn_SpatialSubSampling_updateGradInput},
  {"SpatialSubSampling_accGradParameters", gpunn_SpatialSubSampling_accGradParameters},
  {NULL, NULL}
};

static void gpunn_SpatialSubSampling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialSubSampling__, "nn");
  lua_pop(L,1);
}
