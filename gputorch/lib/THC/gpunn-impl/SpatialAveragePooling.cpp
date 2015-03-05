#define GPU_MAX_THREADS 256

/*
 * Description:
 *    this function avg-pools an input 3D tensor along dimensions 1 and 2
 *    3D input, 3D output
 */
void subsample(Concurrency::array_view<float,1> &avInput, long inOffset,
               Concurrency::array_view<float,1> &avOutput, long outOffset,
               int input_n, int input_h, int input_w,
               int kH, int kW, int dH, int dW, int xBlocks)
{
  int yBlocks = (int)(16L / input_n);
  yBlocks = yBlocks < 1 ? 1 : yBlocks;
  Concurrency::extent<2> grdExt(yBlocks * 8 , xBlocks * 32);
  Concurrency::tiled_extent<8, 32> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<8, 32> tidx) restrict(amp)
  {
    // iterators
    int xx, yy;

    // output size
    int output_w = (input_w - kW) / dW + 1;
    int output_h = (input_h - kH) / dH + 1;

    // compute offsets based on thread/block ID
    int o = tidx.tile[1];
    int i = o;

    int xx_start = tidx.local[1];
    int xx_end = output_w;
    int xx_step = tidx.tile_dim1;

    int yy_start = tidx.global[0];
    int yy_end = output_h;
    int yy_step = t_ext[0];

    // select input/output plane
    int output = o * output_w * output_h;
    int input = i * input_w * input_h;

    // For all output pixels...
    for(yy = yy_start; yy < yy_end; yy += yy_step)
    {
      for(xx = xx_start; xx < xx_end; xx += xx_step)
      {
        // Compute the mean of the input image...
        int ptr_input = input + yy * dH * input_w + xx * dW;
        int ptr_output = output + yy * output_w + xx;
        float sum = 0;
        int kx, ky;
        for(ky = 0; ky < kH; ky++)
        {
          for(kx = 0; kx < kW; kx++)
            sum += avInput[inOffset + ptr_input + kx];

          ptr_input += input_w; // next input line
        }
        // Update output
        avOutput[outOffset + ptr_output] = sum;
      }
    }
  });
}

static int gpunn_SpatialAveragePooling_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3)
  {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;
    long nInputPlane = input->size[0];

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");
    input = THGPUTensor_newContiguous(input);
    THGPUTensor_resize3d(output, nInputPlane, nOutputRows, nOutputCols);

    int xBlocks = nInputPlane;

    auto avInput = input->get_array_view();
    auto avOutput = output->get_array_view();
    // run subsample kernel
    subsample (avInput, input->storageOffset, avOutput, output->storageOffset,
               nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);
  }
  else
  {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;
    long nInputPlane = input->size[1];

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");
    input = THGPUTensor_newContiguous(input);
    THGPUTensor_resize4d(output, nbatch, nInputPlane, nOutputRows, nOutputCols);

    int xBlocks = nInputPlane * nbatch;

    auto avInput = input->get_array_view();
    auto avOutput = output->get_array_view();
    // run subsample kernel
    subsample (avInput, input->storageOffset, avOutput, output->storageOffset,
               nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);
  }

  // clean
  THGPUTensor_free(input);
  return 1;
}

/*
 * Description:
 *    this function computes the gradInput from gradOutput
 */
void subgradinput(Concurrency::array_view<float,1> &avGradInput, long gradInOffset,
                  Concurrency::array_view<float,1> &avGradOutput, long gradOutOffset,
                  int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW, int xBlocks)
{
  int yBlocks = (int)(16L / input_n);
  yBlocks = yBlocks < 1 ? 1 : yBlocks;
  Concurrency::extent<2> grdExt(yBlocks * 8 , xBlocks * 32);
  Concurrency::tiled_extent<8, 32> t_ext(grdExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<8, 32> tidx) restrict(amp)
  {
    // iterators
    int xx, yy;

    // output size
    int output_w = (input_w - kW) / dW + 1;
    int output_h = (input_h - kH) / dH + 1;

    // compute offsets based on thread/block ID
    int o = tidx.tile[1];
    int i = o;

    int xx_start = tidx.local[1];
    int xx_end = output_w;
    int xx_step = tidx.tile_dim1;

    int yy_start = tidx.global[0];
    int yy_end = output_h;
    int yy_step = t_ext[0];

    // select input/output plane
    int gradOutput = o * output_w * output_h;
    int gradInput = i * input_w * input_h;

    // compute gradInput
    for(yy = yy_start; yy < yy_end; yy += yy_step)
    {
      for(xx = xx_start; xx < xx_end; xx += xx_step)
      {
        int ptr_gradInput = gradInput + yy * dH * input_w + xx * dW;
        int ptr_gradOutput = gradOutput + yy * output_w + xx;
        float z = avGradOutput[gradOutOffset + ptr_gradOutput];
        int kx, ky;
        for(ky = 0; ky < kH; ky++)
        {
          for(kx = 0; kx < kW; kx++)
            avGradInput[gradInOffset + ptr_gradInput + kx] += z;

          ptr_gradInput += input_w;
        }
      }
    }
  });
}


static int gpunn_SpatialAveragePooling_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  luaL_argcheck(L, dW == kW, 1, "dW and kW must be equal (this will be fixed soon)");
  luaL_argcheck(L, dH == kH, 1, "dH and kH must be equal (this will be fixed soon)");

  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  if (input->nDimension == 3)
  {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];

    THGPUTensor_resizeAs(gradInput, input);
    THGPUTensor_zero(gradInput);

    int xBlocks = nInputPlane;

    auto avGradInput = gradInput->get_array_view();
    auto avGradOutput = gradOutput->get_array_view();
    // run updateGradInput kernel
    subgradinput(avGradInput, gradInput->storageOffset,
                 avGradOutput, gradOutput->storageOffset,
                 nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);
  }
  else
  {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];

    THGPUTensor_resizeAs(gradInput, input);
    THGPUTensor_zero(gradInput);

    int xBlocks = nInputPlane * nbatch;
    auto avGradInput = gradInput->get_array_view();
    auto avGradOutput = gradOutput->get_array_view();
    // run updateGradInput kernel
    subgradinput(avGradInput, gradInput->storageOffset,
                 avGradOutput, gradOutput->storageOffset,
                 nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW, xBlocks);
  }
  return 1;
}

static const struct luaL_Reg gpunn_SpatialAveragePooling__ [] = {
  {"SpatialAveragePooling_updateOutput", gpunn_SpatialAveragePooling_updateOutput},
  {"SpatialAveragePooling_updateGradInput", gpunn_SpatialAveragePooling_updateGradInput},
  {NULL, NULL}
};

static void gpunn_SpatialAveragePooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialAveragePooling__, "nn");
  lua_pop(L,1);
}
