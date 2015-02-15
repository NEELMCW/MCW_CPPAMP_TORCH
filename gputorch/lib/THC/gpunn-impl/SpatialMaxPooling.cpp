
#define GPU_MAX_THREADS 256   // this is safe, in reality 256 is our limit

/*
 * Description:
 *    this function maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y 
 */

void maxpool(Concurrency::array_view<float,1> &input_data, long inOffset,
             Concurrency::array_view<float,1> &output_data, long outOffset,
             Concurrency::array_view<float,1> &indices_data, long indOffset,
             int nOutputCols, int nOutputRows,
             int input_n, int input_h, int input_w,
             int kH, int kW, int dH, int dW,
             int xblocks, int yblocks)
{
  Concurrency::extent<2> copyExt(yblocks*8,xblocks*32);
  Concurrency::tiled_extent<8,32> t_ext(copyExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<8,32> tidx) restrict(amp)
  {
    // iterators
    int xx, yy;

    // output size
    const int output_w = (input_w - kW) / dW + 1;
    const int output_h = (input_h - kH) / dH + 1;

    // compute offsets based on thread/block ID
    int o = tidx.tile[1];
    int i = o;

    int xx_start = tidx.local[1];
    int xx_end = output_w;
    const int xx_step = tidx.tile_dim1;

    int yy_start = tidx.global[0];
    int yy_end = output_h;
    const int yy_step = t_ext[0];

    // select input/output plane

    int output = o*output_w*output_h;
    int input = i*input_w*input_h;
    int indices_x = xblocks * nOutputCols * nOutputRows + o*output_w*output_h;
    int indices_y = o*output_w*output_h;
    // For all output pixels...
    for (yy = yy_start; yy < yy_end; yy+=yy_step) {
      for (xx = xx_start; xx < xx_end; xx+=xx_step) {
        // Compute the mean of the input image...
        int ptr_input = input + yy*dH*input_w + xx*dW;
        int ptr_output = output + yy*output_w + xx;
        int ptr_ind_x = indices_x + yy*output_w + xx;
        int ptr_ind_y = indices_y + yy*output_w + xx;

        int argmax_x = -1;
        int argmax_y = -1;
        float max = -FLT_MAX;
        int kx, ky;
        for (ky = 0; ky < kH; ky++) {
          for (kx = 0; kx < kW; kx++) {
            float val = input_data[inOffset + ptr_input + kx];
            if (val > max) {
              max = val;
              argmax_x = kx;
              argmax_y = ky;
            } 
          }
          ptr_input += input_w; // next input line
        }
        // Update output and argmax
        output_data[outOffset + ptr_output] = max;
        indices_data[indOffset + ptr_ind_x] = argmax_x + 1;
        indices_data[indOffset + ptr_ind_y] = argmax_y + 1;
      }
    }
  });
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
void maxgradinput(Concurrency::array_view<float,1> &input_data, long inOffset,
                  Concurrency::array_view<float,1> &output_data, long outOffset,
                  Concurrency::array_view<float,1> &indices_data, long indOffset,
                  int nOutputCols, int nOutputRows,
                  int input_n, int input_h, int input_w, 
                  int kH, int kW, int dH, int dW, int xblocks, int yblocks)
{
  Concurrency::extent<2> copyExt(yblocks*8,xblocks*32);
  Concurrency::tiled_extent<8,32> t_ext(copyExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<8,32> tidx) restrict(amp)
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
    const int xx_step = tidx.tile_dim1;

    int yy_start = tidx.tile_dim0*tidx.tile[0] + tidx.local[0];
    int yy_end = output_h;
    const int yy_step = t_ext[0];

    int gradOutput = o*output_w*output_h;
    int gradInput = i*input_w*input_h;
    int indices_x = xblocks * nOutputCols * nOutputRows + o*output_w*output_h;
    int indices_y = o*output_w*output_h;
    // compute gradInput
    for (yy = yy_start; yy < yy_end; yy+=yy_step) {
      for (xx = xx_start; xx < xx_end; xx+=xx_step) {
        int ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
        int ptr_gradOutput = gradOutput + yy*output_w + xx;
        int ptr_ind_x = indices_x + yy*output_w + xx;
        int ptr_ind_y = indices_y + yy*output_w + xx;

        float z = output_data[outOffset + ptr_gradOutput];
        int argmax_x = (int)indices_data[indOffset + ptr_ind_x]-1;
        int argmax_y = (int)indices_data[indOffset + ptr_ind_y]-1;

        input_data[indOffset + ptr_gradInput + argmax_x + argmax_y*input_w] += z;
      }
    }
  });
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 *    when kH != dH or kW != dW (uses atomic add)
 */
void atomicmaxgradinput(Concurrency::array_view<float,1> &input_data, long inOffset,
                        Concurrency::array_view<float,1> &output_data, long outOffset,
                        Concurrency::array_view<float,1> &indices_data, long indOffset,
                        int nOutputCols, int nOutputRows,
                        int input_n, int input_h, int input_w,
                        int kH, int kW, int dH, int dW, int xblocks, int yblocks
)
{
  Concurrency::extent<2> copyExt(yblocks,xblocks);
  Concurrency::tiled_extent<1,1> t_ext(copyExt);
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,1> tidx) restrict(amp)
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
    const int xx_step = tidx.tile_dim1;

    int yy_start = tidx.tile_dim0 *tidx.tile[0] + tidx.local[0];
    int yy_end = output_h;
    const int yy_step = t_ext[0];

    // select input/output plane
    int gradOutput = o*output_w*output_h;
    int gradInput = i*input_w*input_h;
    int indices_x = xblocks * nOutputCols * nOutputRows + o*output_w*output_h;
    int indices_y = o*output_w*output_h;

    // compute gradInput
    for (yy = yy_start; yy < yy_end; yy+=yy_step) {
      for (xx = xx_start; xx < xx_end; xx+=xx_step) {
        int ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
        int ptr_gradOutput = gradOutput + yy*output_w + xx;
        int ptr_ind_x = indices_x + yy*output_w + xx;
        int ptr_ind_y = indices_y + yy*output_w + xx;
        float z = output_data[outOffset + ptr_gradOutput];

        int argmax_x = (int)indices_data[indOffset + ptr_ind_x]-1;
        int argmax_y = (int)indices_data[indOffset + ptr_ind_y]-1;

        // atomic add since different threads could update same variable
        //Concurrency::atomic_fetch_add((int*)(&(ptr_gradInput[argmax_x + argmax_y*input_w])), (int)z);
        // TODO: need atomicAdd for float
        input_data[inOffset + ptr_gradInput + argmax_x + argmax_y*input_w] += z;
      }
    }
  });
}

static int gpunn_SpatialMaxPooling_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor *indices = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THGPUTensor_newContiguous(input);
    THGPUTensor_resize3d(output, nInputPlane, nOutputRows, nOutputCols);
    THGPUTensor_resize4d(indices, 2, nInputPlane, nOutputRows, nOutputCols);

    // blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    int xblocks = nInputPlane;
    
    // no need to sync from host for input & output & indices array_view
    PREPARE_AV(input, pavInput);
    PREPARE_AV(output, pavOutput);
    PREPARE_AV(indices, pavIndices);

    // run maxpool kernel
    maxpool(*pavInput, input->storageOffset,
            *pavOutput, output->storageOffset,
            *pavIndices, indices->storageOffset,
            nOutputCols, nOutputRows, nInputPlane, nInputRows, nInputCols, 
            kH, kW, dH, dW, xblocks, yblocks);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;

    luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THGPUTensor_newContiguous(input);
    THGPUTensor_resize4d(output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    THGPUTensor_resize5d(indices, 2, nbatch, nInputPlane, nOutputRows, nOutputCols);

    // blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    int xblocks = nInputPlane * nbatch;

    // no need to sync from host for input & output & indices array_view
    PREPARE_AV(input, pavInput);
    PREPARE_AV(output, pavOutput);
    PREPARE_AV(indices, pavIndices);    
    
    // run maxpool kernel
    maxpool(*pavInput, input->storageOffset,
            *pavOutput, output->storageOffset,
            *pavIndices, indices->storageOffset,
            nOutputCols, nOutputRows, nInputPlane, nInputRows, nInputCols,
            kH, kW, dH, dW, xblocks, yblocks);
  }

  // clean
  THGPUTensor_free(input);
  // check for errors
  return 1;
}

static int gpunn_SpatialMaxPooling_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  // FIXME: Check the argument as we dont support atomicAdd for float for now
  luaL_argcheck(L, dW == kW, 1, "dW and kW must be equal (this will be fixed soon)");
  luaL_argcheck(L, dH == kH, 1, "dH and kH must be equal (this will be fixed soon)");

  bool atomic = (dW != kW) || (dH != kH); 
  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  THGPUTensor *indices = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.GPUTensor");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = gradOutput->size[2];
    long nOutputRows = gradOutput->size[1];

    THGPUTensor_resizeAs(gradInput, input);
    THGPUTensor_zero(gradInput);

    // blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    int xblocks = nInputPlane;

    PREPARE_AV(gradInput, pavGradInput);
    PREPARE_AV(gradOutput, pavGradOutput);
    PREPARE_AV(indices, pavIndices);
    if(atomic)
    {
      atomicmaxgradinput(*pavGradInput, gradInput->storageOffset,
                         *pavGradOutput, gradOutput->storageOffset,
                         *pavIndices, indices->storageOffset,
                         nOutputCols, nOutputRows, nInputPlane, nInputRows, nInputCols, 
                         kH, kW, dH, dW, xblocks, yblocks);
    }
    else
    {
      // run updateGradInput kernel
      maxgradinput(*pavGradInput, gradInput->storageOffset,
                   *pavGradOutput, gradOutput->storageOffset,
                   *pavIndices, indices->storageOffset,
                   nOutputCols, nOutputRows, nInputPlane, nInputRows, nInputCols,
                   kH, kW, dH, dW, xblocks, yblocks);
    }
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = gradOutput->size[3];
    long nOutputRows = gradOutput->size[2];

    THGPUTensor_resizeAs(gradInput, input);
    THGPUTensor_zero(gradInput);

    PREPARE_AV(gradInput, pavGradInput);
    PREPARE_AV(gradOutput, pavGradOutput);
    PREPARE_AV(indices, pavIndices);

    // blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    int xblocks = nInputPlane * nbatch;

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
      atomicmaxgradinput(*pavGradInput, gradInput->storageOffset,
                         *pavGradOutput, gradOutput->storageOffset,
                         *pavIndices, indices->storageOffset,
                         nOutputCols, nOutputRows, nInputPlane, nInputRows, nInputCols,
                         kH, kW, dH, dW, xblocks, yblocks);
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically
      maxgradinput(*pavGradInput, gradInput->storageOffset,
                   *pavGradOutput, gradOutput->storageOffset,
                   *pavIndices, indices->storageOffset,
                   nOutputCols, nOutputRows, nInputPlane, nInputRows, nInputCols,
                   kH, kW, dH, dW, xblocks, yblocks);
    }
  }

  // check for errors
  return 1;
}

static const struct luaL_Reg gpunn_SpatialMaxPooling__ [] = {
  {"SpatialMaxPooling_updateOutput", gpunn_SpatialMaxPooling_updateOutput},
  {"SpatialMaxPooling_updateGradInput", gpunn_SpatialMaxPooling_updateGradInput},
  {NULL, NULL}
};

static void gpunn_SpatialMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialMaxPooling__, "nn");
  lua_pop(L,1);
}

#undef GPU_MAX_THREADS
