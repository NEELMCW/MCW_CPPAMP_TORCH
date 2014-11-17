
/*
 * Description:
 *    this function finds the max along the innermost dimension
 *    Nd input, (N-1)d output, (N-1)d argmax
 */
void max_output(float *input, float *output, float *indices, unsigned int inpSz, unsigned int outSz, unsigned int indSz, long nrows, long ncols, unsigned int numBlocks)
{
  // output offset:
    Concurrency::array_view<float,1> avInp(inpSz, input);
    Concurrency::array_view<float,1> avOut(outSz, output);
    Concurrency::array_view<float,1> avInD(indSz, indices);
    Concurrency::extent<1> grdExt(numBlocks*256);
    Concurrency::tiled_extent<256> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp) 
    {
        long o = tidx.global[0];
        if (o >= nrows) return;
        // input offset:
        long i = o * ncols;
        // move pointers
        //input = input + i;
        // compute max:
        float max = avInp[i];
        long argmax = 0;
        long ii;
        for (ii=1; ii<ncols; ii++) 
        {
            float val = avInp[i+ii];
            if (val > max) 
            {
                max = val;
                argmax = ii;
            }
        }
        // store
        avOut[o] = max;
        avInD[o] =(float) argmax+1;
    });
    avOut.synchronize();
    avInD.synchronize();
}


void max_gradInput(float *input, float *output, float *indices, unsigned int inputSz, unsigned int outSz, unsigned int indSz,
                              long nrows, long ncols, unsigned int numBlocks)
{
    // output offset:
    Concurrency::array_view<float,1> avInp(inputSz, input);
    Concurrency::array_view<float,1> avOut(outSz, output);
    Concurrency::array_view<float,1> avInD(indSz, indices);
    Concurrency::extent<1> grdExt(numBlocks*256);
    Concurrency::tiled_extent<256> t_ext(grdExt);

    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp) 
    {
        long o = tidx.global[0];
        if (o >= nrows) return;

        // input offset:
        long i = o * ncols;

        // bprop max gradient:
        long idx = (long)avInD[o]-1;
        avInp[i+idx] = avOut[o];
    });
    avInp.synchronize();
}

static int cunn_Max_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == input->nDimension-1, 2, "only supported dimension is innermost (CUDA kernel only)");

  input = THCudaTensor_newContiguous(input);

  THLongStorage *dim = THLongStorage_newWithSize(input->nDimension);
  long i;
  for(i = 0; i < input->nDimension; i++)
    dim->data[i] = input->size[i];
  dim->data[dimension] = 1;
  THCudaTensor_resize(output, dim, NULL);
  THCudaTensor_resize(indices, dim, NULL);
  THLongStorage_free(dim);

  float *input_data = THCudaTensor_data(input);
  float *output_data = THCudaTensor_data(output);
  float *indices_data = THCudaTensor_data(indices);

  long nrows = THCudaTensor_nElement(output);
  long ncols = input->size[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);

  // kernel:
    max_output(input_data, output_data, indices_data, THCudaTensor_nElement(input), THCudaTensor_nElement(output), THCudaTensor_nElement(indices), nrows, ncols, nblocks);

 
  // final cut:
  THCudaTensor_free(input); 
  THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}

static int cunn_Max_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *gradInput  = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_zero(gradInput);
  
  float *gradInput_data = THCudaTensor_data(gradInput);
  float *gradOutput_data = THCudaTensor_data(gradOutput);
  float *indices_data = THCudaTensor_data(indices);
  
  long nrows = THCudaTensor_nElement(gradOutput);
  long ncols = gradInput->size[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  
  // kernel:
    max_gradInput(gradInput_data, gradOutput_data, indices_data, THCudaTensor_nElement(gradInput), THCudaTensor_nElement(gradOutput), THCudaTensor_nElement(indices), nrows, ncols, nblocks);


  return 1;
}

static const struct luaL_Reg cunn_Max__ [] = {
  {"Max_updateOutput", cunn_Max_updateOutput},
  {"Max_updateGradInput", cunn_Max_updateGradInput},
  {NULL, NULL}
};

static void cunn_Max_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Max__, "nn");
  lua_pop(L,1);
}
