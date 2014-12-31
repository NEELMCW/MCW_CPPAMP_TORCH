
/*
 * Description:
 *    this function finds the min along the innermost dimension
 *    Nd input, (N-1)d output, (N-1)d argmin
 */
void min_output(Concurrency::array_view<float, 1> &avInp, Concurrency::array_view<float,1> &avOut, Concurrency::array_view<float,1> &avInD, unsigned int inpSz, unsigned int outSz,
               unsigned int indSz, long nrows, long ncols, unsigned int numBlocks)
{
  // output offset:
  Concurrency::extent<1> grdExt(numBlocks * 256);
  Concurrency::tiled_extent<256> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp) 
  {
    long o = tidx.global[0];
    if (o >= nrows) return;
    long i = o * ncols;
    // compute min:
    float min = avInp[i];
    long argmin = 0;
    long ii;
    for (ii = 1; ii < ncols; ii++) 
    {
      float val = avInp[ii + i];
      if (val < min) 
      {
        min = val;
        argmin = ii;
      }
    }
    // store
    avOut[o] = min;
    avInD[o] =(float) argmin + 1;
  });
}

void min_gradInput(Concurrency::array_view<float, 1> &avInp, Concurrency::array_view<float,1> &avOut, Concurrency::array_view<float,1> &avInD, unsigned int inputSz, unsigned int outSz,
                  unsigned int indSz, long nrows, long ncols, unsigned int numBlocks)
{
  // output offset:
  Concurrency::extent<1> grdExt(numBlocks * 256);
  Concurrency::tiled_extent<256> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp) 
  {
    long o = tidx.global[0];
    if (o >= nrows) return;

    // input offset:
    long i = o * ncols;

    // bprop min gradient:
    long idx = (long)avInD[o] - 1;
    avInp[i + idx] = avOut[o];
  });
}

static int gpunn_Min_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  int dimension = luaT_getfieldcheckint(L, 1, "dimension") - 1;
  THGPUTensor *indices = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == input->nDimension - 1, 2, "only supported dimension is innermost (GPU kernel only)");
  THGPUTensor* input_orig = input;
  input = THGPUTensor_newContiguous(input);

  THLongStorage *dim = THLongStorage_newWithSize(input->nDimension);
  long i;
  for (i = 0; i < input->nDimension; i++)
    dim->data[i] = input->size[i];
  dim->data[dimension] = 1;
  THGPUTensor_resize(output, dim, NULL);
  THGPUTensor_resize(indices, dim, NULL);
  THLongStorage_free(dim);

  long nrows = THGPUTensor_nElement(output);
  long ncols = input->size[dimension];

  // gpu blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);

  PREPARE_AV(input, pavInput);
  PREPARE_AV(output, pavOutput);
  PREPARE_AV(indices, pavIndices);
  // kernel:
  min_output(*pavInput, *pavOutput, *pavIndices, THGPUTensor_nElement(input),
             THGPUTensor_nElement(output), THGPUTensor_nElement(indices), nrows, ncols, nblocks);

  // final cut:
  if (input_orig != input) {
    THGPUTensor_free(input);
    input = NULL;
  }
  THGPUTensor_select(output, NULL, dimension, 0);

  return 1;
}

static int gpunn_Min_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *indices = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.GPUTensor");
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension") - 1;
  THGPUTensor *gradInput  = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  THGPUTensor_resizeAs(gradInput, input);
  THGPUTensor_zero(gradInput);

  long nrows = THGPUTensor_nElement(gradOutput);
  long ncols = gradInput->size[dimension];

  // gpu blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);

  PREPARE_AV(gradInput, pavGradInput);
  PREPARE_AV(gradOutput, pavGradOutput);
  PREPARE_AV(indices, pavIndices);
  // kernel:
  min_gradInput(*pavGradInput, *pavGradOutput, *pavIndices, THGPUTensor_nElement(gradInput),
                THGPUTensor_nElement(gradOutput), THGPUTensor_nElement(indices), nrows, ncols, nblocks);

  return 1;
}

static const struct luaL_Reg gpunn_Min__ [] = {
  {"Min_updateOutput", gpunn_Min_updateOutput},
  {"Min_updateGradInput", gpunn_Min_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Min_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Min__, "nn");
  lua_pop(L,1);
}
