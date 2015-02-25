/*
 * Description:
 *    this function finds the max along the innermost dimension
 *    Nd input, (N-1)d output, (N-1)d argmax
 */
void max_output(Concurrency::array_view<float, 1> &avInp, long inpOffset,
               Concurrency::array_view<float,1> &avOut, long outOffset,
               Concurrency::array_view<float,1> &avInD, long indOffset,
               unsigned int inpSz, unsigned int outSz,
               unsigned int indSz, long nrows, long ncols, unsigned int numBlocks)
{
  Concurrency::extent<1> grdExt(numBlocks * 256);
  Concurrency::tiled_extent<256> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp)
  {
    long o = tidx.global[0];
    if (o >= nrows) return;

    long i = o * ncols;

    // compute max:
    float max = avInp[inpOffset + i];
    long argmax = 0;
    long ii;
    for (ii = 1; ii < ncols; ii++)
    {
      float val = avInp[inpOffset + i + ii];
      if (val > max)
      {
        max = val;
        argmax = ii;
      }
    }
    // store
    avOut[outOffset + o] = max;
    avInD[indOffset + o] =(float) argmax + 1;
  });
}

void max_gradInput(Concurrency::array_view<float, 1> &avInp, long inpOffset,
                  Concurrency::array_view<float,1> &avOut, long outOffset,
                  Concurrency::array_view<float,1> &avInD, long indOffset,
                  unsigned int inputSz, unsigned int outSz,
                  unsigned int indSz, long nrows, long ncols, unsigned int numBlocks)
{
  Concurrency::extent<1> grdExt(numBlocks * 256);
  Concurrency::tiled_extent<256> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<256> tidx) restrict(amp)
  {
    long o = tidx.global[0];
    if (o >= nrows) return;

    // input offset:
    long i = o * ncols;

    // bprop max gradient:
    long idx = (long)avInD[indOffset + o] - 1;
    avInp[inpOffset + i + idx] = avOut[outOffset + o];
  });
}

static int gpunn_Max_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  int dimension = luaT_getfieldcheckint(L, 1, "dimension") - 1;
  THGPUTensor *indices = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == input->nDimension - 1, 2, "only supported dimension is innermost (GPU kernel only)");

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

  // blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);

  PREPARE_AV(input, pavInput);
  PREPARE_AV(output, pavOutput);
  PREPARE_AV(indices, pavIndices);
  // kernel:
  max_output(*pavInput, input->storageOffset,
             *pavOutput, output->storageOffset,
             *pavIndices, indices->storageOffset,
             THGPUTensor_nElement(input), THGPUTensor_nElement(output),
             THGPUTensor_nElement(indices), nrows, ncols, nblocks);

  // final cut:
  //  (1) Either for it is cloned, need to free
  //  (2) Or it is retained, need to free once as well to reduce its reference count
  THGPUTensor_free(input);
  THGPUTensor_select(output, NULL, dimension, 0);

  return 1;
}

static int gpunn_Max_updateGradInput(lua_State *L)
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

  // blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);

  PREPARE_AV(gradInput, pavGradInput);
  PREPARE_AV(gradOutput, pavGradOutput);
  PREPARE_AV(indices, pavIndices);
  // kernel:
  max_gradInput(*pavGradInput, gradInput->storageOffset,
                *pavGradOutput, gradOutput->storageOffset,
                *pavIndices, indices->storageOffset,
                THGPUTensor_nElement(gradInput), THGPUTensor_nElement(gradOutput),
                THGPUTensor_nElement(indices), nrows, ncols, nblocks);

  return 1;
}

static const struct luaL_Reg gpunn_Max__ [] = {
  {"Max_updateOutput", gpunn_Max_updateOutput},
  {"Max_updateGradInput", gpunn_Max_updateGradInput},
  {NULL, NULL}
};

static void gpunn_Max_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_Max__, "nn");
  lua_pop(L,1);
}
