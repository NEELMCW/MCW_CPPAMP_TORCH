#include "luaT.h"
#include "THC.h"


/*
 * Description:
 */

int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) restrict(amp)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w / scale_factor;
  z = z / scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x * d1 + y) * d2) + z) * d3 + w;
}
int translate_idx_inv(int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) restrict(amp)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii / d3;
  z = ii % d2;
  ii = ii / d2;
  y = ii % d1;
  ii = ii / d1;
  x = ii;
  w = w * scale_factor + off_x;
  z = z * scale_factor + off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x * d1 + y) * d2) + z) * d3 + w;
}

void upscale(Concurrency::array_view<float,1> &avInp, long inpOffset,
             Concurrency::array_view<float,1> &avOut, long outOffset,
             unsigned int inpSz, unsigned int outSz, long no_elements,
             int scale_factor, int d1, int d2, int d3, unsigned int grdConf[])
{
  Concurrency::extent<2> grdExt(grdConf[1],grdConf[0]*256);
  Concurrency::tiled_extent<1,256> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,256> tidx) restrict(amp) 
  {
    long ii = tidx.global[1];
    ii += tidx.local[0] + t_ext.tile_dim0 * (t_ext.tile_dim1 * t_ext[1]) * tidx.tile[0];
    if (ii >= no_elements) return;
    int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
    avOut[outOffset + ii] = avInp[inpOffset + ipidx];
  });
}


static int gpunn_SpatialUpSamplingNearest_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");
  THGPUTensor_zero(output);
  int scale_factor = luaT_getfieldcheckint(L, 1, "scale_factor");

  input = THGPUTensor_newContiguous(input);
  // This is for allocating output Tensor
  long no_elements = 1;
  for (int i = 0; i < input->nDimension; i++)
  {
    no_elements *= input->size[i];
  }
  no_elements *= scale_factor * scale_factor;

  int d1;
  int d2;
  int d3;

  if (input->nDimension == 3)
  {
    d1 = output->size[0];
    d2 = output->size[1];
    d3 = output->size[2];
  }
  else
  {
    d1 = output->size[1];
    d2 = output->size[2];
    d3 = output->size[3];
  }

  // blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/GPU
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = std::min(std::max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535)
  {
    THError("Input size is too large!  aborting");
  }

  unsigned int grdConf[2];

  grdConf[0] = n_xblocks;
  grdConf[1] = n_yblocks;

  unsigned int inpSz = THGPUTensor_nElement(input);
  unsigned int outSz = THGPUTensor_nElement(output);
  
  PREPARE_AV(input, pavInput);
  PREPARE_AV(output, pavOutput);
  // kernel:
  upscale(*pavInput, input->storageOffset,
          *pavOutput, output->storageOffset,
          inpSz, outSz, no_elements, scale_factor, d1, d2, d3, grdConf);
 
  // final cut:
  THGPUTensor_free(input); 

  return 1;
}

/*
 * Description:
 */
void downscale(Concurrency::array_view<float,1> &avInp, long inpOffset,
               Concurrency::array_view<float,1> &avOut, long outOffset,
               unsigned int gradInpSz, unsigned int gradOutSz, long no_elements,
               int scale_factor, int d1, int d2, int d3, unsigned int gridConf[])
{
  Concurrency::extent<2> grdExt(gridConf[1],gridConf[0]*256);
  Concurrency::tiled_extent<1,256> t_ext(grdExt);
  // output offset:
  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<1,256> tidx) restrict(amp) 
  {
    long ii = tidx.global[1];
    ii += tidx.local[0] + t_ext.tile_dim0 * (t_ext.tile_dim1 * t_ext[1]) * tidx.tile[0];
    if (ii >= no_elements) return;
    for (int i=0; i < scale_factor; i++)
    {
      for (int j=0; j < scale_factor; j++)
      {
        int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
        avInp[inpOffset + ii] += avOut[outOffset + ipidx];
      }
    }
  });
}


static int gpunn_SpatialUpSamplingNearest_updateGradInput(lua_State *L)
{
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradInput  = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  int scale_factor = luaT_getfieldcheckint(L, 1, "scale_factor");

  THGPUTensor_zero(gradInput);

  long no_elements = 1;
  for (int i = 0; i < gradInput->nDimension; i++)
  {
    no_elements *= gradInput->size[i];
  }

  int d1;
  int d2;
  int d3;

  if (gradInput->nDimension == 3)
  {
    d1 = gradInput->size[0];
    d2 = gradInput->size[1];
    d3 = gradInput->size[2];
  }
  else
  {
    d1 = gradInput->size[1];
    d2 = gradInput->size[2];
    d3 = gradInput->size[3];
  }

  // blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/GPU
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = std::min(std::max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) 
  {
    THError("Input size is too large!  aborting");
  }

  unsigned int gradConf[2];
  gradConf[0] = n_xblocks;
  gradConf[1] = n_yblocks;

  //dim3 blocks(n_xblocks, n_yblocks);
  //dim3 threads(nthreads);

  PREPARE_AV(gradInput, pavGradInput);
  PREPARE_AV(gradOutput, pavGradOutput);
  // kernel:
  downscale(*pavGradInput, gradInput->storageOffset,
            *pavGradOutput, gradOutput->storageOffset,
            THGPUTensor_nElement(gradInput), THGPUTensor_nElement(gradOutput),
            no_elements, scale_factor, d1, d2, d3, gradConf);

  return 1;
}

static const struct luaL_Reg gpunn_SpatialUpSamplingNearest__ [] = {
  {"SpatialUpSamplingNearest_updateOutput", gpunn_SpatialUpSamplingNearest_updateOutput},
  {"SpatialUpSamplingNearest_updateGradInput", gpunn_SpatialUpSamplingNearest_updateGradInput},
  {NULL, NULL}
};

void gpunn_SpatialUpSamplingNearest_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialUpSamplingNearest__, "nn");
  lua_pop(L,1);
}
