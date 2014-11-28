
#include "SpatialPoolingGPU/updateOutput.cpp"
#include "SpatialPoolingGPU/updateGradInput.cpp"

static int gpunn_SpatialMaxPoolingGPU_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 4, 2, "4D (batch) tensor expected");

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nInputPlane = input->size[0];
  long batchSize = input->size[3];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  luaL_argcheck(L, THGPUTensor_isContiguous(input), 2, "input must be contiguous");
  float *input_data = THGPUTensor_data(input);
  
  THGPUTensor_resize4d(output, nInputPlane, nOutputRows, nOutputCols, batchSize);
  float *output_data = THGPUTensor_data(output);

  spatialMaxPooling_updateOutput<MaxPooler>
    (input, output, 
     nInputPlane, nInputRows, nInputCols, batchSize,
     nOutputRows, nOutputCols, 
     kH, kW,
     0, dW);

  return 1;
}

static int gpunn_SpatialMaxPoolingGPU_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nInputPlane = input->size[0];
  long batchSize = input->size[3];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  THGPUTensor_resizeAs(gradInput, input);
  THGPUTensor_zero(gradInput);

  float *input_data = THGPUTensor_data(input);
  float *output_data = THGPUTensor_data(output);
  float *gradOutput_data = THGPUTensor_data(gradOutput);
  float *gradInput_data = THGPUTensor_data(gradInput);

 spatialMaxPooling_updateGradInput
    (input, gradOutput, output, gradInput,
     nInputPlane, nInputRows, nInputCols, batchSize,
     nOutputRows, nOutputCols, 
     kH, kW,
     0, dW);
    return 1;

}

static const struct luaL_Reg gpunn_SpatialMaxPoolingGPU__ [] = {
  {"SpatialMaxPoolingGPU_updateOutput", gpunn_SpatialMaxPoolingGPU_updateOutput},
  {"SpatialMaxPoolingGPU_updateGradInput", gpunn_SpatialMaxPoolingGPU_updateGradInput},
  {NULL, NULL}
};

static void gpunn_SpatialMaxPoolingGPU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialMaxPoolingGPU__, "nn");
  lua_pop(L,1);
}
