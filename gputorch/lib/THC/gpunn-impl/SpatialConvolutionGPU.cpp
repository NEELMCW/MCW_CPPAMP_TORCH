#ifndef DIVUP
#define DIVUP(x,y) (((x) + (y) - 1) / (y))
#endif

#define MIN(a,b) (a) < (b) ? (a) : (b)

#ifndef assert
#define assert(e)                              \
    if (!(e)) {                                \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting...");                \
    };
#endif

extern  void spatialConv_updateOutput(
  // raw pointers:
  Concurrency::array_view<float,1>&images, Concurrency::array_view<float,1>&filters, Concurrency::array_view<float,1>&targets,
  // input dim:
  int numImgColors, int imgSizeY, int imgSizeX, int numImages,
  // output dim:
  int numFilters, int numModulesY, int numModulesX,
  // filter size:
  int filterSizeY, int filterSizeX,
  // input params:
  int paddingStart, int moduleStride,
  // output params:
  float scaleTargets, float scaleOutput,
  // are filters convolutional or local:
  bool conv);

extern  void spatialConv_updateGradInput(Concurrency::array_view<float, 1>&hidActs,
                                  Concurrency::array_view<float, 1>&filters, Concurrency::array_view<float, 1>&targets,
                                  int numImgColors, int imgSizeY, int imgSizeX, int numImages,int numFilters,int numModulesY,
                                  int numModulesX, int filterSizeY, int filterSizeX, int paddingStart,
                                  int moduleStride, float scaleTargets, float scaleOutput, bool conv);

extern  void spatialConv_accGradParameters(
    // raw pointers:
    Concurrency::array_view<float,1>&images, Concurrency::array_view<float,1>&hidActs, Concurrency::array_view<float,1>&targets,
    // input dim:
    int numImgColors, int imgSizeY, int imgSizeX, int numImages,
    // output dim:
    int numFilters, int numModulesY, int numModulesX, 
    // filter size:
    int filterSizeY, int filterSizeX,
    // input params:
    int paddingStart, int moduleStride,
    // output params:
    float scaleTargets, float scaleOutput,
    int partialSum
);

static int gpunn_SpatialConvolutionGPU_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THGPUTensor *weight = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  luaL_argcheck(L, input->nDimension == 4, 2, "4D (batch mode) tensor is expected");

  long nOutputPlane = weight->size[3];
  long nInputPlane  = weight->size[0];
  long kH           = weight->size[1];
  long kW           = weight->size[2];
  long inputHeight  = input->size[1];
  long inputWidth   = input->size[2];
  long batchSize    = input->size[3];
  long outputHeight = (padding + inputHeight - kH) / dH + 1;
  long outputWidth  = (padding + inputWidth - kW) / dW + 1;

  // resize output
  THGPUTensor_resize4d(output, nOutputPlane, outputHeight, outputWidth, batchSize);

  // asserts
  luaL_argcheck(L, inputWidth == inputHeight, 1, "input must be square");
  luaL_argcheck(L, kH == kW, 1, "kH must be equal to kW");
  luaL_argcheck(L, dH == dW, 1, "dH must be equal to dW");

  // all the data must be contiguous: 
  luaL_argcheck(L, THGPUTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THGPUTensor_isContiguous(weight), 1, "weight must be contiguous");
  luaL_argcheck(L, THGPUTensor_isContiguous(output), 1, "output must be contiguous");

  PREPARE_AV(input, pavInput);
  PREPARE_AV(weight, pavWeight);
  PREPARE_AV(output, pavOutput);
  // convolutions
  spatialConv_updateOutput(*pavInput, *pavWeight, *pavOutput, nInputPlane, inputHeight, inputWidth,
                          batchSize, nOutputPlane, outputHeight, outputWidth, kH, kW,
                          -floor((double)padding/2), dW, 0, 1, true);

  return 1;
}

static int gpunn_SpatialConvolutionGPU_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THGPUTensor *weight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  long nOutputPlane = weight->size[3];
  long nInputPlane  = weight->size[0];
  long kH           = weight->size[1];
  long kW           = weight->size[2];
  long inputHeight  = input->size[1];
  long inputWidth   = input->size[2];
  long batchSize    = input->size[3];
  long outputHeight = (padding + inputHeight - kH) / dH + 1;
  long outputWidth  = (padding + inputWidth - kW) / dW + 1;

  // resize gradInput
  THGPUTensor_resize4d(gradInput, nInputPlane, inputHeight, inputWidth, batchSize);

  // asserts
  luaL_argcheck(L, inputWidth == inputHeight, 1, "input must be square");
  luaL_argcheck(L, kH == kW, 1, "kH must be equal to kW");
  luaL_argcheck(L, dH == dW, 1, "dH must be equal to dW");

  // all the data must be contiguous: 
  luaL_argcheck(L, THGPUTensor_isContiguous(gradInput), 2, "input must be contiguous");
  luaL_argcheck(L, THGPUTensor_isContiguous(weight), 1, "weight must be contiguous");
  luaL_argcheck(L, THGPUTensor_isContiguous(gradOutput), 1, "output must be contiguous");

  PREPARE_AV(gradInput, pavGradInput);
  PREPARE_AV(gradOutput, pavGradOutput);
  PREPARE_AV(weight, pavWeight);
  // convolutions
  spatialConv_updateGradInput(*pavGradOutput, *pavWeight, *pavGradInput, nInputPlane, inputHeight,
                             inputWidth, batchSize, nOutputPlane, outputHeight, outputWidth, kH, kW,
                             -floor((double)padding/2), dW, 0, 1, true);

  return 1;
}

static int gpunn_SpatialConvolutionGPU_accGradParameters(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor *)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor *)luaT_checkudata(L, 3, "torch.GPUTensor");
  THGPUTensor *gradWeight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.GPUTensor");
  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  int partialSum = luaT_getfieldcheckint(L, 1, "partialSum");
  float scale = luaL_optnumber(L, 4, 1);

  long nOutputPlane = gradWeight->size[3];
  long nInputPlane  = gradWeight->size[0];
  long kH           = gradWeight->size[1];
  long kW           = gradWeight->size[2];
  long inputHeight  = input->size[1];
  long inputWidth   = input->size[2];
  long batchSize    = input->size[3];
  long outputHeight = (padding + inputHeight - kH) / dH + 1;
  long outputWidth  = (padding + inputWidth - kW) / dW + 1;

  // asserts
  luaL_argcheck(L, inputWidth == inputHeight, 1, "input must be square");
  luaL_argcheck(L, kH == kW, 1, "kH must be equal to kW");
  luaL_argcheck(L, dH == dW, 1, "dH must be equal to dW");

  if (partialSum)
  {
    // compute partial gradients for outputHeight*outputWidth/partialSum groups of filters separately
    gradWeight = (THGPUTensor *)luaT_getfieldcheckudata(L, 1, "gradWeightPartial", "torch.GPUTensor");
    THGPUTensor_resize4d(gradWeight, outputHeight * outputWidth / partialSum, nInputPlane, kH * kW,
                         nOutputPlane);
    // numModuleY*numModulesX/partialSum, numFilterColors, filterPixels, numFilters
  }

  // all the data must be contiguous: 
  luaL_argcheck(L, THGPUTensor_isContiguous(input), 2, "input must be contiguous");
  luaL_argcheck(L, THGPUTensor_isContiguous(gradWeight), 1, "weight must be contiguous");
  luaL_argcheck(L, THGPUTensor_isContiguous(gradOutput), 1, "output must be contiguous");

  PREPARE_AV(input, pavInput);
  PREPARE_AV(gradOutput, pavGradOutput);
  PREPARE_AV(gradWeight, pavGradWeight);
  // convolutions
  spatialConv_accGradParameters(*pavInput, *pavGradOutput, *pavGradWeight, nInputPlane, inputHeight,
                               inputWidth, batchSize, nOutputPlane, outputHeight, outputWidth, kH, kW,
                               -floor((double)padding/2), dW, 0, scale, partialSum);

  return 0;
}

static const struct luaL_Reg gpunn_SpatialConvolutionGPU__ [] = {
  {"SpatialConvolutionGPU_updateOutput", gpunn_SpatialConvolutionGPU_updateOutput},
  {"SpatialConvolutionGPU_updateGradInput", gpunn_SpatialConvolutionGPU_updateGradInput},
  {"SpatialConvolutionGPU_accGradParameters", gpunn_SpatialConvolutionGPU_accGradParameters},
  {NULL, NULL}
};

static void gpunn_SpatialConvolutionGPU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_SpatialConvolutionGPU__, "nn");
  lua_pop(L,1);
}
