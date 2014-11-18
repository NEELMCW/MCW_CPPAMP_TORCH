struct sigmoidupdateOutput_functor
{
  float operator()(const float& input) const
  {
    return 1./(1.+ exp(-input));
  }
};

static int cunn_Sigmoid_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

   std::vector<float> output_data(THCudaTensor_data(output), THCudaTensor_data(output)+THCudaTensor_nElement(output));
  //thrust::device_ptr<float> input_data(THCudaTensor_data(input));
   std::vector<float> input_data(THCudaTensor_data(input), THCudaTensor_data(input)+THCudaTensor_nElement(input));
 // thrust::transform(input_data, input_data+size, output_data, absupdateOutput_functor());
   std::transform(input_data.begin(), input_data.end(), output_data.begin(), sigmoidupdateOutput_functor());

   std::copy(output_data.begin(), output_data.end(), output->storage->data);

  THCudaTensor_free(input);
  return 1;
}

struct sigmoidupdateGradInput_functor
{
  float operator()(const float& output, const float& gradOutput) const
  {
    return gradOutput * (1.-output) * output;
  }
};

static int cunn_Sigmoid_updateGradInput(lua_State *L)
{
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(output);

  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, output);

   std::vector<float> output_data(THCudaTensor_data(output), THCudaTensor_data(output)+THCudaTensor_nElement(output));
  //thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
   std::vector<float> gradOutput_data(THCudaTensor_data(gradOutput), THCudaTensor_data(gradOutput)+THCudaTensor_nElement(gradOutput));
  //thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));
   std::vector<float> gradInput_data(THCudaTensor_data(gradInput), THCudaTensor_data(gradInput)+THCudaTensor_nElement(gradInput));
  //thrust::transform(input_data, input_data+size, gradOutput_data, gradInput_data, absupdateGradInput_functor());
   std::transform(output_data.begin(), output_data.end(), gradOutput_data.begin(),gradInput_data.begin(), sigmoidupdateGradInput_functor());

   
   std::copy(gradInput_data.begin(), gradInput_data.end(), gradInput->storage->data);


  THCudaTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_Sigmoid__ [] = {
  {"Sigmoid_updateOutput", cunn_Sigmoid_updateOutput},
  {"Sigmoid_updateGradInput", cunn_Sigmoid_updateGradInput},
  {NULL, NULL}
};

static void cunn_Sigmoid_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Sigmoid__, "nn");
  lua_pop(L,1);
}
