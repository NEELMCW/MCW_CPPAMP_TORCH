static int gpunn_TemporalConvolution_updateOutput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
  int outputFrameSize = luaT_getfieldcheckint(L, 1, "outputFrameSize");

  THGPUTensor *weight = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *bias = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.GPUTensor");
  THGPUTensor *output = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.GPUTensor");

  THGPUTensor *outputWindow, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k, i;

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  luaL_argcheck(L, input->nDimension == 2 || input->nDimension == 3, 2, "2D or 3D(batch mode) tensor expected");

  if (input->nDimension == 3)
  {
    dimS = 1;
    dimF = 2;
  }
  luaL_argcheck(L, input->size[dimF] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[dimS] >= kW, 2, "input sequence smaller than kernel size");

  input = THGPUTensor_newContiguous(input);
  outputWindow = THGPUTensor_new();
  inputWindow = THGPUTensor_new();

  nInputFrame = input->size[dimS];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  if (input->nDimension == 2)
  {
    THGPUTensor_resize2d(output, nOutputFrame, outputFrameSize);

    /* bias first */
    for (k = 0; k < nOutputFrame; k++)
    {
      THGPUTensor_select(outputWindow, output, 0, k);
      THGPUTensor_copy(outputWindow, bias);
    }

    /* ouch */
    for (k = 0; nOutputFrame > 0; k++)
    {
      long outputFrameStride = (kW - 1) / dW + 1;
      long inputFrameStride = outputFrameStride * dW;
      long nFrame = (nInputFrame - k * dW - kW) / inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THGPUTensor_setStorage2d(inputWindow, input->storage, input->storageOffset + k * dW * input->size[1],
                               nFrame, inputFrameStride*input->size[1], kW * input->size[1], 1);

      THGPUTensor_setStorage2d(outputWindow, output->storage, output->storageOffset + k * output->size[1],
                               nFrame, outputFrameStride * output->size[1], output->size[1], 1);

      THGPUTensor_transpose(weight, NULL, 0, 1);
      THGPUTensor_addmm(outputWindow, 1, outputWindow, 1, inputWindow, weight);
      THGPUTensor_transpose(weight, NULL, 0, 1);
    }
  }
  else
  {
    THGPUTensor *outputSample = THGPUTensor_new();
    THGPUTensor *inputSample = THGPUTensor_new();
    int nBatchFrame = input->size[0];

    THGPUTensor_resize3d(output, nBatchFrame, nOutputFrame, outputFrameSize);

    for (i = 0; i < nBatchFrame; i++)
    {
      THGPUTensor_select(outputSample, output, 0, i);
      THGPUTensor_select(inputSample, input, 0, i);
      long nOutputSampleFrame = nOutputFrame;

      /* bias first */
      for (k = 0; k < nOutputFrame; k++)
      {
        THGPUTensor_select(outputWindow, outputSample, 0, k);
        THGPUTensor_copy(outputWindow, bias);
      }

      /* ouch */
      for (k = 0; nOutputSampleFrame > 0; k++)
      {
        long outputFrameStride = (kW - 1) / dW + 1;
        long inputFrameStride = outputFrameStride * dW;
        long nFrame = (nInputFrame - k * dW - kW) / inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THGPUTensor_setStorage2d(inputWindow, inputSample->storage,
                                 inputSample->storageOffset + k * dW * inputSample->size[1],
                                 nFrame, inputFrameStride*inputSample->size[1],
                                 kW * inputSample->size[1], 1);

        THGPUTensor_setStorage2d(outputWindow, outputSample->storage,
                                 outputSample->storageOffset + k * outputSample->size[1], nFrame,
                                 outputFrameStride * outputSample->size[1],
                                 outputSample->size[1], 1);

        THGPUTensor_transpose(weight, NULL, 0, 1);
        THGPUTensor_addmm(outputWindow, 1, outputWindow, 1, inputWindow, weight);
        THGPUTensor_transpose(weight, NULL, 0, 1);
      }
    }
    THGPUTensor_free(outputSample);
    THGPUTensor_free(inputSample);
  }

  THGPUTensor_free(outputWindow);
  THGPUTensor_free(inputWindow);
  THGPUTensor_free(input);

  return 1;
}

static int gpunn_TemporalConvolution_updateGradInput(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame;
  long nOutputFrame;

  THGPUTensor *weight = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.GPUTensor");
  THGPUTensor *gradInput = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.GPUTensor");

  THGPUTensor *gradOutputWindow;
  THGPUTensor *gradInputWindow;
  long k, i;

  int dimS = 0; // sequence dimension

  if (gradOutput->nDimension == 3)
  {
    dimS = 1;
  }

  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];

  /* Not necessary with partial backprop: */
  gradOutputWindow = THGPUTensor_new();
  gradInputWindow = THGPUTensor_new();

  THGPUTensor_resizeAs(gradInput, input);
  THGPUTensor_zero(gradInput);

  if (gradOutput->nDimension == 2)
  {
    /* ouch */
    for (k = 0; nOutputFrame > 0; k++)
    {
      long outputFrameStride = (kW - 1) / dW + 1;
      long inputFrameStride = outputFrameStride * dW;
      long nFrame = (nInputFrame - k * dW - kW) / inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THGPUTensor_setStorage2d(gradOutputWindow, gradOutput->storage,
                               gradOutput->storageOffset + k*gradOutput->size[1], nFrame,
                               outputFrameStride * gradOutput->size[1], gradOutput->size[1], 1);

      THGPUTensor_setStorage2d(gradInputWindow, gradInput->storage,
                               gradInput->storageOffset + k * dW * gradInput->size[1],
                               nFrame, inputFrameStride * gradInput->size[1],
                               kW * gradInput->size[1], 1);

      THGPUTensor_addmm(gradInputWindow, 1, gradInputWindow, 1, gradOutputWindow, weight);
    }
  }
  else
  {
    THGPUTensor *gradOutputSample = THGPUTensor_new();
    THGPUTensor *gradInputSample = THGPUTensor_new();
    long nBatchFrame = input->size[0];
    for (i = 0; i < nBatchFrame; i++)
    {
      THGPUTensor_select(gradOutputSample, gradOutput, 0, i);
      THGPUTensor_select(gradInputSample, gradInput, 0, i);
      long nOutputSampleFrame = nOutputFrame;

      /* ouch */
      for (k = 0; nOutputSampleFrame > 0; k++)
      {
        long outputFrameStride = (kW - 1) / dW + 1;
        long inputFrameStride = outputFrameStride * dW;
        long nFrame = (nInputFrame - k * dW - kW) / inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THGPUTensor_setStorage2d(gradOutputWindow, gradOutputSample->storage,
                                 gradOutputSample->storageOffset + k * gradOutputSample->size[1],
                                 nFrame, outputFrameStride * gradOutputSample->size[1],
                                 gradOutputSample->size[1], 1);

        THGPUTensor_setStorage2d(gradInputWindow, gradInputSample->storage,
                                 gradInputSample->storageOffset + k * dW * gradInputSample->size[1],
                                 nFrame, inputFrameStride * gradInputSample->size[1],
                                 kW * gradInputSample->size[1], 1);

        THGPUTensor_addmm(gradInputWindow, 1, gradInputWindow, 1, gradOutputWindow, weight);
      }
    }
    THGPUTensor_free(gradOutputSample);
    THGPUTensor_free(gradInputSample);
  }

  THGPUTensor_free(gradOutputWindow);
  THGPUTensor_free(gradInputWindow);

  return 1;
}

static int gpunn_TemporalConvolution_accGradParameters(lua_State *L)
{
  THGPUTensor *input = (THGPUTensor*)luaT_checkudata(L, 2, "torch.GPUTensor");
  THGPUTensor *gradOutput = (THGPUTensor*)luaT_checkudata(L, 3, "torch.GPUTensor");
  float scale = luaL_optnumber(L, 4, 1);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame;
  long nOutputFrame;

  THGPUTensor *gradWeight = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.GPUTensor");
  THGPUTensor *gradBias = (THGPUTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.GPUTensor");

  THGPUTensor *gradOutputWindow;
  THGPUTensor *inputWindow;
  long k, i;

  int dimS = 0; // sequence dimension

  if (gradOutput->nDimension == 3)
  {
    dimS = 1;
  }

  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];

  /* Not necessary with partial backprop: */
  input = THGPUTensor_newContiguous(input);
  gradOutputWindow = THGPUTensor_new();
  inputWindow = THGPUTensor_new();

  if (input->nDimension == 2)
  {
    /* bias first */
    for (k = 0; k < nOutputFrame; k++)
    {
      THGPUTensor_select(gradOutputWindow, gradOutput, 0, k);
      THGPUTensor_cadd(gradBias, gradBias, scale, gradOutputWindow);
    }

    /* ouch */
    for (k = 0; nOutputFrame > 0; k++)
    {
      long outputFrameStride = (kW - 1) / dW + 1;
      long inputFrameStride = outputFrameStride * dW;
      long nFrame = (nInputFrame - k * dW - kW) / inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THGPUTensor_setStorage2d(inputWindow, input->storage,
                               input->storageOffset + k * dW * input->size[1],
                               nFrame, inputFrameStride * input->size[1],
                               kW * input->size[1], 1);

      THGPUTensor_setStorage2d(gradOutputWindow, gradOutput->storage,
                               gradOutput->storageOffset + k * gradOutput->size[1],
                               nFrame, outputFrameStride * gradOutput->size[1],
                               gradOutput->size[1], 1);

      THGPUTensor_transpose(gradOutputWindow, NULL, 0, 1);
      THGPUTensor_addmm(gradWeight, 1, gradWeight, scale, gradOutputWindow, inputWindow);
      THGPUTensor_transpose(gradOutputWindow, NULL, 0, 1);
    }
  }
  else
  {
    THGPUTensor *gradOutputSample = THGPUTensor_new();
    THGPUTensor *inputSample = THGPUTensor_new();
    long nBatchFrame = input->size[0];

    for (i = 0; i < nBatchFrame; i++)
    {
      THGPUTensor_select(gradOutputSample, gradOutput, 0, i);
      THGPUTensor_select(inputSample, input, 0, i);
      long nOutputSampleFrame = nOutputFrame;

      /* bias first */
      for (k = 0; k < nOutputFrame; k++)
      {
        THGPUTensor_select(gradOutputWindow, gradOutputSample, 0, k);
        THGPUTensor_cadd(gradBias, gradBias, scale, gradOutputWindow);
      }

      /* ouch */
      for (k = 0; nOutputSampleFrame > 0; k++)
      {
        long outputFrameStride = (kW - 1) / dW + 1;
        long inputFrameStride = outputFrameStride * dW;
        long nFrame = (nInputFrame - k * dW - kW) / inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THGPUTensor_setStorage2d(inputWindow, inputSample->storage,
                                 inputSample->storageOffset + k * dW * inputSample->size[1],
                                 nFrame, inputFrameStride * inputSample->size[1],
                                 kW * inputSample->size[1], 1);

        THGPUTensor_setStorage2d(gradOutputWindow, gradOutputSample->storage,
                                 gradOutputSample->storageOffset + k * gradOutputSample->size[1],
                                 nFrame, outputFrameStride * gradOutputSample->size[1],
                                 gradOutputSample->size[1], 1);

        THGPUTensor_transpose(gradOutputWindow, NULL, 0, 1);
        THGPUTensor_addmm(gradWeight, 1, gradWeight, scale, gradOutputWindow, inputWindow);
        THGPUTensor_transpose(gradOutputWindow, NULL, 0, 1);
      }
    }
    THGPUTensor_free(gradOutputSample);
    THGPUTensor_free(inputSample);
  }

  THGPUTensor_free(gradOutputWindow);
  THGPUTensor_free(inputWindow);
  THGPUTensor_free(input);

  return 1;
}

static const struct luaL_Reg gpunn_TemporalConvolution__ [] = {
  {"TemporalConvolution_updateOutput", gpunn_TemporalConvolution_updateOutput},
  {"TemporalConvolution_updateGradInput", gpunn_TemporalConvolution_updateGradInput},
  {"TemporalConvolution_accGradParameters", gpunn_TemporalConvolution_accGradParameters},
  {NULL, NULL}
};

static void gpunn_TemporalConvolution_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.GPUTensor");
  luaT_registeratname(L, gpunn_TemporalConvolution__, "nn");
  lua_pop(L,1);
}
