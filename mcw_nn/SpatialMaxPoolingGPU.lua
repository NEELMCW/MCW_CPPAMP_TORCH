local SpatialMaxPoolingGPU, parent = torch.class('nn.SpatialMaxPoolingGPU', 'nn.Module')

function SpatialMaxPoolingGPU:__init(kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
end

function SpatialMaxPoolingGPU:updateOutput(input)
   input.nn.SpatialMaxPoolingGPU_updateOutput(self, input)
   return self.output
end

function SpatialMaxPoolingGPU:updateGradInput(input, gradOutput)
   input.nn.SpatialMaxPoolingGPU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

