require "torch"
gputorch = require "libgputorch"

torch.GPUStorage.__tostring__ = torch.FloatStorage.__tostring__
torch.GPUTensor.__tostring__ = torch.FloatTensor.__tostring__

include('Tensor.lua')
include('FFI.lua')
include('test.lua')

return gputorch
