require "libgputorch"
require "nn"
gpunn = require "libgpunn"

torch.include('gpunn', 'test.lua')
return gpunn
