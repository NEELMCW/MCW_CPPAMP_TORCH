if jit then

   local ffi = require 'ffi'

   local cdefs = [[
typedef struct THGPUStorage
{
    float *data;
    long size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
} THGPUStorage;

typedef struct THGPUTensor
{
    long *size;
    long *stride;
    int nDimension;

    THGPUStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

} THGPUTensor;
]]
   ffi.cdef(cdefs)

   local Storage = torch.getmetatable('torch.GPUStorage')
   local Storage_tt = ffi.typeof('THGPUStorage**')

   rawset(Storage, "cdata", function(self) return Storage_tt(self)[0] end)
   rawset(Storage, "data", function(self) return Storage_tt(self)[0].data end)
   -- Tensor
   local Tensor = torch.getmetatable('torch.GPUTensor')
   local Tensor_tt = ffi.typeof('THGPUTensor**')

   rawset(Tensor, "cdata", function(self) return Tensor_tt(self)[0] end)

   rawset(Tensor, "data",
          function(self)
             self = Tensor_tt(self)[0]
             return self.storage ~= nil and self.storage.data + self.storageOffset or nil
          end
   )

end
