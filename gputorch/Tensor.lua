function torch.GPUTensor.apply(self, func)
   local x = torch.FloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
end

local function Tensor__type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end
local function Tensor__typeAs(self,tensor)
   return self:type(tensor:type())
end
local function Tensor__gpu(self,type)
   return self:type('torch.GPUTensor')
end
local function Tensor__double(self,type)
   return self:type('torch.DoubleTensor')
end
local function Tensor__float(self,type)
   return self:type('torch.FloatTensor')
end

rawset(torch.getmetatable('torch.DoubleTensor'), 'gpu', Tensor__gpu)
rawset(torch.getmetatable('torch.FloatTensor'), 'gpu', Tensor__gpu)
rawset(torch.getmetatable('torch.GPUTensor'), 'gpu', Tensor__gpu)

rawset(torch.getmetatable('torch.GPUTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.GPUTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.GPUTensor'), 'double', Tensor__double)
rawset(torch.getmetatable('torch.GPUTensor'), 'float', Tensor__float)

do
    local metatable = torch.getmetatable('torch.GPUTensor')
    for _,func in pairs{'expand', 'expandAs', 'view', 'viewAs', 'repeatTensor'} do
        rawset(metatable, func, torch[func])
    end
end
