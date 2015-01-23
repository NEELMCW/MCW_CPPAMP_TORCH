
local runtests = false
if not gputorch then
   require 'gputorch'
   runtests = true
end
local tester
local test = {}
local msize = 100
local minsize = 100
local maxsize = 600
local minvalue = 2
local maxvalue = 20
local nloop = 100
local times = {}

--e.g. unit test cmd: th -lgputorch -e "gputorch.test{'view','viewAs'}"

local function isEqual(a, b, tolerance, ...)
   if a == nil and b == nil then return true end
   if a == nil and b ~= nil then return false end
   if a ~= nil and b == nil then return false end
   if torch.type(b) ~= torch.type(a) then
      --[[
        TODO: memory leak for original b is lost and won't be released (At least 1 GPU memory allocation)
             type(a) is torch.FloatTensor , type(b) is torch.GPUTensor
      ]]--
      b = b:typeAs(a) -- TODO: remove the need for this (a-b doesnt work for bytetensor, gputensor pairs)
   end
   local diff = a-b
   --print(diff)
   --tolerance = tolerance or 0.000001
   tolerance = 0.1
   if type(a) == 'number' then
      return math.abs(diff) < tolerance
   else
      if torch.type(diff) ~= 'torch.FloatTensor' then
         diff = diff:float() -- TODO: remove the need for this (byteTensor and abs)
      end
      return diff:abs():max() < tolerance
   end
end

local function compareFloatAndGPU(x, fn, ...)
   local x_cpu    = x:float()
   local x_gpu   = x_cpu:gpu()
   local res1_cpu, res2_cpu, res3_cpu, res4_cpu
   local res1_gpu, res2_gpu, res3_gpu, res4_gpu
   if type(fn) == 'string' then
      tester:assertne(x_gpu[fn], nil,
         string.format("Missing function GPUTensor.%s", fn))
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = x_cpu[fn](x_cpu, ...)
      res1_gpu, res2_gpu, res3_gpu, res4_gpu = x_gpu[fn](x_gpu, ...)
   elseif type(fn) == 'function' then
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = fn(x_cpu, ...)
      res1_gpu, res2_gpu, res3_gpu, res4_gpu = fn(x_gpu, ...)
   else
      error("Incorrect function type")
   end
   local tolerance = 1e-5
   tester:assert(isEqual(res1_cpu, res1_gpu, tolerance),
      string.format("Divergent results between CPU and CUDA for function '%s'", tostring(fn)))
   tester:assert(isEqual(res2_cpu, res2_gpu, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", tostring(fn)))
   tester:assert(isEqual(res3_cpu, res3_gpu, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", tostring(fn)))
   tester:assert(isEqual(res4_cpu, res4_gpu, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", tostring(fn)))
end

local function compareFloatAndGPUTensorArgs(x, fn, ...)
   local x_cpu = x:float()
   local x_gpu = x_cpu:gpu()
   local res1_cpu, res2_cpu, res3_cpu, res4_cpu
   local res1_gpu, res2_gpu, res3_gpu, res4_gpu
   -- Transformation of args
   local tranform_args = function(t, type)
      for k,v in pairs(t) do
         local v_type = torch.Tensor.type(v)
         if v_type == 'torch.FloatTensor' or v_type == 'torch.GPUTensor' or v_type == 'torch.DoubleTensor' then
            t[k] = v:type(type)
         end
      end
      return t
   end
   local cpu_args = tranform_args({...}, 'torch.FloatTensor')
   local gpu_args = tranform_args({...}, 'torch.GPUTensor')
   if type(fn) == 'string' then
      tester:assertne(x_gpu[fn], nil,
         string.format("Missing function GPUTensor.%s", fn))
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = x_cpu[fn](x_cpu, unpack(cpu_args))
      res1_gpu, res2_gpu, res3_gpu, res4_gpu = x_gpu[fn](x_gpu, unpack(gpu_args))
   elseif type(fn) == 'function' then
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = fn(x_cpu, unpack(cpu_args))
      res1_gpu, res2_gpu, res3_gpu, res4_gpu = fn(x_gpu, unpack(gpu_args))
   else
      error("Incorrect function type")
   end
   local tolerance = 1e-5
   tester:assert(isEqual(res1_cpu, res1_gpu, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
   tester:assert(isEqual(res2_cpu, res2_gpu, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
   tester:assert(isEqual(res3_cpu, res3_gpu, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
   tester:assert(isEqual(res4_cpu, res4_gpu, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
end

function test.squeeze()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1, sz, 1)
   compareFloatAndGPU(x, 'squeeze')

   local y = x:gpu():squeeze()
   tester:assert(y:dim() == 2, "squeeze err")

   x = torch.FloatTensor():rand(sz, 1, 1, sz)
   compareFloatAndGPU(x, 'squeeze', 2)

   local y = x:gpu():squeeze(2)
   tester:assert(y:dim() == 3, "squeeze1d err")
end

function test.expand()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1)
   compareFloatAndGPU(x, 'expand', sz, sz)

   x = torch.FloatTensor():rand(1, sz)
   compareFloatAndGPU(x, 'expand', sz, sz)
end

function test.view()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 3)
   compareFloatAndGPU(x, 'view', sz, 3, 1)
end

function test.viewAs()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 3)
   local y = torch.FloatTensor():rand(sz, 3, 1)
   compareFloatAndGPUTensorArgs(x, 'viewAs', y)
end

function test.repeatTensor()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 3)
   compareFloatAndGPU(x, 'repeatTensor', sz, 2)
end

function test.copyNoncontiguous()
     local x = torch.FloatTensor():rand(1, 1)
     local f = function(src)
        return src.new(2, 2):copy(src:expand(2, 2))
     end
     compareFloatAndGPU(x, f)

   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1)
   local f = function(src)
      return src.new(sz, sz):copy(src:expand(sz, sz))
   end
   compareFloatAndGPU(x, f)

   x = torch.FloatTensor():rand(sz, sz, 2)
   local f = function(src)
      return src.new(sz, sz):copy(src[{{},{},{2}}])
   end
   compareFloatAndGPU(x, f)

   x = torch.FloatTensor():rand(2, sz, sz)
   local f = function(src)
      return src.new(sz, sz):copy(src[{{2},{},{}}])
   end
   compareFloatAndGPU(x, f)

   x = torch.FloatTensor():rand(sz, 2, sz)
   local f = function(src)
      return src.new(sz, sz):copy(src[{{},{2},{}}])
   end
   compareFloatAndGPU(x, f)

   x = torch.FloatTensor():rand(sz, 2, sz)
   local f = function(src)
      return src.new(sz, 1, sz):copy(src[{{},{2},{}}])
   end
   compareFloatAndGPU(x, f)

   x = torch.FloatTensor():rand(sz, sz):transpose(1,2)
   local f = function(src)
      return src.new(sz, sz):copy(src)
   end
   compareFloatAndGPU(x, f)
end

function test.largeNoncontiguous()
   local x = torch.FloatTensor():randn(20, 1, 60, 60)
   local sz = math.floor(torch.uniform(maxsize, maxsize))
   local f = function(src)
      return src.new(20, sz, 60, 60):copy(src:expand(20, sz, 60, 60))
   end
   compareFloatAndGPU(x, f)
end

function test.zero()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'zero')
end

function test.fill()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   compareFloatAndGPUTensorArgs(x, 'fill', v)
end

function test.reshape()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))*2
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'reshape', sz1/2, sz2*2)
end

--[[function test.zeros()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local t = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.GPUTensor')
   local x = torch.zeros(sz1, sz2)
   assert(x:sum() == 0)
   torch.setdefaulttensortype(t)
end

function test.ones()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local t = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.GPUTensor')
   local x = torch.ones(sz1, sz2)
   assert(x:sum() == x:nElement())
   torch.setdefaulttensortype(t)
end]]--


function test.add()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   compareFloatAndGPUTensorArgs(x, 'add', z)
   compareFloatAndGPUTensorArgs(x, 'add', z, v)
   compareFloatAndGPUTensorArgs(x, 'add', y, z)
   compareFloatAndGPUTensorArgs(x, 'add', y, v, z)
end

function test.cmul()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'cmul', y)
end

function test.cdiv()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'cdiv', y)
end

function test.cdiv3()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor(sz1, sz2)
   compareFloatAndGPUTensorArgs(z, 'cdiv', x, y)
end

function test.addcmul()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'addcmul', y, z)
   compareFloatAndGPUTensorArgs(x, 'addcmul', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   compareFloatAndGPUTensorArgs(r, 'addcmul', x, y, z)
   compareFloatAndGPUTensorArgs(r, 'addcmul', x, torch.uniform(), y, z)
end

function test.addcdiv()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'addcdiv', y, z)
   compareFloatAndGPUTensorArgs(x, 'addcdiv', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   compareFloatAndGPUTensorArgs(r, 'addcdiv', x, y, z)
   compareFloatAndGPUTensorArgs(r, 'addcdiv', x, torch.uniform(), y, z)
end

function test.logicalValue()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'gt', y, 0.3)
   compareFloatAndGPU(x, 'gt', 0.3)
end

function test.logicalTensor()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPUTensorArgs(x, 'gt', y, z)
end

function test.mean()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPU(x, 'mean')
   compareFloatAndGPU(x, 'mean', 1)
   --compareFloatAndGPU(x, 'mean', 2)
end

function test.max()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPU(x, 'max')
   --compareFloatAndGPU(x, 'max', 1)
   --compareFloatAndGPU(x, 'max', 2)
end

function test.min()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPU(x, 'min')
   --compareFloatAndGPU(x, 'min', 1)
   --compareFloatAndGPU(x, 'min', 2)
end

function test.sum()
   local minsize = 10
   local maxsize = 20
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPU(x, 'sum')
   compareFloatAndGPU(x, 'sum', 1)
   --compareFloatAndGPU(x, 'sum', 2)
end

function test.prod()
   local minsize = 10
   local maxsize = 20
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   --compareFloatAndGPU(x, 'prod')
   compareFloatAndGPU(x, 'prod', 1)
   compareFloatAndGPU(x, 'prod', 2)
end

function test.round()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPU(x, 'round')
end

function test.var()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPU(x, 'var')
   -- multi-dim var is not implemented
   -- compareFloatAndGPU(x, 'var', 1, true)
   -- compareFloatAndGPU(x, 'var', 1, false)
   -- compareFloatAndGPU(x, 'var', 2, true)
   -- compareFloatAndGPU(x, 'var', 2, false)
end

function test.std()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndGPU(x, 'std')
   -- multi-dim std is not implemented
   -- compareFloatAndGPU(x, 'std', 1, true)
   -- compareFloatAndGPU(x, 'std', 1, false)
   -- compareFloatAndGPU(x, 'std', 2, true)
   -- compareFloatAndGPU(x, 'std', 2, false)
end

-- Test element-wise unary operators with both one and two arguments.
local function testUnary1(fn)
   local function test()
      local sz1 = math.floor(torch.uniform(minsize,maxsize))
      local sz2 = math.floor(torch.uniform(minsize,maxsize))
      local x = torch.FloatTensor():rand(sz1, sz2)
      compareFloatAndGPUTensorArgs(x, fn)
   end
   return test
end

local function testUnary2(fn)
   local function test()
      local sz1 = math.floor(torch.uniform(minsize,maxsize))
      local sz2 = math.floor(torch.uniform(minsize,maxsize))
      local x = torch.FloatTensor():rand(sz1, sz2)
      local y = torch.FloatTensor()
      compareFloatAndGPUTensorArgs(y, fn, x)
   end
   return test
end

for _,name in ipairs({"log", "log1p", "exp",
                      "cos", "acos", "cosh",
                      "sin", "asin", "sinh",
                      "tan", "atan", "tanh",
                      "sqrt",
                      "ceil", "floor",
                      "abs", "sign"}) do

   test[name .. "1"] = testUnary1(name)
   test[name .. "2"] = testUnary2(name)

end

function test.atan2(fn)
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor()
   compareFloatAndGPUTensorArgs(z, 'atan2', x, y)
end

function test.pow1()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local pow = torch.uniform(minvalue,maxvalue)
   compareFloatAndGPUTensorArgs(x, 'pow', pow)
end

function test.pow2()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor()
   local pow = torch.uniform(minvalue,maxvalue)
   compareFloatAndGPUTensorArgs(y, 'pow', x, pow)
end

function test.clamp1()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5):add(-2.5)
   local min_val = -1
   local max_val = 1
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   compareFloatAndGPUTensorArgs(x, 'clamp', min_val, max_val)
end

function test.clamp2()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5):add(-2.5)
   local min_val = -1
   local max_val = 1
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   local y = torch.FloatTensor():resizeAs(x)
   compareFloatAndGPUTensorArgs(y, 'clamp', x, min_val, max_val)
end


function test.indexCopy()
   local sz1 = math.floor(torch.uniform(minsize,maxsize)) -- dim1
   local sz2 = math.floor(torch.uniform(minsize,maxsize)) -- dim2
   local x = torch.FloatTensor():rand(sz1, sz2) -- input


   -- Case 1: 2D tensor, indexCopy over first dimension, 2 indices
   -- choose two indices from the first dimension, i.e. [1,sz1]
   local longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
   local index = 1
   local src = torch.Tensor(2, sz2):uniform()
   compareFloatAndGPUTensorArgs(x, 'indexCopy', index, longIndex, src)

   -- Case 2: 2D tensor, indexCopy over second dimension, 2 indices
   index = 2
   longIndex =  torch.LongTensor{math.floor(torch.uniform(1, sz2)), math.floor(torch.uniform(1, sz2))}
   src = torch.Tensor(sz1, 2):uniform():gpu()
   compareFloatAndGPUTensorArgs(x, 'indexCopy', index, longIndex, src)

   -- Case 3: 1D tensor, indexCopy over 1st dimension, 2 indices
   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
   src = torch.Tensor(2):uniform()
   compareFloatAndGPUTensorArgs(x, 'indexCopy', index, longIndex, src)
end

function test.indexFill()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)

   local longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
   local index = 1
   local val = torch.randn(1)[1]
   compareFloatAndGPU(x, 'indexFill', index, longIndex, val)

   index = 2
   longIndex =  torch.LongTensor{math.floor(torch.uniform(1, sz2)), math.floor(torch.uniform(1, sz2))}
   val = torch.randn(1)[1]
   compareFloatAndGPU(x, 'indexFill', index, longIndex, val)

   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
   val = torch.randn(1)[1]
   compareFloatAndGPU(x, 'indexFill', index, longIndex, val)
end

function test.renorm()
   local x = torch.randn(10,5):float()
   local maxnorm = x:norm(2,1):mean()

   compareFloatAndGPU(x, 'renorm', 2, 2, maxnorm)

   x = torch.randn(3,4,5)
   compareFloatAndGPU(x, 'renorm', 2, 2, maxnorm)

   x = torch.randn(3,4,5)
   compareFloatAndGPU(x, 'renorm', 3, 2, maxnorm)

   x = torch.randn(3,4,5,100)
   compareFloatAndGPU(x, 'renorm', 3, 2, maxnorm)

   x = torch.randn(3,4,5,100)
   compareFloatAndGPU(x, 'renorm', 4, 2, maxnorm)
end

function test.indexSelect()
   --  test for speed
   local n_row = math.random(minsize,maxsize)
   local n_col = math.random(minsize,maxsize)
   local n_idx = math.random(n_col)

   local x = torch.randn(n_row, n_col):float()
   local indices = torch.randperm(n_idx):long()
   local z = torch.FloatTensor()

   local tm = {}
   local title = string.format('indexSelect ')
   times[title] = tm

   z:index(x, 2, indices)
   local groundtruth = z:clone()
   local clock = torch.Timer()
   for i=1,nloop do
      z:index(x, 2, indices)
   end
   tm.cpu = clock:time().real

   x = x:gpu()
   z = torch.GPUTensor()

   z:index(x, 2, indices)
   local resgpu = z:clone():float()
   clock:reset()
   for i=1,nloop do
      z:index(x, 2, indices)
   end
   tm.gpu = clock:time().real
   tester:assertTensorEq(groundtruth, resgpu, 0.00001, "Error in indexSelect")
end

function test.addmv()
   local sizes = {
      {2,1},
      {1,2},
      {1,1},
      {3,4},
      {3,3},
      {15,18},
      {19,15}
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n)
      local a = torch.randn(n, m)
      local b = torch.randn(m)
      compareFloatAndGPUTensorArgs(c, 'addmv', torch.normal(), torch.normal(), a, b)
   end
end

--[[function test.mv()
   local sizes = {
      {2,1},
      {1,2},
      {1,1},
      {3,4},
      {3,3},
      {15,18},
      {19,15}
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n)
      local a = torch.randn(n, m)
      local b = torch.randn(m)
      compareFloatAndGPUTensorArgs(c, 'mv', a, b)
   end
end]]--

function test.addr()
   local sizes = {
      {2,1},
      {1,2},
      {1,1},
      {3,4},
      {3,3},
      {15,18},
      {19,15}
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n,m)
      local a = torch.randn(n)
      local b = torch.randn(m)
      compareFloatAndGPUTensorArgs(c, 'addr', torch.normal(), a, b)
   end
end

function test.addmm()
   local sizes = {
      {16, 3, 1},
      {1, 12, 1},
      {24, 23, 22},
      {1, 1, 1},
      {1, 1, 7},
      {12, 1, 12},
      {10, 10, 10},
   }
   for _, size in pairs(sizes) do
      local n, k, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n, k)
      local b = torch.randn(k, m)
      compareFloatAndGPUTensorArgs(c, 'addmm', torch.normal(), torch.normal(), a, b)
   end
end

--[[function test.mm()
   local sizes = {
      {16, 3, 1},
      {1, 12, 1},
      {24, 23, 22},
      {1, 1, 1},
      {1, 1, 7},
      {12, 1, 12},
      {10, 10, 10},
   }
   for _, size in pairs(sizes) do
      local n, k, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n, k)
      local b = torch.randn(k, m)
      compareFloatAndGPUTensorArgs(c, 'mm', a, b)
   end
end]]--

--[[function test.ger()
   local sizes = {
      {16, 1},
      {1, 12},
      {24, 23},
      {1, 1},
      {33, 7},
      {12, 14},
      {10, 10},
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n)
      local b = torch.randn(m)
      compareFloatAndGPUTensorArgs(c, 'ger', a, b)
   end
end]]--

function test.isSameSizeAs()
   local t1 = torch.GPUTensor(3, 4, 9, 10)
   local t2 = torch.GPUTensor(3, 4)
   local t3 = torch.GPUTensor(1, 9, 3, 3)
   local t4 = torch.GPUTensor(3, 4, 9, 10)

   tester:assert(t1:isSameSizeAs(t2) == false, "wrong answer ")
   tester:assert(t1:isSameSizeAs(t3) == false, "wrong answer ")
   tester:assert(t1:isSameSizeAs(t4) == true, "wrong answer ")
end

-- Test random number generation.
local function checkIfUniformlyDistributed(t, min, max)
   tester:assertge(t:min(), min - 1e-6, "values are too low")
   tester:assertle(t:max(), max + 1e-6, "values are too high")
   tester:assertalmosteq(t:mean(), (min + max) / 2, 0.1, "mean is wrong")
end

function test.uniform()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local min = torch.uniform()
   local max = min + torch.uniform()
   local t = torch.GPUTensor(sz1, sz2)

   t:uniform(min, max)
   checkIfUniformlyDistributed(t, min, max, tolerance)
end

function test.bernoulli()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local p = torch.uniform()
   local t = torch.GPUTensor(sz1, sz2)
   t:bernoulli(p)
   tester:assertalmosteq(t:mean(), p, 0.1, "mean is not equal to p")
   local f = t:float()
   tester:assertTensorEq(f:eq(1):add(f:eq(0)):float(),
                         torch.FloatTensor(sz1, sz2):fill(1),
                         1e-6,
                         "each value must be either 0 or 1")
end

function test.normal()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.1
   local t = torch.GPUTensor(sz1, sz2)

   t:normal(mean, std)
   tester:assertalmosteq(t:mean(), mean, tolerance, "mean is wrong")
   tester:assertalmosteq(t:std(), std, tolerance, "standard deviation is wrong")
end

function test.logNormal()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.1
   local t = torch.GPUTensor(sz1, sz2)

   t:logNormal(mean, std)
   local logt = t:log()
   tester:assertalmosteq(logt:mean(), mean, tolerance, "mean is wrong")
   tester:assertalmosteq(logt:std(), std, tolerance, "standard deviation is wrong")
end

function test.geometric()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local p = torch.uniform()
   local t = torch.GPUTensor(sz1, sz2)

   t:geometric(p)
   local u = torch.FloatTensor(sz1, sz2):fill(1) -
                 ((t:float() - 1) * math.log(p)):exp()
   checkIfUniformlyDistributed(u, 0, 1)
end

function test.exponential()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local lambda = torch.uniform()
   local t = torch.GPUTensor(sz1, sz2)

   t:exponential(lambda)
   local u = torch.FloatTensor(sz1, sz2):fill(1) -
                 (t:float() * -lambda):exp()
   checkIfUniformlyDistributed(u, 0, 1)
end

function test.cauchy()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local median, sigma = torch.uniform(), torch.uniform()
   local t = torch.GPUTensor(sz1, sz2)

   t:cauchy(median, sigma)
   local u = ((t:float() - median) / sigma):atan() / math.pi + 0.5
   checkIfUniformlyDistributed(u, 0, 1)
end

--[[function test.random_seed()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.01
   local t = torch.GPUTensor(sz1, sz2)
   local u = torch.GPUTensor(sz1, sz2)

   local seed = gputorch.seed()
   t:normal(mean, std)
   gputorch.manualSeed(seed)
   u:normal(mean, std)
   tester:assertTensorEq(t:float(), u:float(), 1e-6, "values not equal after resetting the seed")
end]]--

--[[function test.restore_rng()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.01
   local t = torch.GPUTensor(sz1, sz2)
   local u = torch.GPUTensor(sz1, sz2)

   local seed = gputorch.seed()
   local rng = gputorch.getRNGState()
   t:normal(mean, std)
   -- Change the seed so we can check that restoring the RNG state also restores the seed.
   gputorch.manualSeed(seed + 123)
   gputorch.setRNGState(rng)
   u:normal(mean, std)
   tester:assertTensorEq(t:float(), u:float(), 1e-6, "values not equal after restoring the RNG state")
   tester:asserteq(gputorch.initialSeed(), seed, "seed was not restored")
end]]--



function gputorch.test(tests)
   math.randomseed(os.time())
   torch.manualSeed(os.time())
   tester = torch.Tester()
   tester:add(test)
   tester:run(tests)
   print ''
   for module,tm in pairs(times) do
      print(module .. ': \t average speedup is ' .. (tm.cpu / (tm.gpu or 1e6)))
   end
end

if runtests then
   gputorch.test()
end

