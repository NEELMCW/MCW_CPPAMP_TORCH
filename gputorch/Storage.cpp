#include "torch/utils.h"
#include "THC.h"
#include "THFile.h"
#include "luaT.h"
#include "copyHelpers.h"

/* everything is as the generic Storage.c, except few things (see below) */

#define real float
#define Real GPU
#define TH_GENERIC_FILE "generic/Storage.c"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
// Note that 'data' is on host and so we use the device_data to get the corresponding device mapping 
#define THFile_readRealRaw(file, data, size)                                                      \
  {                                                                                               \
    float *fdata = (float*)THAlloc(sizeof(float) * size);                                         \
    THFile_readFloatRaw(file, fdata, size);                                                       \
    float* device_ptr = static_cast<float *>(Concurrency::getAllocator().device_data(data));      \
    THGPUCheck(gpuMemcpy(device_ptr, 0, fdata, 0, size * sizeof(float), gpuMemcpyHostToDevice));  \
    THFree(fdata);                                                                                \
  }

#define THFile_writeRealRaw(file, data, size)                                                      \
  {                                                                                                \
    float *fdata = (float *)THAlloc(sizeof(float) * size);                                         \
    float* device_ptr = static_cast<float *>(Concurrency::getAllocator().device_data(data));       \
    THGPUCheck(gpuMemcpy(fdata, 0, device_ptr, 0, size * sizeof(float), gpuMemcpyDeviceToHost));   \
    THFile_writeFloatRaw(file, fdata, size);                                                       \
    THFree(fdata);                                                                                 \
  }

#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"

#undef real
#undef Real
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to GPUStorage */

static int gputorch_GPUStorage_copy(lua_State *L)
{
  THGPUStorage *storage = (THGPUStorage *)luaT_checkudata(L, 1, "torch.GPUStorage");
  void *src;
  if ( (src = (THGPUStorage *)luaT_toudata(L, 2, "torch.GPUStorage")) )
    THGPUStorage_copy(storage, (THGPUStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THGPUStorage_copyByte(storage, (THByteStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THGPUStorage_copyChar(storage, (THCharStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THGPUStorage_copyShort(storage, (THShortStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THGPUStorage_copyInt(storage, (THIntStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THGPUStorage_copyLong(storage, (THLongStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THGPUStorage_copyFloat(storage, (THFloatStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THGPUStorage_copyDouble(storage, (THDoubleStorage *)src);
  else if ( (src = luaT_toudata(L, 2, "torch.GPUStorage")) )
    THGPUStorage_copyGPU(storage, (THGPUStorage *)src);
  else
    luaL_typerror(L, 2, "torch.*Storage");

  lua_settop(L, 1);
  return 1;
}

#define GPU_IMPLEMENT_STORAGE_COPY(TYPEC)                                                                 \
  static int gputorch_##TYPEC##Storage_copy(lua_State *L)                                                 \
  {                                                                                                       \
    TH##TYPEC##Storage *storage = (TH##TYPEC##Storage *)luaT_checkudata(L, 1, "torch." #TYPEC "Storage"); \
    void *src;                                                                                            \
    if ( (src = luaT_toudata(L, 2, "torch." #TYPEC "Storage")) )                                          \
      TH##TYPEC##Storage_copy(storage, (TH##TYPEC##Storage *)src);                                        \
    else if ( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )                                           \
      TH##TYPEC##Storage_copyByte(storage, (THByteStorage *)src);                                         \
    else if ( (src = luaT_toudata(L, 2, "torch.CharStorage")) )                                           \
      TH##TYPEC##Storage_copyChar(storage, (THCharStorage *)src);                                         \
    else if ( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )                                          \
      TH##TYPEC##Storage_copyShort(storage, (THShortStorage *)src);                                       \
    else if ( (src = luaT_toudata(L, 2, "torch.IntStorage")) )                                            \
      TH##TYPEC##Storage_copyInt(storage, (THIntStorage *)src);                                           \
    else if ( (src = luaT_toudata(L, 2, "torch.LongStorage")) )                                           \
      TH##TYPEC##Storage_copyLong(storage, (THLongStorage *)src);                                         \
    else if ( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )                                          \
      TH##TYPEC##Storage_copyFloat(storage, (THFloatStorage *)src);                                       \
    else if ( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )                                         \
      TH##TYPEC##Storage_copyDouble(storage, (THDoubleStorage*)src);                                      \
    else if ( (src = luaT_toudata(L, 2, "torch.GPUStorage")) )                                            \
      TH##TYPEC##Storage_copyGPU(storage, (THGPUStorage *)src);                                           \
    else                                                                                                  \
      luaL_typerror(L, 2, "torch.*Storage");                                                              \
                                                                                                          \
    lua_settop(L, 1);                                                                                     \
    return 1;                                                                                             \
}

GPU_IMPLEMENT_STORAGE_COPY(Byte)
GPU_IMPLEMENT_STORAGE_COPY(Char)
GPU_IMPLEMENT_STORAGE_COPY(Short)
GPU_IMPLEMENT_STORAGE_COPY(Int)
GPU_IMPLEMENT_STORAGE_COPY(Long)
GPU_IMPLEMENT_STORAGE_COPY(Float)
GPU_IMPLEMENT_STORAGE_COPY(Double)

void gputorch_GPUStorage_init(lua_State* L)
{
  /* the standard stuff */
  torch_GPUStorage_init(L);

  /* the copy methods */
  {
    int i;

    const char* tnames[8] = {"torch.ByteStorage",
                             "torch.CharStorage",
                             "torch.ShortStorage",
                             "torch.IntStorage",
                             "torch.LongStorage",
                             "torch.FloatStorage",
                             "torch.DoubleStorage",
                             "torch.GPUStorage"};

    static int (*funcs[8])(lua_State *) = {gputorch_ByteStorage_copy,
                                           gputorch_CharStorage_copy,
                                           gputorch_ShortStorage_copy,
                                           gputorch_IntStorage_copy,
                                           gputorch_LongStorage_copy,
                                           gputorch_FloatStorage_copy,
                                           gputorch_DoubleStorage_copy,
                                           gputorch_GPUStorage_copy
                                          };

    for (i = 0; i < 8; i++)
    {
      luaT_pushmetatable(L, tnames[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}
