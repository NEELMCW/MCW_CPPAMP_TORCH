#include "THC.h"
#include "THFile.h"
#include "luaT.h"

/* everything is as the generic Storage.c, except few things (see below) */

#define real float
#define Real Cuda
#define TH_GENERIC_FILE "generic/Storage.c"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)

#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    float *fdata = (float *)THAlloc(sizeof(float)*size);                         \
    THFile_readFloatRaw(file, fdata, size);                             \
    Concurrency::array_view<float> avData(Concurrency::extent<1>(size),data); \
    Concurrency::array_view<float> avFData(Concurrency::extent<1>(size),fdata); \
    avFData.copy_to(avData); \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    float *fdata = (float *)THAlloc(sizeof(float)*size);                         \
    Concurrency::array_view<float> avData(Concurrency::extent<1>(size),data); \
    Concurrency::array_view<float> avFData(Concurrency::extent<1>(size),fdata); \
    avData.copy_to(avFData); \
    THFile_writeFloatRaw(file, fdata, size);                            \
    THFree(fdata);                                                      \
  }

#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"

#undef real
#undef Real
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to CudaStorage */

#define CUDA_IMPLEMENT_STORAGE_COPY(TYPEC)                              \
  static int cutorch_##TYPEC##Storage_copy(lua_State *L)                \
  {                                                                     \
    TH##TYPEC##Storage *storage = (TH##TYPEC##Storage *)luaT_checkudata(L, 1, "torch." #TYPEC "Storage"); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, "torch." #TYPEC "Storage")) )         \
      TH##TYPEC##Storage_copy(storage, (TH##TYPEC##Storage *)src);                            \
    else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )          \
      TH##TYPEC##Storage_copyByte(storage, (THByteStorage *)src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )          \
      TH##TYPEC##Storage_copyChar(storage, (THCharStorage *)src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )         \
      TH##TYPEC##Storage_copyShort(storage, (THShortStorage *)src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )           \
      TH##TYPEC##Storage_copyInt(storage, (THIntStorage *)src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )          \
      TH##TYPEC##Storage_copyLong(storage, (THLongStorage *)src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )         \
      TH##TYPEC##Storage_copyFloat(storage, (THFloatStorage *)src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )        \
      TH##TYPEC##Storage_copyDouble(storage, (THDoubleStorage*)src);                      \
    else if( (src = luaT_toudata(L, 2, "torch.CudaStorage")) )          \
      TH##TYPEC##Storage_copyCuda(storage, (THCudaStorage *)src);                        \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Storage");                            \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
}

CUDA_IMPLEMENT_STORAGE_COPY(Byte)
CUDA_IMPLEMENT_STORAGE_COPY(Char)
CUDA_IMPLEMENT_STORAGE_COPY(Short)
CUDA_IMPLEMENT_STORAGE_COPY(Int)
CUDA_IMPLEMENT_STORAGE_COPY(Long)
CUDA_IMPLEMENT_STORAGE_COPY(Float)
CUDA_IMPLEMENT_STORAGE_COPY(Double)
CUDA_IMPLEMENT_STORAGE_COPY(Cuda)

void cutorch_CudaStorage_init(lua_State* L)
{
  /* the standard stuff */
  torch_CudaStorage_init(L);

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
                             "torch.CudaStorage"};

    static int (*funcs[8])(lua_State*) = {cutorch_ByteStorage_copy,
                                          cutorch_CharStorage_copy,
                                          cutorch_ShortStorage_copy,
                                          cutorch_IntStorage_copy,
                                          cutorch_LongStorage_copy,
                                          cutorch_FloatStorage_copy,
                                          cutorch_DoubleStorage_copy,
                                          cutorch_CudaStorage_copy};

    for (i = 0; i < 8; i++)
    {
      luaT_pushmetatable(L, tnames[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}
