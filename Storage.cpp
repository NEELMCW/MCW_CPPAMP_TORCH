#include "THC.h"
#include "amp.h"

#include "THFile.h"
#include "luaT.h"

/* everything is as the generic Storage.c, except few things (see below) */

#define real float 
#define Real Camp
#define TH_GENERIC_FILE "generic/Storage.cpp"


#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)

#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    float *fdata = (float *)THAlloc(sizeof(float)*size);                \
    THFile_readFloatRaw(file, fdata, size);                             \
    /*THCampCheck(cudaMemcpy(data, fdata, size * sizeof(float), cudaMemcpyHostToDevice));*/ \
    Concurrency::array_view<float> arrSrcData(size,fdata);                 \
    Concurrency::array_view<float> arrDesData(size,data);                 \
    Concurrency::copy(arrSrcData,arrDesData);                          \
    arrDesData.synchronize();                                          \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    float *fdata = (float *)THAlloc(sizeof(float)*size);                \
    /*THCampCheck(cudaMemcpy(fdata, data, size * sizeof(float), cudaMemcpyDeviceToHost));*/ \
    Concurrency::array_view<float> arrSrcData(size,data);                 \
    Concurrency::array_view<float> arrDesData(size,fdata);                 \
    Concurrency::copy(arrSrcData,arrDesData);                          \
    arrDesData.synchronize();                                          \
    THFile_writeFloatRaw(file, fdata, size);                            \
    THFree(fdata);                                                      \
  }

#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.cpp"

#undef real
#undef Real
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to CampStorage */

#define CUDA_IMPLEMENT_STORAGE_COPY(TYPEC)                              \
static int clamptorch_##TYPEC##Storage_copy(lua_State *L)               \
{                                                                       \
    TH##TYPEC##Storage *storage = (TH##TYPEC##Storage *)luaT_checkudata(L, 1, "torch." #TYPEC "Storage"); \
    TH##TYPEC##Storage *src;                                                          \
    if( (src = (TH##TYPEC##Storage *)luaT_toudata(L, 2, "torch." #TYPEC "Storage")) ) \
        TH##TYPEC##Storage_copy(storage, src);                          \
    else                                                                \
        luaL_typerror(L, 2, "torch.*Storage");                          \
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
CUDA_IMPLEMENT_STORAGE_COPY(Camp)

void clamptorch_CampStorage_init(lua_State* L)
{
    /* the standard stuff */
    torch_CampStorage_init(L);

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
                                 "torch.CampStorage"};

        static int (*funcs[8])(lua_State*) = {clamptorch_ByteStorage_copy,
                                              clamptorch_CharStorage_copy,
                                              clamptorch_ShortStorage_copy,
                                              clamptorch_IntStorage_copy,
                                              clamptorch_LongStorage_copy,
                                              clamptorch_FloatStorage_copy,
                                              clamptorch_DoubleStorage_copy,
                                              clamptorch_CampStorage_copy};

        for(i = 0; i < 8; i++)
        {
              luaT_pushmetatable(L, tnames[i]);
              lua_pushcfunction(L, funcs[i]);
              lua_setfield(L, -2, "copy");
              lua_pop(L, 1);
        }
    }
}
