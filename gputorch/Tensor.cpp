#include "THC.h"
#include "THFile.h"
#include "luaT.h"

/* everything is as the generic Storage.c, except few things (see below) */

static void THGPUTensor_maskedFill(THGPUTensor *tensor, THByteTensor *mask, float value)
{
  THError("not yet implemented for CUDA");
}

static void THGPUTensor_maskedCopy(THGPUTensor *tensor, THByteTensor *mask, THGPUTensor* src)
{
  THError("not yet implemented for CUDA");
}

void THGPUTensor_maskedSelect(THGPUTensor *tensor, THGPUTensor* src, THByteTensor *mask)
{
  THError("not yet implemented for CUDA");
}

#define real float
#define Real GPU

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,Real,Tensor_,NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#define TH_GENERIC_FILE "generic/Tensor.c"
#include "generic/Tensor.c"
#undef TH_GENERIC_FILE

#undef real
#undef Real

/* now we overwrite some methods specific to GPUTensor */

#define CUDA_IMPLEMENT_TENSOR_COPY(TYPEC)                               \
  static int gputorch_##TYPEC##Tensor_copy(lua_State *L)                 \
  {                                                                     \
    TH##TYPEC##Tensor *storage = (TH##TYPEC##Tensor *)luaT_checkudata(L, 1, "torch." #TYPEC "Tensor"); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, "torch." #TYPEC "Tensor")) )          \
      TH##TYPEC##Tensor_copy(storage, (TH##TYPEC##Tensor *)src);                             \
    else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )           \
      TH##TYPEC##Tensor_copyByte(storage, (THByteTensor *)src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )           \
      TH##TYPEC##Tensor_copyChar(storage, (THCharTensor *)src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )          \
      TH##TYPEC##Tensor_copyShort(storage, (THShortTensor *)src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )            \
      TH##TYPEC##Tensor_copyInt(storage, (THIntTensor *)src);                          \
    else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )           \
      TH##TYPEC##Tensor_copyLong(storage, (THLongTensor *)src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )          \
      TH##TYPEC##Tensor_copyFloat(storage, (THFloatTensor *)src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )         \
      TH##TYPEC##Tensor_copyDouble(storage, (THDoubleTensor *)src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.GPUTensor")) )           \
      TH##TYPEC##Tensor_copyGPU(storage, (THGPUTensor *)src);                         \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Tensor");                             \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
  }

CUDA_IMPLEMENT_TENSOR_COPY(Byte)
CUDA_IMPLEMENT_TENSOR_COPY(Char)
CUDA_IMPLEMENT_TENSOR_COPY(Short)
CUDA_IMPLEMENT_TENSOR_COPY(Int)
CUDA_IMPLEMENT_TENSOR_COPY(Long)
CUDA_IMPLEMENT_TENSOR_COPY(Float)
CUDA_IMPLEMENT_TENSOR_COPY(Double)
CUDA_IMPLEMENT_TENSOR_COPY(GPU)

static void THFloatTensor_computesz(THFloatTensor *self, long **sz_, long **st_)
{
  long *sz, *st, *szh;
  int i;

  sz = (long *)THAlloc(sizeof(long) * self->nDimension);
  st = (long *)THAlloc(sizeof(long) * self->nDimension);
  szh = (long *)THAlloc(sizeof(long) * self->nDimension);

  for (i = self->nDimension - 1; i >= 0; i--)
  {
    if (i == self->nDimension - 1)
      szh[i] = 1;
    else
      szh[i] = szh[i+1] * self->size[i + 1];
  }

  memcpy(sz, szh, self->nDimension * sizeof(long));
  memcpy(st, self->stride, self->nDimension * sizeof(long));
  THFree(szh);

  *sz_ = sz;
  *st_ = st;
}

void THFloatTensor_kernel_copy(float *dst, long *dst_sz, long *dst_st, int dst_dim, float *src,
                              long *src_sz, long *src_st, int src_dim, long n_elem)
{
  long k;

  for (k = 0; k < n_elem; k++)
  {
    long src_idx = 0;
    long src_rest = k;
    long dst_idx = 0;
    long dst_rest = k;
    int dim;

    for (dim = 0; dim < dst_dim; dim++)
    {
      dst_idx += (dst_rest / dst_sz[dim]) * dst_st[dim];
      dst_rest = dst_rest % dst_sz[dim];
    }

    for (dim = 0; dim < src_dim; dim++)
    {
      src_idx += (src_rest / src_sz[dim]) * src_st[dim];
      src_rest = src_rest % src_sz[dim];
    }

    dst[dst_idx] = src[src_idx];
  }
}

static int gpu_FloatTensor_fakecopy(lua_State *L)
{
  THFloatTensor *self = (THFloatTensor *)luaT_checkudata(L, 1, "torch.FloatTensor");
  THFloatTensor *src = (THFloatTensor *)luaT_checkudata(L, 2, "torch.FloatTensor");
  long *d_self_sz, *d_self_st, *d_src_sz, *d_src_st;
  long nElement = THFloatTensor_nElement(self);

  THArgCheck(THFloatTensor_nElement(self) == THFloatTensor_nElement(src), 2, "sizes do not match"); 

  THFloatTensor_computesz(self, &d_self_sz, &d_self_st);
  THFloatTensor_computesz(src, &d_src_sz, &d_src_st);

  THFloatTensor_kernel_copy(THFloatTensor_data(self), 
                            d_self_sz, d_self_st, self->nDimension,
                            THFloatTensor_data(src),
                            d_src_sz, d_src_st, src->nDimension,
                            nElement);

  THFree(d_self_sz);
  THFree(d_self_st);
  THFree(d_src_sz);
  THFree(d_src_st);

  lua_settop(L, 1);
  return 1;
}

void gputorch_GPUTensor_init(lua_State* L)
{
  /* the standard stuff */
  torch_GPUTensor_init(L);

  /* additional methods */
  luaT_pushmetatable(L, "torch.FloatTensor");
  lua_pushcfunction(L, gpu_FloatTensor_fakecopy);
  lua_setfield(L, -2, "fakecopy");
  lua_pop(L, 1);

  /* the copy methods */
  {
    int i;

    const char* tnames[8] = {"torch.ByteTensor",
                             "torch.CharTensor",
                             "torch.ShortTensor",
                             "torch.IntTensor",
                             "torch.LongTensor",
                             "torch.FloatTensor",
                             "torch.DoubleTensor",
                             "torch.GPUTensor"};

    static int (*funcs[8])(lua_State*) = {gputorch_ByteTensor_copy,
                                          gputorch_CharTensor_copy,
                                          gputorch_ShortTensor_copy,
                                          gputorch_IntTensor_copy,
                                          gputorch_LongTensor_copy,
                                          gputorch_FloatTensor_copy,
                                          gputorch_DoubleTensor_copy,
                                          gputorch_GPUTensor_copy};

    for (i = 0; i < 8; i++)
    {
      luaT_pushmetatable(L, tnames[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}
