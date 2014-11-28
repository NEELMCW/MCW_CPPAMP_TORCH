#include "luaT.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"

extern void gputorch_GPUStorage_init(lua_State* L);
extern void gputorch_GPUTensor_init(lua_State* L);
extern void gputorch_GPUTensorMath_init(lua_State* L);

static int gputorch_synchronize(lua_State *L)
{
/*  gpuDeviceSynchronize();  */
  return 0;
}

static int gputorch_getDevice(lua_State *L)
{
/*  int device;
  THGPUCheck(gpuGetDevice(&device));
  device++;
  lua_pushnumber(L, device);*/
  return 1;
}

static int gputorch_deviceReset(lua_State *L)
{
/*  THGPUCheck(gpuDeviceReset());*/
  return 0;
}

static int gputorch_getDeviceCount(lua_State *L)
{
/*  int ndevice;
  THGPUCheck(gpuGetDeviceCount(&ndevice));
  lua_pushnumber(L, ndevice);*/
  return 1;
}

static int gputorch_setDevice(lua_State *L)
{
/*  int device = (int)luaL_checknumber(L, 1)-1;
  THGPUCheck(gpuSetDevice(device));
  THCRandom_setGenerator(device);*/
  return 0;
}

#define SET_DEVN_PROP(NAME) \
  lua_pushnumber(L, prop.NAME); \
  lua_setfield(L, -2, #NAME);

static int gputorch_getDeviceProperties(lua_State *L)
{
  /*struct gpuDeviceProp prop;
  int device = (int)luaL_checknumber(L, 1)-1;
  THGPUCheck(gpuGetDeviceProperties(&prop, device));
  lua_newtable(L);
  SET_DEVN_PROP(canMapHostMemory);
  SET_DEVN_PROP(clockRate);
  SET_DEVN_PROP(computeMode);
  SET_DEVN_PROP(deviceOverlap);
  SET_DEVN_PROP(integrated);
  SET_DEVN_PROP(kernelExecTimeoutEnabled);
  SET_DEVN_PROP(major);
  SET_DEVN_PROP(maxThreadsPerBlock);
  SET_DEVN_PROP(memPitch);
  SET_DEVN_PROP(minor);
  SET_DEVN_PROP(multiProcessorCount);
  SET_DEVN_PROP(regsPerBlock);
  SET_DEVN_PROP(sharedMemPerBlock);
  SET_DEVN_PROP(textureAlignment);
  SET_DEVN_PROP(totalConstMem);
  SET_DEVN_PROP(totalGlobalMem);
  SET_DEVN_PROP(warpSize);
  SET_DEVN_PROP(pciBusID);
  SET_DEVN_PROP(pciDeviceID);
  SET_DEVN_PROP(pciDomainID);
  SET_DEVN_PROP(maxTexture1D);
  SET_DEVN_PROP(maxTexture1DLinear);
  
  size_t freeMem;
  THGPUCheck(gpuMemGetInfo (&freeMem, NULL));
  lua_pushnumber(L, freeMem);
  lua_setfield(L, -2, "freeGlobalMem");

  lua_pushstring(L, prop.name);
  lua_setfield(L, -2, "name");
*/
  return 1;
}

static int gputorch_seed(lua_State *L)
{
  /*unsigned long seed = THCRandom_seed();
  lua_pushnumber(L, seed);*/
  return 1;
}

static int gputorch_initialSeed(lua_State *L)
{
/*  unsigned long seed = THCRandom_initialSeed();
  lua_pushnumber(L, seed);*/
  return 1;
}

static int gputorch_manualSeed(lua_State *L)
{
/*  unsigned long seed = luaL_checknumber(L, 1);
  THCRandom_manualSeed(seed);*/
  return 0;
}

static const struct luaL_Reg gputorch_stuff__ [] = {
  {"synchronize", gputorch_synchronize},
  {"getDevice", gputorch_getDevice},
  {"deviceReset", gputorch_deviceReset},
  {"getDeviceCount", gputorch_getDeviceCount},
  {"getDeviceProperties", gputorch_getDeviceProperties},
  {"setDevice", gputorch_setDevice},
  {"seed", gputorch_seed},
  {"initialSeed", gputorch_initialSeed},
  {"manualSeed", gputorch_manualSeed},
  {NULL, NULL}
};

LUA_EXTERNC DLL_EXPORT int luaopen_libgputorch(lua_State *L);

int luaopen_libgputorch(lua_State *L)
{
  lua_newtable(L);
  luaL_register(L, NULL, gputorch_stuff__);

  THGPUInit();

  gputorch_GPUStorage_init(L);
  gputorch_GPUTensor_init(L);
  gputorch_GPUTensorMath_init(L);


  return 1;
}
