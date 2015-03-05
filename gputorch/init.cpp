#include "utils.h"
#include "luaT.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"

extern void gputorch_GPUStorage_init(lua_State* L);
extern void gputorch_GPUTensor_init(lua_State* L);
extern void gputorch_GPUTensorMath_init(lua_State* L);

static int gputorch_synchronize(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 0;
}

static int gputorch_getDevice(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 1;
}

static int gputorch_deviceReset(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 0;
}

static int gputorch_getDeviceCount(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 1;
}

static int gputorch_setDevice(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 0;
}

#define SET_DEVN_PROP(NAME) \
  lua_pushnumber(L, prop.NAME); \
  lua_setfield(L, -2, #NAME);

static int gputorch_getDeviceProperties(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 1;
}

static int gputorch_seed(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 1;
}

static int gputorch_initialSeed(lua_State *L)
{
  /* TO BE IMPLEMENTED */
  return 1;
}

static int gputorch_manualSeed(lua_State *L)
{
  /* TO BE IMPLEMENTED */
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

  gputorch_GPUStorage_init(L);
  gputorch_GPUTensor_init(L);
  gputorch_GPUTensorMath_init(L);


  return 1;
}
