#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

extern int open_libgpunn(lua_State*);
LUA_EXTERNC DLL_EXPORT int luaopen_libgpunn(lua_State *L);

int luaopen_libgpunn(lua_State *L)
{
  return open_libgpunn(L);
}
