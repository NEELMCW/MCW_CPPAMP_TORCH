/// Source to build gputorch.test executable
/// The script's location should be set manually or specify $PATH
#include <stdio.h>
#include <iostream>
extern "C"
{
    #include <lua.h>
    #include <lualib.h>
    #include <lauxlib.h>
};

int main()
{
  lua_State *L;
  L = lua_open();
  luaopen_base(L);
  luaopen_table(L);
  luaL_openlibs(L);
  luaopen_string(L);
  luaopen_math(L);

  // FIXME: need to manually specify this path according to your project location
  const char* Script = "/home/neelakandan/Documents/mcw_torch_clamp/gpunn/test/test.lua";
  int ret = luaL_loadfile(L, Script);
  if (ret) {
    std::cout << "Failed to load scipt: "<< Script << "\n" << lua_tostring(L, -1) << std::endl;
    lua_close(L);
    return 1;
  }

  // launch the script
  ret = lua_pcall( L, 0, 0, 0);
  if( ret ) {
    std::cout << "Failed to runscipt: "<< Script << "\n" << lua_tostring(L, -1) << std::endl;
    lua_close(L);
    return 1;
  }

  lua_close(L);
  return 0;
}

