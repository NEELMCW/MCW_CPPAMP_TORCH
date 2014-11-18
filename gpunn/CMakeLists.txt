CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src-camp init.cpp)

SET (OPENCL_INC "/opt/AMDAPP/include/")
SET(CLAMP_CXXFLAGS "-std=c++amp -I/opt/AMDAPP/include -I/home/neelakandan/Downloads/mcw_cppamp/cppamp-driver-ng/include -I/home/neelakandan/Downloads/mcw_cppamp/cppamp-driver-ng/libc++/libcxx/include")
SET(CLAMP_LDFLAGS  "-std=c++amp -L/home/neelakandan/Downloads/mcw_cppamp/build/build/Release/lib -L/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxx/lib -L/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxxrt/lib -Wl,--rpath=/home/neelakandan/Downloads/mcw_cppamp/build/build/Release/lib:/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxx/lib:/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxxrt/lib -lOpenCL -lc++ -lcxxrt -ldl -lpthread -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive ")


SET_PROPERTY(SOURCE init.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
SET(luasrc init.lua test/test.lua)

ADD_LIBRARY(cunn MODULE ${src-camp})
TARGET_LINK_LIBRARIES(cunn luaT THC TH)

SET_PROPERTY(TARGET cunn APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${CLAMP_SHAREDFLAGS}")
### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(cunn PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")
    
INSTALL(TARGETS cunn
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/cunn")
