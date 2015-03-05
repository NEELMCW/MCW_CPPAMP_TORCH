CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

# Torch package contains the macros
#SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

# Unused
#LIST(APPEND GPU_NVCC_FLAGS "-arch=sm_20")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src init.cpp)


SET (OPENCL_INC "$ENV{AMDAPPSDKROOT}/include")
SET (OPENCL_LIB "$ENV{AMDAPPSDKROOT}/lib/x86_64")
SET (BOLT_SRC_INC "$ENV{MCWCPPAMPROOT}/src/Bolt/include/")
SET (BOOST_SRC_INC "$ENV{MCWCPPAMPROOT}/gmac_exp_build_cache/Bolt/superbuild/external/boost/src/Boost/")
SET (BOLT_VER_INC "$ENV{MCWCPPAMPROOT}/gmac_exp_build_cache/Bolt/superbuild/Bolt-build/include/")
SET (PREFIX "$ENV{MCWCPPAMPROOT}/gmac_exp_build_cache")
SET (CLANG_AMP "${PREFIX}/compiler/bin/clang++")
SET (CLAMP_CONFIG "${PREFIX}/build/Release/bin/clamp-config")
execute_process(COMMAND ${CLAMP_CONFIG} --build --cxxflags
    OUTPUT_VARIABLE CLAMP_CXXFLAGS)
string(STRIP ${CLAMP_CXXFLAGS} CLAMP_CXXFLAGS)
set (CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS} -I${OPENCL_INC}")
execute_process(COMMAND ${CLAMP_CONFIG} --build --ldflags --shared
    OUTPUT_VARIABLE CLAMP_LDFLAGS)
string(STRIP ${CLAMP_LDFLAGS} CLAMP_LDFLAGS)
set (CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS} -I${OPENCL_INC} -I${BOLT_SRC_INC} -I${BOLT_VER_INC} -I${BOOST_SRC_INC} -Wall")
set (CLAMP_LDFLAGS "${CLAMP_LDFLAGS} -L${OPENCL_LIB}")




SET_PROPERTY(SOURCE init.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
SET(luasrc init.lua test/test.lua)
ADD_TORCH_PACKAGE(gpunn "${src}" "${luasrc}")
SET_PROPERTY(TARGET gpunn APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${CLAMP_SHAREDFLAGS}")
TARGET_LINK_LIBRARIES(gpunn luaT THC TH)

ADD_EXECUTABLE(gpunn.test test/test.cpp)
TARGET_LINK_LIBRARIES(gpunn.test luaT THC TH luajit)
SET_PROPERTY(TARGET gpunn.test APPEND_STRING PROPERTY LINK_FLAGS " -I/usr/local/include -I./test")

# get imagenet-barebones folder path
set (fbpath ${CMAKE_CURRENT_SOURCE_DIR}/../imagenet-barebones)
set (gpunnpath ${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS "${fbpath}" AND EXISTS "${gpunnpath}/test/fbtest.cpp")
  ADD_EXECUTABLE(fbtest test/fbtest.cpp)  
  TARGET_LINK_LIBRARIES(fbtest luaT THC TH luajit)
  SET_PROPERTY(TARGET fbtest APPEND_STRING PROPERTY LINK_FLAGS " -I/usr/local/include -I./test")  
  
   GET_TARGET_PROPERTY(FB fbtest LOCATION)
   ADD_CUSTOM_COMMAND( TARGET fbtest
     POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E copy ${FB} ${fbpath}
     COMMAND ${CMAKE_COMMAND} -E remove -f ${FB}
   )
endif()

