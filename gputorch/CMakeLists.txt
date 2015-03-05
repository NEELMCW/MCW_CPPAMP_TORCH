CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

# FIXME: Torch package already contains such macros
#SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")

SET(src Storage.cpp init.cpp Tensor.cpp TensorMath.cpp torch/utils.cpp)

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

FOREACH(src_file ${src})
  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
ENDFOREACH()

SET(luasrc init.lua Tensor.lua FFI.lua test/test.lua)

ADD_TORCH_PACKAGE(gputorch "${src}" "${luasrc}")

SET_PROPERTY(TARGET gputorch APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${CLAMP_SHAREDFLAGS}")
TARGET_LINK_LIBRARIES(gputorch luaT THC)

ADD_EXECUTABLE(gputorch.test test/test.cpp)
TARGET_LINK_LIBRARIES(gputorch.test qlua qtlua luajit m)
SET_PROPERTY(TARGET gputorch.test APPEND_STRING PROPERTY LINK_FLAGS " -I/usr/local/include -I./test")
