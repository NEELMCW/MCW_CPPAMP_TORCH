CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)

CMAKE_POLICY(VERSION 2.6)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")

INCLUDE_DIRECTORIES("/opt/clamp/include/")


SET(src Storage.cpp init.cpp Tensor.cpp TensorMath.cpp torch/utils.c)

# MCW CLAMP Specific. Version >= 0.3 is Must
#SET (OPENCL_INC "/usr/local/cuda/include/")
SET (OPENCL_INC "/opt/AMDAPP/include/")
SET(CLAMP_CXXFLAGS "-std=c++amp -I/opt/AMDAPP/include -I/opt/AMDAPP/clBLAS/include -I/home/neelakandan/Downloads/mcw_cppamp/cppamp-driver-ng/include -I/home/neelakandan/Downloads/mcw_cppamp/cppamp-driver-ng/libc++/libcxx/include")
SET(CLAMP_LDFLAGS "-std=c++amp -L/home/neelakandan/Downloads/mcw_cppamp/build/build/lib/ -L/home/neelakandan/Downloads/mcw_cppamp/build/build/Release/lib -L/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxx/lib -L/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxxrt/lib -Wl,--rpath=/home/neelakandan/Downloads/mcw_cppamp/build/build/Release/lib:/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxx/lib:/home/neelakandan/Downloads/mcw_cppamp/build/libc++/libcxxrt/lib -lc++ -lcxxrt -ldl -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive")
SET(CLAMP_SHAREDFLAGS "-shared -fPIC -Wl,-Bsymbolic")

SET_PROPERTY(SOURCE Storage.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
SET_PROPERTY(SOURCE init.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
SET_PROPERTY(SOURCE Tensor.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
SET_PROPERTY(SOURCE TensorMath.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")

SET(luasrc init.lua Tensor.lua FFI.lua test/test.lua)

ADD_TORCH_WRAP(clamptensormathwrap TensorMath.lua)
  
ADD_TORCH_PACKAGE(gputorch "${src}" "${luasrc}")

SET_PROPERTY(TARGET gputorch APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${CLAMP_SHAREDFLAGS}")
TARGET_LINK_LIBRARIES(gputorch luaT THC)
