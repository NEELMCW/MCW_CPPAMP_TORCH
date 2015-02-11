INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/gpunn-impl/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/gpunn-impl/SpatialConvolutionGPU/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/gpunn-impl/SpatialPoolingGPU/)
INCLUDE_DIRECTORIES($ENV{MCWCPPAMPROOT}/cppamp-driver-ng/include)
SET(src 
   THCGeneral.cpp THCAMPBlasImpl.cpp
   THCBlas.cpp THCStorage.cpp THCTensor.cpp THCTensorCopy.cpp THCTensorConv.cpp THCTensorMath.cpp THCTensorRandom.cpp)
SET(gpunnsrc gpunn-impl/SpatialConvolutionGPU/updateOutput.cpp gpunn-impl/SpatialConvolutionGPU/updateGradInput.cpp
    gpunn-impl/SpatialConvolutionGPU/accGradParameters.cpp  gpunn-impl/init.cpp)

SET(OPENCL_INC "$ENV{AMDAPPSDKROOT}/include")
SET(OPENCL_LIB "$ENV{AMDAPPSDKROOT}/lib/x86_64")
SET(CLBLAS_INC "$ENV{CLBLASROOT}/include")
SET(CLBLAS_LIB "$ENV{CLBLASROOT}/lib64")
SET(BOLT_SRC_INC "$ENV{MCWCPPAMPROOT}/src/Bolt/include/")
SET(BOOST_SRC_INC "$ENV{MCWCPPAMPROOT}/gmac_exp_build_cache/Bolt/superbuild/external/boost/src/Boost/")
SET(BOLT_VER_INC "$ENV{MCWCPPAMPROOT}/gmac_exp_build_cache/Bolt/superbuild/Bolt-build/include/")
SET(PREFIX "$ENV{MCWCPPAMPROOT}/gmac_exp_build_cache")
SET(CLANG_AMP "${PREFIX}/compiler/bin/clang++")
SET(CLAMP_CONFIG "${PREFIX}/build/Release/bin/clamp-config")
EXECUTE_PROCESS(COMMAND ${CLAMP_CONFIG} --build --cxxflags
    OUTPUT_VARIABLE CLAMP_CXXFLAGS)
STRING(STRIP ${CLAMP_CXXFLAGS} CLAMP_CXXFLAGS)
SET(CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS} -I${OPENCL_INC}")
EXECUTE_PROCESS(COMMAND ${CLAMP_CONFIG} --build --ldflags --shared
    OUTPUT_VARIABLE CLAMP_LDFLAGS)
STRING(STRIP ${CLAMP_LDFLAGS} CLAMP_LDFLAGS)
SET(CLAMP_CXXFLAGS "${CLAMP_CXXFLAGS} -I${OPENCL_INC} -I${CLBLAS_INC} -I${BOLT_SRC_INC} -I${BOLT_VER_INC} -I${BOOST_SRC_INC} -Wall")
SET(CLAMP_LDFLAGS "${CLAMP_LDFLAGS} -L${OPENCL_LIB} -L${CLBLAS_LIB} -lclBLAS")

FOREACH(src_file ${src})
  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
ENDFOREACH()

FOREACH(gpunnsrc_file ${gpunnsrc})
  SET_PROPERTY(SOURCE ${gpunnsrc_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${CLAMP_CXXFLAGS}")
ENDFOREACH()

ADD_LIBRARY(THC SHARED ${src} ${gpunnsrc})

TARGET_LINK_LIBRARIES(THC TH)

SET_PROPERTY(TARGET THC APPEND_STRING PROPERTY LINK_FLAGS " ${CLAMP_LDFLAGS} ${CLAMP_SHAREDFLAGS}")

INSTALL(TARGETS THC
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          THC.h
          THCGeneral.h
          THCBlas.h
          THCStorage.h
          THCTensor.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorConv.h
          DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/THC")
