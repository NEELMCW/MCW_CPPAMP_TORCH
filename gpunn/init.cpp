#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

/*#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>*/

#include "HardTanh.cpp"
#include "Tanh.cpp"
#include "Max.cpp"
#include "Min.cpp"
#include "LogSoftMax.cpp"
#include "SoftMax.cpp"
#include "TemporalConvolution.cpp"
#include "SpatialConvolutionMM.cpp"
#include "SpatialConvolutionMM_BHWD.cpp"
#include "SpatialConvolutionCUDA.cpp"
#include "SpatialSubSampling.cpp"
#include "SpatialMaxPooling.cpp"
#include "SpatialMaxPoolingCUDA.cpp"
#include "Square.cpp"
#include "Sqrt.cpp"
#include "MultiMarginCriterion.cpp"
#include "MSECriterion.cpp"
#include "DistKLDivCriterion.cpp"
#include "Threshold.cpp"
#include "Sigmoid.cpp"
#include "AbsCriterion.cpp"
#include "Abs.cpp"
#include "SoftPlus.cpp"
#include "Exp.cpp"
#include "SpatialUpSamplingNearest.cpp"

LUA_EXTERNC DLL_EXPORT int luaopen_libgpunn(lua_State *L);

int luaopen_libgpunn(lua_State *L)
{
  lua_newtable(L);

  gpunn_Tanh_init(L);
  gpunn_Sigmoid_init(L);
  gpunn_Max_init(L);
  gpunn_Min_init(L);
  gpunn_HardTanh_init(L);
  gpunn_LogSoftMax_init(L);
  gpunn_SoftMax_init(L);
  gpunn_TemporalConvolution_init(L);
  gpunn_SpatialConvolutionCUDA_init(L);
  gpunn_SpatialConvolutionMM_init(L);
  gpunn_SpatialConvolutionMM_BHWD_init(L);
  gpunn_SpatialMaxPooling_init(L);
  gpunn_SpatialMaxPoolingCUDA_init(L);
  gpunn_SpatialSubSampling_init(L);
  gpunn_MultiMarginCriterion_init(L);
  gpunn_Square_init(L);
  gpunn_Sqrt_init(L);
  gpunn_Threshold_init(L);
  gpunn_MSECriterion_init(L);
  gpunn_AbsCriterion_init(L);
  gpunn_DistKLDivCriterion_init(L);
  gpunn_Abs_init(L);
  gpunn_SoftPlus_init(L);
  gpunn_Exp_init(L);
  gpunn_SpatialUpSamplingNearest_init(L);

  return 1;
}
