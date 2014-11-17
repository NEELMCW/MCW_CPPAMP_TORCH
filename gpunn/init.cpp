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

LUA_EXTERNC DLL_EXPORT int luaopen_libcunn(lua_State *L);

int luaopen_libcunn(lua_State *L)
{
  lua_newtable(L);

  cunn_Tanh_init(L);
  cunn_Sigmoid_init(L);
  cunn_Max_init(L);
  cunn_Min_init(L);
  cunn_HardTanh_init(L);
  cunn_LogSoftMax_init(L);
  cunn_SoftMax_init(L);
  cunn_TemporalConvolution_init(L);
  cunn_SpatialConvolutionCUDA_init(L);
  cunn_SpatialConvolutionMM_init(L);
  cunn_SpatialConvolutionMM_BHWD_init(L);
  cunn_SpatialMaxPooling_init(L);
  cunn_SpatialMaxPoolingCUDA_init(L);
  cunn_SpatialSubSampling_init(L);
  cunn_MultiMarginCriterion_init(L);
  cunn_Square_init(L);
  cunn_Sqrt_init(L);
  cunn_Threshold_init(L);
  cunn_MSECriterion_init(L);
  cunn_AbsCriterion_init(L);
  cunn_DistKLDivCriterion_init(L);
  cunn_Abs_init(L);
  cunn_SoftPlus_init(L);
  cunn_Exp_init(L);
  cunn_SpatialUpSamplingNearest_init(L);

  return 1;
}
