#pragma once

#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
#include "bolt/amp/copy.h"    

#ifdef DECLARE_BOLT_DEVICE_VECTOR
#undef DECLARE_BOLT_DEVICE_VECTOR
#endif
#define DECLARE_BOLT_DEVICE_VECTOR(THGPUTensor_ptr, dv_a) \
  Concurrency::array_view<float, 1> *pav_##THGPUTensor_ptr = \
    static_cast<Concurrency::array_view<float, 1> *>(THGPUTensor_ptr->storage->allocatorContext);\
  bolt::amp::device_vector<float> dv_a(*pav_##THGPUTensor_ptr, THGPUTensor_nElement(THGPUTensor_ptr), true);

#ifdef THGPUTensorMemcpyDeviceToHost
#undef THGPUTensorMemcpyDeviceToHost
#endif
#define THGPUTensorMemcpyDeviceToHost(THGPUTensor_Ptr)\
  Concurrency::array_view<float, 1> *av_##THGPUTensor_Ptr = static_cast<Concurrency::array_view<float, 1> *>(THGPUTensor_Ptr->storage->allocatorContext);\
  Concurrency::copy(*av_##THGPUTensor_Ptr, THGPUTensor_Ptr->storage->data);

#define PREPARE_AV_WITH_STORAGE(Storage, av_ptr) \
  Concurrency::array_view<float, 1> *av_ptr = \
    static_cast<Concurrency::array_view<float, 1> *>(Storage->allocatorContext);\
  av_ptr->discard_data();

#define PREPARE_AV(THGPUTensor_ptr, av_ptr) \
  Concurrency::array_view<float, 1> *av_ptr = \
    static_cast<Concurrency::array_view<float, 1> *>(THGPUTensor_ptr->storage->allocatorContext);\
  av_ptr->discard_data();
