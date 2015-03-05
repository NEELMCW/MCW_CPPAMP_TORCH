#pragma once

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

