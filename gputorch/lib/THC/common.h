#pragma once

void MemcpyHostToTHGPUTensor(float* first, int size, void* dest, int offset = 0);
void MemcpyHostToAV(float* first, int size, Concurrency::array_view<float,1> &dest);
void MemcpyAVToAV(void* src, int size, void* dest);
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
