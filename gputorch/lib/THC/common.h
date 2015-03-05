#pragma once

#ifdef THGPUTensorMemcpyDeviceToHost
#undef THGPUTensorMemcpyDeviceToHost
#endif

#define PREPARE_AV_WITH_STORAGE(Storage, av_ptr) \
  Concurrency::array_view<float, 1> *av_ptr = \
    static_cast<Concurrency::array_view<float, 1> *>(Storage->allocatorContext);\
  av_ptr->discard_data();

