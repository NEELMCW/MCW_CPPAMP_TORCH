#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include "THTensor.h"
#include "THCStorage.h"
#include "THCGeneral.h"

#define TH_TENSOR_REFCOUNTED 1

typedef struct THGPUTensor
{
  long *size;
  long *stride;
  int nDimension;
  THGPUStorage *storage;
  long storageOffset;
  int refcount;
  char flag;
} THGPUTensor;

/**** access methods ****/
THC_API THGPUStorage* THGPUTensor_storage(const THGPUTensor *self);
THC_API long THGPUTensor_storageOffset(const THGPUTensor *self);
THC_API int THGPUTensor_nDimension(const THGPUTensor *self);
THC_API long THGPUTensor_size(const THGPUTensor *self, int dim);
THC_API long THGPUTensor_stride(const THGPUTensor *self, int dim);
THC_API THLongStorage *THGPUTensor_newSizeOf(THGPUTensor *self);
THC_API THLongStorage *THGPUTensor_newStrideOf(THGPUTensor *self);
THC_API float *THGPUTensor_data(const THGPUTensor *self);

THC_API void THGPUTensor_setFlag(THGPUTensor *self, const char flag);
THC_API void THGPUTensor_clearFlag(THGPUTensor *self, const char flag);


/**** creation methods ****/
THC_API THGPUTensor *THGPUTensor_new();
THC_API THGPUTensor *THGPUTensor_newWithTensor(THGPUTensor *tensor);
/* stride might be NULL */
THC_API THGPUTensor *THGPUTensor_newWithStorage(THGPUStorage *storage_, long storageOffset_,
                                                THLongStorage *size_, THLongStorage *stride_);
THC_API THGPUTensor *THGPUTensor_newWithStorage1d(THGPUStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
THC_API THGPUTensor *THGPUTensor_newWithStorage2d(THGPUStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
THC_API THGPUTensor *THGPUTensor_newWithStorage3d(THGPUStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
THC_API THGPUTensor *THGPUTensor_newWithStorage4d(THGPUStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);

/* stride might be NULL */
THC_API THGPUTensor *THGPUTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
THC_API THGPUTensor *THGPUTensor_newWithSize1d(long size0_);
THC_API THGPUTensor *THGPUTensor_newWithSize2d(long size0_, long size1_);
THC_API THGPUTensor *THGPUTensor_newWithSize3d(long size0_, long size1_, long size2_);
THC_API THGPUTensor *THGPUTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

THC_API THGPUTensor *THGPUTensor_newClone(THGPUTensor *self);
THC_API THGPUTensor *THGPUTensor_newContiguous(THGPUTensor *tensor);
THC_API THGPUTensor *THGPUTensor_newSelect(THGPUTensor *tensor, int dimension_, long sliceIndex_);
THC_API THGPUTensor *THGPUTensor_newNarrow(THGPUTensor *tensor, int dimension_, long firstIndex_, long size_);
THC_API THGPUTensor *THGPUTensor_newTranspose(THGPUTensor *tensor, int dimension1_, int dimension2_);
THC_API THGPUTensor *THGPUTensor_newUnfold(THGPUTensor *tensor, int dimension_, long size_, long step_);

THC_API void THGPUTensor_resize(THGPUTensor *tensor, THLongStorage *size, THLongStorage *stride);
THC_API void THGPUTensor_resizeAs(THGPUTensor *tensor, THGPUTensor *src);
THC_API void THGPUTensor_resize1d(THGPUTensor *tensor, long size0_);
THC_API void THGPUTensor_resize2d(THGPUTensor *tensor, long size0_, long size1_);
THC_API void THGPUTensor_resize3d(THGPUTensor *tensor, long size0_, long size1_, long size2_);
THC_API void THGPUTensor_resize4d(THGPUTensor *tensor, long size0_, long size1_, long size2_, long size3_);
THC_API void THGPUTensor_resize5d(THGPUTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

THC_API void THGPUTensor_set(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_setStorage(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                                    THLongStorage *size_, THLongStorage *stride_);
THC_API void THGPUTensor_setStorage1d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
THC_API void THGPUTensor_setStorage2d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
THC_API void THGPUTensor_setStorage3d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
THC_API void THGPUTensor_setStorage4d(THGPUTensor *self, THGPUStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

THC_API void THGPUTensor_narrow(THGPUTensor *self, THGPUTensor *src, int dimension_, long firstIndex_, long size_);
THC_API void THGPUTensor_select(THGPUTensor *self, THGPUTensor *src, int dimension_, long sliceIndex_);
THC_API void THGPUTensor_transpose(THGPUTensor *self, THGPUTensor *src, int dimension1_, int dimension2_);
THC_API void THGPUTensor_unfold(THGPUTensor *self, THGPUTensor *src, int dimension_, long size_, long step_);

THC_API void THGPUTensor_squeeze(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_squeeze1d(THGPUTensor *self, THGPUTensor *src, int dimension_);

THC_API int THGPUTensor_isContiguous(const THGPUTensor *self);
THC_API int THGPUTensor_isSameSizeAs(const THGPUTensor *self, const THGPUTensor *src);
THC_API long THGPUTensor_nElement(const THGPUTensor *self);

THC_API void THGPUTensor_retain(THGPUTensor *self);
THC_API void THGPUTensor_free(THGPUTensor *self);
THC_API void THGPUTensor_freeCopyTo(THGPUTensor *self, THGPUTensor *dst);

/* Slow access methods [check everything] */
THC_API void THGPUTensor_set1d(THGPUTensor *tensor, long x0, float value);
THC_API void THGPUTensor_set2d(THGPUTensor *tensor, long x0, long x1, float value);
THC_API void THGPUTensor_set3d(THGPUTensor *tensor, long x0, long x1, long x2, float value);
THC_API void THGPUTensor_set4d(THGPUTensor *tensor, long x0, long x1, long x2, long x3, float value);

THC_API float THGPUTensor_get1d(const THGPUTensor *tensor, long x0);
THC_API float THGPUTensor_get2d(const THGPUTensor *tensor, long x0, long x1);
THC_API float THGPUTensor_get3d(const THGPUTensor *tensor, long x0, long x1, long x2);
THC_API float THGPUTensor_get4d(const THGPUTensor *tensor, long x0, long x1, long x2, long x3);

#endif
