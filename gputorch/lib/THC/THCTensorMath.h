#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"
#include "THCTensorRandom.h"

THC_API void THGPUTensor_fill(THGPUTensor *self, float value);
THC_API void THGPUTensor_zero(THGPUTensor *self);

THC_API void THGPUTensor_zeros(THGPUTensor *r_, THLongStorage *size);
THC_API void THGPUTensor_ones(THGPUTensor *r_, THLongStorage *size);
THC_API void THGPUTensor_reshape(THGPUTensor *r_, THGPUTensor *t, THLongStorage *size);
THC_API long THGPUTensor_numel(THGPUTensor *t);

THC_API void THGPUTensor_add(THGPUTensor *self, THGPUTensor *src, float value);
THC_API void THGPUTensor_mul(THGPUTensor *self, THGPUTensor *src, float value);
THC_API void THGPUTensor_div(THGPUTensor *self, THGPUTensor *src, float value);

THC_API void THGPUTensor_cadd(THGPUTensor *self, THGPUTensor *src1, float value, THGPUTensor *src2);
THC_API void THGPUTensor_cadd_tst(THGPUTensor *self, THGPUTensor *src1, float value, THGPUTensor *src2);
THC_API void THGPUTensor_cmul(THGPUTensor *self, THGPUTensor *src1, THGPUTensor *src2);
THC_API void THGPUTensor_cdiv(THGPUTensor *self, THGPUTensor *src1, THGPUTensor *src2);

THC_API void THGPUTensor_addcmul(THGPUTensor *self, THGPUTensor *t, float value, THGPUTensor *src1, THGPUTensor *src2);
THC_API void THGPUTensor_addcdiv(THGPUTensor *self, THGPUTensor *t, float value, THGPUTensor *src1, THGPUTensor *src2);

THC_API float THGPUTensor_dot(THGPUTensor *self, THGPUTensor *src);
  
THC_API float THGPUTensor_minall(THGPUTensor *self);
THC_API float THGPUTensor_maxall(THGPUTensor *self);
THC_API float THGPUTensor_sumall(THGPUTensor *self);
THC_API float THGPUTensor_prodall(THGPUTensor *self);
THC_API void THGPUTensor_min(THGPUTensor *values, THGPUTensor *indices, THGPUTensor *src, long dim);
THC_API void THGPUTensor_max(THGPUTensor *values, THGPUTensor *indices, THGPUTensor *src, long dim);
THC_API void THGPUTensor_sum(THGPUTensor *self, THGPUTensor *src, long dim);
THC_API void THGPUTensor_prod(THGPUTensor *self, THGPUTensor *src, long dim);

THC_API void THGPUTensor_addmv(THGPUTensor *self, float beta, THGPUTensor *t, float alpha, THGPUTensor *mat, THGPUTensor *vec);
THC_API void THGPUTensor_addmm(THGPUTensor *self, float beta, THGPUTensor *t, float alpha, THGPUTensor *mat1, THGPUTensor *mat2);
THC_API void THGPUTensor_addr(THGPUTensor *self, float beta, THGPUTensor *t, float alpha, THGPUTensor *vec1, THGPUTensor *vec2);

THC_API void THGPUTensor_log(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_log1p(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_exp(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_cos(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_acos(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_cosh(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_sin(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_asin(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_sinh(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_tan(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_atan(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_tanh(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_pow(THGPUTensor *self, THGPUTensor *src, float value);
THC_API void THGPUTensor_clamp(THGPUTensor *self, THGPUTensor *src, float min_value, float max_value);
THC_API void THGPUTensor_sqrt(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_ceil(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_floor(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_abs(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_sign(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_round(THGPUTensor *self, THGPUTensor *src);
THC_API void THGPUTensor_atan2(THGPUTensor *r_, THGPUTensor *tx, THGPUTensor *ty);

THC_API void THGPUTensor_ltValue(THGPUTensor *self_, THGPUTensor *src, float value);
THC_API void THGPUTensor_gtValue(THGPUTensor *self_, THGPUTensor *src, float value);
THC_API void THGPUTensor_leValue(THGPUTensor *self_, THGPUTensor *src, float value);
THC_API void THGPUTensor_geValue(THGPUTensor *self_, THGPUTensor *src, float value);
THC_API void THGPUTensor_eqValue(THGPUTensor *self_, THGPUTensor *src, float value);
THC_API void THGPUTensor_neValue(THGPUTensor *self_, THGPUTensor *src, float value);

THC_API void THGPUTensor_ltTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2);
THC_API void THGPUTensor_gtTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2);
THC_API void THGPUTensor_leTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2);
THC_API void THGPUTensor_geTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2);
THC_API void THGPUTensor_eqTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2);
THC_API void THGPUTensor_neTensor(THGPUTensor *self_, THGPUTensor *src1, THGPUTensor *src2);

THC_API float THGPUTensor_meanall(THGPUTensor *self);
THC_API void  THGPUTensor_mean(THGPUTensor *self, THGPUTensor *src, long dim);
THC_API float THGPUTensor_varall(THGPUTensor *self);
THC_API float THGPUTensor_stdall(THGPUTensor *self);
THC_API float THGPUTensor_normall(THGPUTensor *self, float value);
THC_API void  THGPUTensor_norm(THGPUTensor* self, THGPUTensor* src, float value, long dimension);
THC_API void  THGPUTensor_renorm(THGPUTensor* self, THGPUTensor* src, float value, long dimension, float max_norm);
THC_API float THGPUTensor_dist(THGPUTensor *self, THGPUTensor *src, float value);

THC_API void THGPUTensor_rand(THGPURNGState* rng_state,THGPUTensor *r_, THLongStorage *size);
THC_API void THGPUTensor_randn(THGPURNGState* rng_state,THGPUTensor *r_, THLongStorage *size);

THC_API void THGPUTensor_indexCopy(THGPUTensor *res_, int dim, THLongTensor *indices, THGPUTensor *src);
THC_API void THGPUTensor_indexFill(THGPUTensor *tensor, int dim, THLongTensor *index, float val);
THC_API void THGPUTensor_indexSelect(THGPUTensor *tensor, THGPUTensor *src, int dim, THLongTensor *index);


#endif
