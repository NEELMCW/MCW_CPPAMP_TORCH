#include "THCBolt.h"

float boltInnerProduct_plus_mse(THGPUTensor *input, THGPUTensor *target){
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, target, target_data);
  return bolt::amp::inner_product(input_data.begin() + input->storageOffset,
                                  input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                                  target_data.begin() + target->storageOffset,
                                  (float) 0, bolt::amp::plus<float>(), mse_functor());
}

float boltInnerProduct_plus_abs(THGPUTensor *input, THGPUTensor *target){
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, target, target_data);
  return bolt::amp::inner_product(input_data.begin() + input->storageOffset,
                                  input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                                  target_data.begin() + target->storageOffset,
                                  (float) 0, bolt::amp::plus<float>(), binary_abs_functor());
}

float boltInnerProduct_plus_dist(THGPUTensor *self, THGPUTensor *src, float value){
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  return bolt::amp::inner_product(self_data.begin() + self->storageOffset,
                                  self_data.begin() + self->storageOffset + size,
                                  src_data.begin() + src->storageOffset,
                                  (float) 0,
                                  bolt::amp::plus<float>(),
                                  dist_functor(value));
}

float boltInnerProduct_plus_kl(THGPUTensor *input, THGPUTensor *target){
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, target, target_data);
  return bolt::amp::inner_product(input_data.begin() + input->storageOffset,
                                  input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                                  target_data.begin() + target->storageOffset,
                                  (float) 0, bolt::amp::plus<float>(), kl_functor());

}

void boltTransform_mse(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm){
  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, target, target_data, gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin() + input->storageOffset,
                       input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                       target_data.begin() + target->storageOffset,
                       gradInput_data.begin() + gradInput->storageOffset,
                       mse_updateGradInput_functor(norm));
}

void boltTransform_abs(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm){
  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, target, target_data, gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin() + input->storageOffset,
                       input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                       target_data.begin() + target->storageOffset,
                       gradInput_data.begin() + gradInput->storageOffset,
                       abs_updateGradInput_functor(norm));
}

void boltTransform_kl(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm){
  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, target, target_data, gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin() + input->storageOffset,
                       input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                       target_data.begin() + target->storageOffset,
                       gradInput_data.begin() + gradInput->storageOffset,
                       kl_updateGradInput_functor(norm));
}

float boltTransform_var_all(THGPUTensor *self, float mean){
  long size = THGPUTensor_nElement(self);
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  bolt::amp::device_vector<float> diff(size);
  bolt::amp::transform(self_data.begin() + self->storageOffset,
                       self_data.begin() + self->storageOffset + size,
                       diff.begin(),
                       std::bind2nd(bolt::amp::minus<float>(), mean));

  return bolt::amp::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
}

void boltTransform_addvalue(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       dest_data.begin() + self->storageOffset,
                       addvalue_functor(value));
}

void boltTransform_mulvalue(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       dest_data.begin() + self->storageOffset,
                       mulvalue_functor(value));
}

void boltTransform_divvalue(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       dest_data.begin() + self->storageOffset,
                       divvalue_functor(value));
}

void boltTransform_pow(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset,
                       pow_functor(value));
}

void boltTransform_clamp(THGPUTensor *src, THGPUTensor *self, float min_value, float max_value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset,
                       clamp_functor(min_value,max_value));
}

void boltTransform_log(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, log_functor());
}

void boltTransform_log1p(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, log1p_functor());
}

void boltTransform_exp(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, exp_functor());
}

void boltTransform_cos(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, cos_functor());
}

void boltTransform_acos(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, acos_functor());
}

void boltTransform_cosh(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, cosh_functor());
}

void boltTransform_sin(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, sin_functor());
}

void boltTransform_asin(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, asin_functor());
}

void boltTransform_sinh(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, sinh_functor());
}

void boltTransform_tan(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, tan_functor());
}

void boltTransform_atan(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, atan_functor());
}

void boltTransform_tanh(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, tanh_functor());
}

void boltTransform_sqrt(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, sqrt_functor());
}

void boltTransform_ceil(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, ceil_functor());
}

void boltTransform_floor(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, floor_functor());
}

void boltTransform_abs(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, abs_functor());
}

void boltTransform_round(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset, round_functor());
}

void boltTransform_sign(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src_data.begin() + src->storageOffset,
                       src_data.begin() + src->storageOffset + size,
                       self_data.begin() + self->storageOffset,
                       sign_functor());
}

void boltTransformBinary_multiply(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_3(src1, src1_data, src2, src2_data, self, self_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src2_data.begin() + src2->storageOffset,
                       src2_data.begin() + src2->storageOffset + size,
                       src1_data.begin() + src1->storageOffset,
                       self_data.begin() + self->storageOffset,
                       bolt::amp::multiplies<float>());
}

void boltTransformBinary_divide(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_3(src1, src1_data, src2, src2_data, self, self_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(src1_data.begin() + src1->storageOffset,
                       src1_data.begin() + src1->storageOffset + size,
                       src2_data.begin() + src2->storageOffset,
                       self_data.begin() + self->storageOffset,
                       bolt::amp::divides<float>());
}

void boltTransformBinary_atan2(THGPUTensor *tx, THGPUTensor *ty, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_3(tx, tx_data, ty, ty_data, self, self_data);
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(tx_data.begin() + tx->storageOffset,
                       tx_data.begin() + tx->storageOffset + size,
                       ty_data.begin() + ty->storageOffset,
                       self_data.begin() + self->storageOffset,
                       atan2_functor());
}

float boltReduce_minimum(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  return bolt::amp::reduce(self_data.begin() + self->storageOffset,
                           self_data.begin() + self->storageOffset + THGPUTensor_nElement(self),
                           (float)(THInf),
                           bolt::amp::minimum<float>());
}

float boltReduce_maximum(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  return bolt::amp::reduce(self_data.begin() + self->storageOffset,
                           self_data.begin() + self->storageOffset + THGPUTensor_nElement(self),
                           (float)(-THInf),
                           bolt::amp::maximum<float>());
}

float boltReduce_plus(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  return bolt::amp::reduce(self_data.begin() + self->storageOffset,
                           self_data.begin() + THGPUTensor_nElement(self),
                           (float)(0),
                           bolt::amp::plus<float>());
}

float boltReduce_multiply(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  return bolt::amp::reduce(self_data.begin() + self->storageOffset,
                           self_data.begin() + self->storageOffset + THGPUTensor_nElement(self),
                           (float)(1),
                           bolt::amp::multiplies<float>());
}
