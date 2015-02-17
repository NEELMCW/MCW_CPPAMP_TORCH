#include "THCBolt.h"

float boltInnerProduct_plus_mse(THGPUTensor *input, THGPUTensor *target){
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, target, target_data);
  return  bolt::amp::inner_product(input_data.begin(), input_data.end(), target_data.begin(), (float) 0, bolt::amp::plus<float>(), mse_functor());
}

float boltInnerProduct_plus_abs(THGPUTensor *input, THGPUTensor *target){
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, target, target_data);
  return bolt::amp::inner_product(input_data.begin(), input_data.end(), target_data.begin(), (float) 0, bolt::amp::plus<float>(), binary_abs_functor());
}

float boltInnerProduct_plus_dist(THGPUTensor *self, THGPUTensor *src, float value){
  DECLARE_BOLT_DEVICE_VECTOR_2(self, self_data, src, src_data);
  return bolt::amp::inner_product(self_data.begin(), self_data.end(), src_data.begin(), (float) 0,bolt::amp::plus<float>(), dist_functor(value));

}

float boltInnerProduct_plus_kl(THGPUTensor *input, THGPUTensor *target){
  DECLARE_BOLT_DEVICE_VECTOR_2(input, input_data, target, target_data);
  return bolt::amp::inner_product(input_data.begin(), input_data.end(), target_data.begin(), (float) 0, bolt::amp::plus<float>(), kl_functor());

}

void boltTransform_mse(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm){
  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, target, target_data, gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), target_data.begin(), gradInput_data.begin(), mse_updateGradInput_functor(norm));
}

void boltTransform_abs(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm){
  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, target, target_data, gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), target_data.begin(), gradInput_data.begin(), abs_updateGradInput_functor(norm));
}

void boltTransform_kl(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm){
  DECLARE_BOLT_DEVICE_VECTOR_3(input, input_data, target, target_data, gradInput, gradInput_data);
  bolt::amp::transform(input_data.begin(), input_data.end(), target_data.begin(), gradInput_data.begin(), kl_updateGradInput_functor(norm));
}

float boltTransform_var_all(THGPUTensor *self, float mean){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);
  bolt::amp::device_vector<float> diff(self_data.size());
  bolt::amp::transform(self_data.begin(), self_data.end(), diff.begin(),std::bind2nd(bolt::amp::minus<float>(), mean));

  return bolt::amp::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
}

void boltTransform_addvalue(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), addvalue_functor(value));
}

void boltTransform_mulvalue(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), mulvalue_functor(value));
}

void boltTransform_divvalue(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), divvalue_functor(value)); 
}

void boltTransform_pow(THGPUTensor *src, THGPUTensor *self, float value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), pow_functor(value));
}

void boltTransform_clamp(THGPUTensor *src, THGPUTensor *self, float min_value, float max_value) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), clamp_functor(min_value, max_value));
}

void boltTransform_log(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), log_functor()); 
}

void boltTransform_log1p(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), log1p_functor()); 
}

void boltTransform_exp(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), exp_functor()); 
}

void boltTransform_cos(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), cos_functor()); 
}

void boltTransform_acos(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), acos_functor()); 
}

void boltTransform_cosh(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), cosh_functor()); 
}

void boltTransform_sin(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), sin_functor()); 
}

void boltTransform_asin(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), asin_functor()); 
}

void boltTransform_sinh(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), sinh_functor()); 
}

void boltTransform_tan(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), tan_functor()); 
}

void boltTransform_atan(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), atan_functor()); 
}

void boltTransform_tanh(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), tanh_functor()); 
}

void boltTransform_sqrt(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), sqrt_functor()); 
}

void boltTransform_ceil(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), ceil_functor()); 
}

void boltTransform_floor(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), floor_functor()); 
}

void boltTransform_abs(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), abs_functor()); 
}

void boltTransform_round(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), round_functor()); 
}

void boltTransform_sign(THGPUTensor *src, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_2(self, dest_data, src, src_data);
  bolt::amp::transform(src_data.begin(), src_data.end(), dest_data.begin(), sign_functor()); 
}

void boltTransformBinary_multiply(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_3(src1, Dsrc1_data, src2, Dsrc2_data, self, Dself_data);             
  bolt::amp::transform(Dsrc1_data.begin(), Dsrc1_data.end(), Dsrc2_data.begin(), Dself_data.begin(), bolt::amp::multiplies<float>()); 
}

void boltTransformBinary_divide(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_3(src1, Dsrc1_data, src2, Dsrc2_data, self, Dself_data);             
  bolt::amp::transform(Dsrc1_data.begin(), Dsrc1_data.end(), Dsrc2_data.begin(), Dself_data.begin(), bolt::amp::divides<float>()); 
}

void boltTransformBinary_atan2(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self) {
  DECLARE_BOLT_DEVICE_VECTOR_3(src1, Dsrc1_data, src2, Dsrc2_data, self, Dself_data);             
  bolt::amp::transform(Dsrc1_data.begin(), Dsrc1_data.end(), Dsrc2_data.begin(), Dself_data.begin(), atan2_functor()); 
}

float boltReduce_minimum(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);  
  return bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(THInf), bolt::amp::minimum<float>());
}

float boltReduce_maximum(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);  
  return bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(-THInf), bolt::amp::maximum<float>());
}

float boltReduce_plus(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);  
  return bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(0), bolt::amp::plus<float>());
}

float boltReduce_multiply(THGPUTensor *self){
  DECLARE_BOLT_DEVICE_VECTOR(self, self_data);  
  return bolt::amp::reduce(self_data.begin(), self_data.begin()+THGPUTensor_nElement(self), (float)(1), bolt::amp::multiplies<float>());
}
