#include "THCBolt.h"

float boltInnerProduct_plus_mse(THGPUTensor *input, THGPUTensor *target)
{
  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_target_data = target->get_bolt_dev_vec();
  return bolt::amp::inner_product(dv_input_data.begin() + input->storageOffset,
                                  dv_input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                                  dv_target_data.begin() + target->storageOffset,
                                  (float) 0, bolt::amp::plus<float>(), mse_functor());
}

float boltInnerProduct_plus_abs(THGPUTensor *input, THGPUTensor *target)
{
  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_target_data = target->get_bolt_dev_vec();
  return bolt::amp::inner_product(dv_input_data.begin() + input->storageOffset,
                                  dv_input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                                  dv_target_data.begin() + target->storageOffset,
                                  (float) 0, bolt::amp::plus<float>(), binary_abs_functor());
}

float boltInnerProduct_plus_dist(THGPUTensor *self, THGPUTensor *src, float value)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  return bolt::amp::inner_product(dv_self_data.begin() + self->storageOffset,
                                  dv_self_data.begin() + self->storageOffset + size,
                                  dv_src_data.begin() + src->storageOffset,
                                  (float) 0,
                                  bolt::amp::plus<float>(),
                                  dist_functor(value));
}

float boltInnerProduct_plus_kl(THGPUTensor *input, THGPUTensor *target)
{
  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_target_data = target->get_bolt_dev_vec();
  return bolt::amp::inner_product(dv_input_data.begin() + input->storageOffset,
                                  dv_input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                                  dv_target_data.begin() + target->storageOffset,
                                  (float) 0, bolt::amp::plus<float>(), kl_functor());

}

void boltTransform_mse(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm)
{
  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_target_data = target->get_bolt_dev_vec();
  auto dv_gradInput_data = gradInput->get_bolt_dev_vec();
  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                       dv_target_data.begin() + target->storageOffset,
                       dv_gradInput_data.begin() + gradInput->storageOffset,
                       mse_updateGradInput_functor(norm));
}

void boltTransform_abs(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm)
{
  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_target_data = target->get_bolt_dev_vec();
  auto dv_gradInput_data = gradInput->get_bolt_dev_vec();
  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                       dv_target_data.begin() + target->storageOffset,
                       dv_gradInput_data.begin() + gradInput->storageOffset,
                       abs_updateGradInput_functor(norm));
}

void boltTransform_kl(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm)
{
  auto dv_input_data = input->get_bolt_dev_vec();
  auto dv_target_data = target->get_bolt_dev_vec();
  auto dv_gradInput_data = gradInput->get_bolt_dev_vec();
  bolt::amp::transform(dv_input_data.begin() + input->storageOffset,
                       dv_input_data.begin() + input->storageOffset + THGPUTensor_nElement(input),
                       dv_target_data.begin() + target->storageOffset,
                       dv_gradInput_data.begin() + gradInput->storageOffset,
                       kl_updateGradInput_functor(norm));
}

float boltTransform_var_all(THGPUTensor *self, float mean)
{
  long size = THGPUTensor_nElement(self);
  auto dv_self_data = self->get_bolt_dev_vec();
  bolt::amp::device_vector<float> dvdiff(size);
  bolt::amp::transform(dv_self_data.begin() + self->storageOffset,
                       dv_self_data.begin() + self->storageOffset + size,
                       dvdiff.begin(),
                       std::bind2nd(bolt::amp::minus<float>(), mean));

  return bolt::amp::inner_product(dvdiff.begin(), dvdiff.end(), dvdiff.begin(), 0.0);
}

void boltTransform_addvalue(THGPUTensor *src, THGPUTensor *self, float value)
{
  auto dv_dest_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_dest_data.begin() + self->storageOffset,
                       addvalue_functor(value));
}

void boltTransform_mulvalue(THGPUTensor *src, THGPUTensor *self, float value)
{
  auto dv_dest_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_dest_data.begin() + self->storageOffset,
                       mulvalue_functor(value));
}

void boltTransform_divvalue(THGPUTensor *src, THGPUTensor *self, float value)
{
  auto dv_dest_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_dest_data.begin() + self->storageOffset,
                       divvalue_functor(value));
}

void boltTransform_pow(THGPUTensor *src, THGPUTensor *self, float value)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset,
                       pow_functor(value));
}

void boltTransform_clamp(THGPUTensor *src, THGPUTensor *self, float min_value, float max_value)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset,
                       clamp_functor(min_value,max_value));
}

void boltTransform_log(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, log_functor());
}

void boltTransform_log1p(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, log1p_functor());
}

void boltTransform_exp(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, exp_functor());
}

void boltTransform_cos(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, cos_functor());
}

void boltTransform_acos(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, acos_functor());
}

void boltTransform_cosh(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, cosh_functor());
}

void boltTransform_sin(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, sin_functor());
}

void boltTransform_asin(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, asin_functor());
}

void boltTransform_sinh(THGPUTensor *src, THGPUTensor *self) {
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, sinh_functor());
}

void boltTransform_tan(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, tan_functor());
}

void boltTransform_atan(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, atan_functor());
}

void boltTransform_tanh(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, tanh_functor());
}

void boltTransform_sqrt(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, sqrt_functor());
}

void boltTransform_ceil(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, ceil_functor());
}

void boltTransform_floor(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, floor_functor());
}

void boltTransform_abs(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, abs_functor());
}

void boltTransform_round(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset, round_functor());
}

void boltTransform_sign(THGPUTensor *src, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src_data = src->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src_data.begin() + src->storageOffset,
                       dv_src_data.begin() + src->storageOffset + size,
                       dv_self_data.begin() + self->storageOffset,
                       sign_functor());
}

void boltTransformBinary_multiply(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src1_data = src1->get_bolt_dev_vec();
  auto dv_src2_data = src2->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src2_data.begin() + src2->storageOffset,
                       dv_src2_data.begin() + src2->storageOffset + size,
                       dv_src1_data.begin() + src1->storageOffset,
                       dv_self_data.begin() + self->storageOffset,
                       bolt::amp::multiplies<float>());
}

void boltTransformBinary_divide(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  auto dv_src1_data = src1->get_bolt_dev_vec();
  auto dv_src2_data = src2->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_src1_data.begin() + src1->storageOffset,
                       dv_src1_data.begin() + src1->storageOffset + size,
                       dv_src2_data.begin() + src2->storageOffset,
                       dv_self_data.begin() + self->storageOffset,
                       bolt::amp::divides<float>());
}

void boltTransformBinary_atan2(THGPUTensor *tx, THGPUTensor *ty, THGPUTensor *self)
{
  auto dv_tx_data = tx->get_bolt_dev_vec();
  auto dv_ty_data = ty->get_bolt_dev_vec();
  auto dv_self_data = self->get_bolt_dev_vec();
  long size = THGPUTensor_nElement(self);
  bolt::amp::transform(dv_tx_data.begin() + tx->storageOffset,
                       dv_tx_data.begin() + tx->storageOffset + size,
                       dv_ty_data.begin() + ty->storageOffset,
                       dv_self_data.begin() + self->storageOffset,
                       atan2_functor());
}

float boltReduce_minimum(THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  return bolt::amp::reduce(dv_self_data.begin() + self->storageOffset,
                           dv_self_data.begin() + self->storageOffset + THGPUTensor_nElement(self),
                           (float)(THInf),
                           bolt::amp::minimum<float>());
}

float boltReduce_maximum(THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  return bolt::amp::reduce(dv_self_data.begin() + self->storageOffset,
                           dv_self_data.begin() + self->storageOffset + THGPUTensor_nElement(self),
                           (float)(-THInf),
                           bolt::amp::maximum<float>());
}

float boltReduce_plus(THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  return bolt::amp::reduce(dv_self_data.begin() + self->storageOffset,
                           dv_self_data.begin() + THGPUTensor_nElement(self),
                           (float)(0),
                           bolt::amp::plus<float>());
}

float boltReduce_multiply(THGPUTensor *self)
{
  auto dv_self_data = self->get_bolt_dev_vec();
  return bolt::amp::reduce(dv_self_data.begin() + self->storageOffset,
                           dv_self_data.begin() + self->storageOffset + THGPUTensor_nElement(self),
                           (float)(1),
                           bolt::amp::multiplies<float>());
}
