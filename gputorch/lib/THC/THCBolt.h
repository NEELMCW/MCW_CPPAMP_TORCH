#ifndef THC_BOLT_INC
#define THC_BOLT_INC

#include "THCTensor.h"
#include "THCGeneral.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/fill.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/transform.h"
#include "bolt/amp/transform_reduce.h"
#include "bolt/amp/reduce.h"
#include "bolt/amp/inner_product.h"
#include "bolt/amp/copy.h"
#include "amp_math.h"

#ifdef DECLARE_BOLT_DEVICE_VECTOR
#undef DECLARE_BOLT_DEVICE_VECTOR
#endif
#define DECLARE_BOLT_DEVICE_VECTOR(THGPUTensor_ptr, dv_a) \
  Concurrency::array_view<float, 1> *pav_##THGPUTensor_ptr = \
    static_cast<Concurrency::array_view<float, 1> *>(THGPUTensor_ptr->storage->allocatorContext);\
  bolt::amp::device_vector<float> dv_a(*pav_##THGPUTensor_ptr, THGPUTensor_nElement(THGPUTensor_ptr), true);

#ifdef DECLARE_BOLT_DEVICE_VECTOR_2
#undef DECLARE_BOLT_DEVICE_VECTOR_2
#endif
#define DECLARE_BOLT_DEVICE_VECTOR_2(host_1, dv_1, host_2, dv_2) \
DECLARE_BOLT_DEVICE_VECTOR(host_1, dv_1); \
DECLARE_BOLT_DEVICE_VECTOR(host_2, dv_2);
  
#ifdef DECLARE_BOLT_DEVICE_VECTOR_3
#undef DECLARE_BOLT_DEVICE_VECTOR_3
#endif
#define DECLARE_BOLT_DEVICE_VECTOR_3(host_1, dv_1, host_2, dv_2, host_3, dv_3) \
DECLARE_BOLT_DEVICE_VECTOR(host_1, dv_1); \
DECLARE_BOLT_DEVICE_VECTOR(host_2, dv_2); \
DECLARE_BOLT_DEVICE_VECTOR(host_3, dv_3);

struct addvalue_functor
{
  const float value;
  addvalue_functor(float value_) restrict(cpu,amp) : value(value_) {}
  float operator()(const float& x) const restrict(cpu,amp)
  {
    return (x+value);
  }
};

struct mse_functor
{
  mse_functor() restrict(amp,cpu) {}
  float operator()(const float& x, const float& y) const restrict(amp,cpu) 
  {
    float z = x-y;
    return z*z;
  }
};

struct mulvalue_functor
{ 
  const float value;
  mulvalue_functor(float value_)restrict(cpu,amp) : value(value_) {}
  float operator()(const float& x) const restrict(cpu,amp)
  { 
    return (x*value);
  }
}; 

struct divvalue_functor
{
  const float value;
  divvalue_functor(float value_)restrict(amp,cpu) : value(value_) {}
  float operator()(const float& x) const restrict(amp,cpu)
  {
    return (x/value);
  }
};

struct pow_functor
{
  const float value;
  pow_functor(float value_) restrict(amp,cpu) : value(value_) {}
  float operator()(const float& x) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::pow(x, value);
    return 0;
  }
};

struct atan2_functor
{
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::atan2f(x, y);
  }
};

struct clamp_functor
{
  const float min_value;
  const float max_value;
  clamp_functor(float min_value_, float max_value_) restrict(amp,cpu): min_value(min_value_), max_value(max_value_) {}
  float operator()(const float& x) const restrict(amp,cpu)
  {
    if (x < min_value)
    {
      return min_value;
    }
    if (x > max_value)
    {
      return max_value;
    }
    return x;
  }
};

struct sign_functor
{
  float operator()(const float &v) const restrict(amp,cpu)
  {
    return (v > 0) - (v < 0);
  }
};

struct dist_functor
{
  const float exponent;
  dist_functor(float exponent_) restrict(amp,cpu) : exponent(exponent_) {}
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return Concurrency::fast_math::pow(Concurrency::fast_math::fabs(x-y), exponent);
  }
};

struct norm_functor
{
  const float exponent;
  norm_functor(float exponent_)restrict(cpu,amp) : exponent(exponent_) {}
  float operator()(const float& x) const restrict(cpu,amp)
  {
    return Concurrency::fast_math::pow(Concurrency::fast_math::fabs(x), exponent);
  }
};

struct partial_not_equal_functor
{
  const float rhs;
  partial_not_equal_functor(float rhs) restrict(cpu,amp) : rhs(rhs) {}
  bool operator()(const float &lhs) const restrict(cpu,amp) {return lhs != rhs;}
};

struct mse_updateGradInput_functor
{
  const float norm;
  mse_updateGradInput_functor(float norm_) restrict(amp,cpu) : norm(norm_) {}
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return norm * (x - y);
  }
};

struct binary_abs_functor
{
  binary_abs_functor() {}
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    float z = x-y;
    return z >= 0 ? z : -z;
  }
};

struct abs_updateGradInput_functor
{
  const float norm;
  abs_updateGradInput_functor(float norm_) restrict(amp,cpu): norm(norm_) {}
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return (x - y) >= 0 ? norm : -norm;
  }
};

struct kl_functor
{
  kl_functor() {}
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return y > 0 ? y * (Concurrency::fast_math::log(y) - x) : 0;
  }
};

struct kl_updateGradInput_functor
{
  const float norm;
  kl_updateGradInput_functor(float norm_) restrict(amp,cpu) : norm(norm_) {}
  float operator()(const float& x, const float& y) const restrict(amp,cpu)
  {
    return y > 0 ? norm * (-y) : 0;
  }
};
#define IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(NAME, CFUNC) \
struct NAME##_functor                                    \
{                                                        \
  float operator()(const float& x) const                 \
  {                                                      \
    return CFUNC(x);                                     \
  }                                                      \
}; 

IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(log, Concurrency::fast_math::log)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(log1p, Concurrency::precise_math::log1p)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(exp, Concurrency::fast_math::exp)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(cos, Concurrency::fast_math::cos)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(acos, Concurrency::fast_math::acos)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(cosh, Concurrency::fast_math::cosh)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(sin, Concurrency::fast_math::sin)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(asin, Concurrency::fast_math::asin)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(sinh, Concurrency::fast_math::sinh)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(tan, Concurrency::fast_math::tan)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(atan, Concurrency::fast_math::atan)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(tanh, Concurrency::fast_math::tanh)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(sqrt, Concurrency::fast_math::sqrt)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(ceil, Concurrency::fast_math::ceil)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(floor, Concurrency::fast_math::floor)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(abs, Concurrency::fast_math::fabs)
IMPLEMENT_GPU_TENSOR_BASIC_FUNCTOR(round, Concurrency::fast_math::roundf)


float boltInnerProduct_plus_mse(THGPUTensor *input, THGPUTensor *target);
float boltInnerProduct_plus_abs(THGPUTensor *input, THGPUTensor *target);
float boltInnerProduct_plus_kl(THGPUTensor *input, THGPUTensor *target);
float boltInnerProduct_plus_dist(THGPUTensor *self, THGPUTensor *src, float value);

float boltTransform_var_all(THGPUTensor *self, float mean);
void boltTransform_clamp(THGPUTensor *src, THGPUTensor *self, float min_value, float max_value);
void boltTransform_mse(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm);
void boltTransform_abs(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm);
void boltTransform_kl(THGPUTensor *input, THGPUTensor *target, THGPUTensor *gradInput,float norm);
void boltTransform_addvalue(THGPUTensor *src, THGPUTensor *self, float value);
void boltTransform_mulvalue(THGPUTensor *src, THGPUTensor *self, float value);
void boltTransform_divvalue(THGPUTensor *src, THGPUTensor *self, float value);
void boltTransform_pow(THGPUTensor *src, THGPUTensor *self, float value);
void boltTransform_log(THGPUTensor *src, THGPUTensor *self);
void boltTransform_log1p(THGPUTensor *src, THGPUTensor *self);
void boltTransform_exp(THGPUTensor *src, THGPUTensor *self);
void boltTransform_cos(THGPUTensor *src, THGPUTensor *self);
void boltTransform_acos(THGPUTensor *src, THGPUTensor *self);
void boltTransform_cosh(THGPUTensor *src, THGPUTensor *self);
void boltTransform_sin(THGPUTensor *src, THGPUTensor *self);
void boltTransform_asin(THGPUTensor *src, THGPUTensor *self);
void boltTransform_sinh(THGPUTensor *src, THGPUTensor *self);
void boltTransform_tan(THGPUTensor *src, THGPUTensor *self);
void boltTransform_atan(THGPUTensor *src, THGPUTensor *self);
void boltTransform_tanh(THGPUTensor *src, THGPUTensor *self);
void boltTransform_sqrt(THGPUTensor *src, THGPUTensor *self);
void boltTransform_ceil(THGPUTensor *src, THGPUTensor *self);
void boltTransform_floor(THGPUTensor *src, THGPUTensor *self);
void boltTransform_abs(THGPUTensor *src, THGPUTensor *self);
void boltTransform_round(THGPUTensor *src, THGPUTensor *self);
void boltTransform_sign(THGPUTensor *src, THGPUTensor *self);

void boltTransformBinary_multiply(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self);
void boltTransformBinary_divide(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self);
void boltTransformBinary_atan2(THGPUTensor *src1, THGPUTensor *src2, THGPUTensor *self);

float boltReduce_minimum(THGPUTensor *self);
float boltReduce_maximum(THGPUTensor *self);
float boltReduce_plus(THGPUTensor *self);
float boltReduce_multiply(THGPUTensor *self);
#endif
